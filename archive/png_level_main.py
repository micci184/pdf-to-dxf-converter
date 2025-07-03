#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PNG画像レベル品質 PDF to DXF コンバーター
無駄な線を除去し、PNG画像と同等の綺麗さを実現
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pdf2image
from datetime import datetime
from sklearn.cluster import DBSCAN
from enhanced_dxf_writer import EnhancedDXFWriter

# EasyOCRのインポート
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class PNGLevelConverter:
    """PNG画像レベル品質変換システム"""
    
    def __init__(self, pdf_path):
        """初期化"""
        self.pdf_path = pdf_path
        self.images = []
        self.ocr_reader = None
        self.initialize_ocr()
        self.load_pdf()
    
    def initialize_ocr(self):
        """OCR初期化"""
        if not EASYOCR_AVAILABLE:
            return
        
        try:
            print("🔤 OCRモデル初期化中...")
            self.ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
            print("✅ OCRモデル初期化完了")
        except Exception as e:
            print(f"⚠️ OCR初期化エラー: {e}")
            self.ocr_reader = None
    
    def load_pdf(self):
        """高品質PDFロード"""
        try:
            print("📖 高品質PDF読み込み中...")
            # 400dpiで高解像度変換（精度向上）
            images = pdf2image.convert_from_path(self.pdf_path, dpi=400)
            
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
            
            print(f"✅ 高品質PDF読み込み完了: {len(self.images)}ページ (400dpi)")
        except Exception as e:
            raise Exception(f"PDF読み込み失敗: {str(e)}")
    
    def smart_preprocessing(self, image):
        """PNG画質レベル前処理（改良版）"""
        # カラー情報を保持
        original = image.copy()
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 高品質ノイズ除去（改良）
        denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
        
        # 2. ガンマ補正でコントラスト調整
        gamma = 0.9
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, look_up_table)
        
        # 3. CLAHE適用
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gamma_corrected)
        
        # 4. 適応的二値化（パラメータ最適化）
        binary = cv2.adaptiveThreshold(
            clahe_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 7, 2
        )
        
        # 5. 軽微なモルフォロジー処理
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return original, gray, cleaned
    
    def detect_smart_color_lines(self, original_image):
        """賢い色線検出（赤文字対応版）"""
        # HSV変換
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        
        lines_by_color = {}
        
        # 青い線の検出（大幅改良）
        # 複数の青色域を統合
        blue_masks = []
        
        # 標準的な青
        blue_lower1 = np.array([95, 40, 40])
        blue_upper1 = np.array([135, 255, 255])
        blue_masks.append(cv2.inRange(hsv, blue_lower1, blue_upper1))
        
        # 暗い青
        blue_lower2 = np.array([100, 50, 20])
        blue_upper2 = np.array([125, 200, 150])
        blue_masks.append(cv2.inRange(hsv, blue_lower2, blue_upper2))
        
        # BGRでの青検出（重要）
        bgr_blue_lower1 = np.array([80, 30, 30])
        bgr_blue_upper1 = np.array([255, 150, 150])
        blue_masks.append(cv2.inRange(original_image, bgr_blue_lower1, bgr_blue_upper1))
        
        # 薄い青
        bgr_blue_lower2 = np.array([60, 0, 0])
        bgr_blue_upper2 = np.array([255, 80, 80])
        blue_masks.append(cv2.inRange(original_image, bgr_blue_lower2, bgr_blue_upper2))
        
        # 全ての青マスクを統合
        blue_mask = blue_masks[0]
        for mask in blue_masks[1:]:
            blue_mask = cv2.bitwise_or(blue_mask, mask)
        
        # ノイズ除去（穏やか）
        kernel = np.ones((2, 2), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        blue_lines = cv2.HoughLinesP(
            blue_mask, rho=1, theta=np.pi/180, threshold=20,
            minLineLength=15, maxLineGap=20
        )
        
        if blue_lines is not None:
            # 線分のフィルタリング
            filtered_blue = self.filter_meaningful_lines(blue_lines)
            lines_by_color['blue'] = [(x1, y1, x2, y2, 'blue') for x1, y1, x2, y2 in filtered_blue]
        else:
            lines_by_color['blue'] = []
        
        # 緑の線の検出（蛍光緑対応強化）
        green_masks = []
        
        # 標準的な緑
        green_lower1 = np.array([40, 50, 50])
        green_upper1 = np.array([80, 255, 255])
        green_masks.append(cv2.inRange(hsv, green_lower1, green_upper1))
        
        # 蛍光緑（明るい緑）
        green_lower2 = np.array([45, 100, 100])
        green_upper2 = np.array([75, 255, 255])
        green_masks.append(cv2.inRange(hsv, green_lower2, green_upper2))
        
        # BGRでの緑検出
        bgr_green_lower1 = np.array([30, 80, 30])
        bgr_green_upper1 = np.array([150, 255, 150])
        green_masks.append(cv2.inRange(original_image, bgr_green_lower1, bgr_green_upper1))
        
        # 薄い緑
        bgr_green_lower2 = np.array([0, 60, 0])
        bgr_green_upper2 = np.array([80, 255, 80])
        green_masks.append(cv2.inRange(original_image, bgr_green_lower2, bgr_green_upper2))
        
        # 全ての緑マスクを統合
        green_mask = green_masks[0]
        for mask in green_masks[1:]:
            green_mask = cv2.bitwise_or(green_mask, mask)
        
        # ノイズ除去（穏やか）
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        green_lines = cv2.HoughLinesP(
            green_mask, rho=1, theta=np.pi/180, threshold=20,
            minLineLength=15, maxLineGap=20
        )
        
        if green_lines is not None:
            # 線分のフィルタリング
            filtered_green = self.filter_meaningful_lines(green_lines)
            lines_by_color['green'] = [(x1, y1, x2, y2, 'green') for x1, y1, x2, y2 in filtered_green]
        else:
            lines_by_color['green'] = []
        
        # 🔴 赤い線の検出（手書き文字対応）
        red_masks = []
        
        # 標準的な赤（HSV）
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_masks.append(cv2.inRange(hsv, red_lower1, red_upper1))
        
        # 赤の上位域（HSV）
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_masks.append(cv2.inRange(hsv, red_lower2, red_upper2))
        
        # 暗い赤（HSV）
        red_lower3 = np.array([0, 80, 30])
        red_upper3 = np.array([15, 255, 200])
        red_masks.append(cv2.inRange(hsv, red_lower3, red_upper3))
        
        # BGRでの赤検出（重要）
        bgr_red_lower1 = np.array([30, 30, 80])
        bgr_red_upper1 = np.array([150, 150, 255])
        red_masks.append(cv2.inRange(original_image, bgr_red_lower1, bgr_red_upper1))
        
        # 薄い赤
        bgr_red_lower2 = np.array([0, 0, 60])
        bgr_red_upper2 = np.array([80, 80, 255])
        red_masks.append(cv2.inRange(original_image, bgr_red_lower2, bgr_red_upper2))
        
        # 全ての赤マスクを統合
        red_mask = red_masks[0]
        for mask in red_masks[1:]:
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        # ノイズ除去（穏やか）
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        red_lines = cv2.HoughLinesP(
            red_mask, rho=1, theta=np.pi/180, threshold=20,
            minLineLength=15, maxLineGap=15
        )
        
        if red_lines is not None:
            # 線分のフィルタリング（赤は手書きなので緩めに）
            filtered_red = self.filter_meaningful_lines(red_lines)
            lines_by_color['red'] = [(x1, y1, x2, y2, 'red') for x1, y1, x2, y2 in filtered_red]
        else:
            lines_by_color['red'] = []
        
        # デバッグ用マスク保存（改良）
        cv2.imwrite('debug_blue_mask_v2.png', blue_mask)
        cv2.imwrite('debug_green_mask_v2.png', green_mask)
        cv2.imwrite('debug_red_mask_v2.png', red_mask)
        print(f"🔍 デバッグv2: 青・緑・赤マスクを保存 (青:{len(lines_by_color['blue'])}本, 緑:{len(lines_by_color['green'])}本, 赤:{len(lines_by_color['red'])}本)")
        
        return lines_by_color
    
    def filter_meaningful_lines(self, lines):
        """意味のある線のみを抽出（改良版）"""
        if lines is None or len(lines) == 0:
            return []
        
        lines_reshaped = lines.reshape(-1, 4)
        filtered_lines = []
        
        for line in lines_reshaped:
            x1, y1, x2, y2 = line
            
            # 線の長さを計算
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 短すぎる線は除外（閾値を下げる）
            if length < 10:
                continue
            
            # 角度を計算
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # 水平線・垂直線・45度線を優先（許容範囲拡大）
            if (abs(angle) < 10 or abs(angle - 90) < 10 or 
                abs(angle - 180) < 10 or abs(angle + 90) < 10 or
                abs(angle - 45) < 15 or abs(angle + 45) < 15):
                filtered_lines.append(line)
            elif length > 30:  # 長い線は角度に関係なく採用（閾値を下げる）
                filtered_lines.append(line)
        
        return filtered_lines
    
    def extract_main_structure_lines(self, binary):
        """主要構造線のみを抽出（改良版）"""
        # より緩い閾値で主要な線を検出
        lines = cv2.HoughLinesP(
            binary, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=25, maxLineGap=20
        )
        
        if lines is None:
            return []
        
        # 線分の統合とフィルタリング
        filtered_lines = self.filter_meaningful_lines(lines)
        clustered_lines = self.cluster_similar_lines(filtered_lines)
        
        return clustered_lines
    
    def cluster_similar_lines(self, lines):
        """類似線分を統合"""
        if len(lines) == 0:
            return []
        
        # 線分の中点と角度で特徴量を作成
        features = []
        for x1, y1, x2, y2 in lines:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            features.append([mid_x, mid_y, angle, length])
        
        features = np.array(features)
        
        # DBSCAN で類似線分をクラスタリング
        if len(features) > 10:
            clustering = DBSCAN(eps=20, min_samples=2).fit(features[:, :2])
            labels = clustering.labels_
            
            # 各クラスターから代表線分を選択
            clustered_lines = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # ノイズ
                    continue
                
                cluster_indices = np.where(labels == label)[0]
                cluster_lines = [lines[i] for i in cluster_indices]
                
                # 最長の線分を代表として選択
                best_line = max(cluster_lines, key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))
                clustered_lines.append(best_line)
            
            # ノイズ（単独線分）の中で長いものを追加
            noise_indices = np.where(labels == -1)[0]
            for idx in noise_indices:
                line = lines[idx]
                length = np.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2)
                if length > 60:  # 長い単独線分は採用
                    clustered_lines.append(line)
            
            return clustered_lines
        else:
            return lines
    
    def smart_text_recognition(self, image):
        """賢いテキスト認識（赤文字対応版）"""
        if not self.ocr_reader:
            return []
        
        try:
            print("🔤 賢いテキスト認識処理中（赤文字対応）...")
            
            # 全体画像でのOCR
            results = self.ocr_reader.readtext(image)
            
            # 赤い文字領域を特別に強調してOCR
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 赤いテキスト領域の抽出
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # 赤い領域を白く、他を黒くして赤文字を強調
            red_enhanced = cv2.bitwise_and(image, image, mask=red_mask)
            red_gray = cv2.cvtColor(red_enhanced, cv2.COLOR_BGR2GRAY)
            
            # 赤い文字が検出された場合、専用OCR実行
            if cv2.countNonZero(red_mask) > 100:  # 赤い領域がある程度存在する場合
                print("🔴 赤いテキスト領域を検出、専用OCR実行中...")
                red_results = self.ocr_reader.readtext(red_gray)
                results.extend(red_results)
            
            text_regions = []
            seen_texts = set()  # 重複除去用
            
            for (bbox, text, confidence) in results:
                if confidence > 0.4:  # 赤文字は手書きなので信頼度を下げる
                    # 意味のあるテキストのみ
                    cleaned_text = text.strip()
                    if len(cleaned_text) > 0 and not cleaned_text.isspace() and cleaned_text not in seen_texts:
                        seen_texts.add(cleaned_text)
                        
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))
                        
                        # 赤いテキストかどうか判定
                        text_center_x, text_center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        is_red_text = False
                        if 0 <= text_center_y < red_mask.shape[0] and 0 <= text_center_x < red_mask.shape[1]:
                            is_red_text = red_mask[text_center_y, text_center_x] > 0
                        
                        text_regions.append({
                            'bbox': (x1, y1, x2, y2),
                            'text': cleaned_text,
                            'confidence': confidence,
                            'color': 'red' if is_red_text else 'black'
                        })
            
            # デバッグ用: 赤マスクを保存
            cv2.imwrite('debug_red_text_mask.png', red_mask)
            
            red_text_count = sum(1 for t in text_regions if t.get('color') == 'red')
            print(f"✅ 賢いテキスト認識完了: {len(text_regions)}個のテキスト (赤文字: {red_text_count}個)")
            return text_regions
        except Exception as e:
            print(f"⚠️ テキスト認識エラー: {e}")
            return []
    
    def process_pdf_png_level(self):
        """PNG画像レベル品質で処理"""
        try:
            print("🎨 PNG画像レベル品質処理開始")
            
            all_elements = {
                'blue_lines': [],
                'green_lines': [],
                'red_lines': [],
                'main_lines': [],
                'text_regions': []
            }
            
            for i, image in enumerate(self.images):
                print(f"🔍 ページ {i+1} をPNG品質処理中...")
                
                # 1. PNG品質前処理
                original, gray, binary = self.smart_preprocessing(image)
                
                # 2. 賢い色線検出
                color_lines = self.detect_smart_color_lines(original)
                all_elements['blue_lines'].extend(color_lines.get('blue', []))
                all_elements['green_lines'].extend(color_lines.get('green', []))
                all_elements['red_lines'].extend(color_lines.get('red', []))
                
                # 3. 主要構造線抽出
                main_lines = self.extract_main_structure_lines(binary)
                all_elements['main_lines'].extend(main_lines)
                
                # 4. 賢いテキスト認識
                text_regions = self.smart_text_recognition(original)
                all_elements['text_regions'].extend(text_regions)
                
                print(f"✅ ページ {i+1} PNG品質処理完了:")
                print(f"   🔵 青い線: {len(color_lines.get('blue', []))}本")
                print(f"   🟢 緑の線: {len(color_lines.get('green', []))}本")
                print(f"   🔴 赤い線: {len(color_lines.get('red', []))}本")
                print(f"   📏 主要線: {len(main_lines)}本")
                print(f"   📝 テキスト: {len(text_regions)}個")
            
            return all_elements
            
        except Exception as e:
            print(f"❌ 処理エラー: {str(e)}")
            return None


def convert_pdf_png_level(input_path, output_path, scale=100):
    """PNG画像レベル品質PDF変換"""
    try:
        print("=" * 60)
        print("🎨 PNG画像レベル品質 PDF変換ツール 🎨")
        print("=" * 60)
        
        # 1. 変換システム初期化
        converter = PNGLevelConverter(input_path)
        
        # 2. PNG品質処理
        elements = converter.process_pdf_png_level()
        
        if elements is None:
            return False
        
        # 3. DXF生成
        print("📐 PNG品質DXF生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        # PNG品質要素をDXFに追加
        dxf_elements = {
            'lines': elements['main_lines'],
            'blue_lines': elements['blue_lines'],
            'green_lines': elements['green_lines'],
            'red_lines': elements['red_lines'],
            'text_regions': elements['text_regions']
        }
        
        dxf_writer.add_elements(dxf_elements)
        dxf_writer.save(output_path)
        
        # 4. 結果表示
        total_lines = len(elements['main_lines']) + len(elements['blue_lines']) + len(elements['green_lines']) + len(elements['red_lines'])
        
        print("=" * 60)
        print("🎉 PNG画像レベル品質変換完了 🎉")
        print("=" * 60)
        print(f"📁 入力: {input_path}")
        print(f"📁 出力: {output_path}")
        print(f"📊 PNG品質検出結果:")
        print(f"   🔵 青い線: {len(elements['blue_lines'])}本")
        print(f"   🟢 緑の線: {len(elements['green_lines'])}本")
        print(f"   🔴 赤い線: {len(elements['red_lines'])}本")
        print(f"   📏 主要線: {len(elements['main_lines'])}本")
        print(f"   📝 テキスト: {len(elements['text_regions'])}個")
        print(f"   🎯 総線分数: {total_lines}本 (最適化済み)")
        print("🎨 PNG画像レベルの綺麗さで変換完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🎨 PNG画像レベル品質 PDF変換ツール 🎨')
    parser.add_argument('--input', '-i', required=True, help='入力PDFファイル')
    parser.add_argument('--output', '-o', help='出力DXFファイル')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール')
    
    args = parser.parse_args()
    
    # 出力ファイル名の生成
    if args.output:
        output_path = f"output/{args.output}"
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")
        output_path = f"output/{timestamp}_{base_name}_png_level.dxf"
    
    # 変換実行
    success = convert_pdf_png_level(
        args.input,
        output_path,
        args.scale
    )
    
    if success:
        print("✅ PNG画像レベル品質変換が正常に完了しました！")
        sys.exit(0)
    else:
        print("❌ 変換に失敗しました。")
        sys.exit(1)
