#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超高精度手書き図面専用 PDF to DXF コンバーター
手書き図面の特性を考慮した最高精度変換システム
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


class UltraHighPrecisionConverter:
    """超高精度手書き図面専用変換システム"""
    
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
            print("🔤 高精度OCRモデル初期化中...")
            self.ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
            print("✅ 高精度OCRモデル初期化完了")
        except Exception as e:
            print(f"⚠️ OCR初期化エラー: {e}")
            self.ocr_reader = None
    
    def load_pdf(self):
        """超高解像度PDFロード（手書き図面専用）"""
        try:
            print("📖 超高解像度PDF読み込み中（手書き図面専用）...")
            # 手書き図面に最適な600dpi設定
            images = pdf2image.convert_from_path(self.pdf_path, dpi=600)
            
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
            
            print(f"✅ 超高解像度PDF読み込み完了: {len(self.images)}ページ (600dpi)")
        except Exception as e:
            raise Exception(f"PDF読み込み失敗: {str(e)}")
    
    def advanced_preprocessing(self, image):
        """高度な手書き図面専用前処理"""
        # カラー情報を保持
        original = image.copy()
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 非局所平均デノイジング（手書きノイズ対応）
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 2. ガンマ補正（コントラスト強化）
        gamma = 0.8  # 手書き図面に最適
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, look_up_table)
        
        # 3. CLAHE（局所コントラスト強化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gamma_corrected)
        
        # 4. エッジ保持平滑化
        bilateral = cv2.bilateralFilter(clahe_image, 9, 75, 75)
        
        # 5. 手書き図面専用適応的二値化
        binary = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 5, 3
        )
        
        # 6. 軽微なモルフォロジー処理（手書き線を保護）
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return original, gray, binary
    
    def comprehensive_color_detection(self, original_image):
        """包括的色線検出（手書き図面専用）"""
        # HSVとLABの両方を使用
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
        
        lines_by_color = {}
        
        # 🔵 青い線の包括的検出
        blue_masks = []
        
        # HSVでの青検出（範囲拡大）
        blue_hsv_ranges = [
            ([90, 30, 30], [140, 255, 255]),    # 標準青
            ([80, 20, 20], [150, 255, 255]),    # 範囲拡大青
            ([95, 40, 40], [135, 200, 200]),    # 暗い青
            ([100, 60, 60], [130, 255, 255])    # 鮮やかな青
        ]
        
        for lower, upper in blue_hsv_ranges:
            blue_masks.append(cv2.inRange(hsv, np.array(lower), np.array(upper)))
        
        # BGRでの青検出（範囲拡大）
        bgr_blue_ranges = [
            ([60, 0, 0], [255, 120, 120]),      # 基本青
            ([40, 0, 0], [255, 100, 100]),      # 薄い青
            ([80, 20, 20], [255, 150, 150]),    # 濃い青
            ([50, 10, 10], [255, 130, 130])     # 中間青
        ]
        
        for lower, upper in bgr_blue_ranges:
            blue_masks.append(cv2.inRange(original_image, np.array(lower), np.array(upper)))
        
        # LABでの青検出
        lab_blue_lower = np.array([0, 120, 0])
        lab_blue_upper = np.array([255, 255, 120])
        blue_masks.append(cv2.inRange(lab, lab_blue_lower, lab_blue_upper))
        
        # 全ての青マスクを統合
        blue_mask = blue_masks[0]
        for mask in blue_masks[1:]:
            blue_mask = cv2.bitwise_or(blue_mask, mask)
        
        # ノイズ除去（軽微）
        kernel_light = np.ones((1, 1), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_light)
        
        blue_lines = cv2.HoughLinesP(
            blue_mask, rho=1, theta=np.pi/180, threshold=15,
            minLineLength=10, maxLineGap=15
        )
        
        if blue_lines is not None:
            filtered_blue = self.smart_line_filtering(blue_lines)
            lines_by_color['blue'] = [(x1, y1, x2, y2, 'blue') for x1, y1, x2, y2 in filtered_blue]
        else:
            lines_by_color['blue'] = []
        
        # 🟢 緑の線の包括的検出（蛍光緑重視）
        green_masks = []
        
        # HSVでの緑検出（蛍光緑対応）
        green_hsv_ranges = [
            ([30, 40, 40], [90, 255, 255]),     # 基本緑
            ([35, 60, 60], [85, 255, 255]),     # 鮮やかな緑
            ([40, 80, 80], [80, 255, 255]),     # 蛍光緑
            ([25, 30, 30], [95, 255, 255])      # 範囲拡大緑
        ]
        
        for lower, upper in green_hsv_ranges:
            green_masks.append(cv2.inRange(hsv, np.array(lower), np.array(upper)))
        
        # BGRでの緑検出
        bgr_green_ranges = [
            ([0, 60, 0], [120, 255, 120]),      # 基本緑
            ([0, 80, 0], [150, 255, 150]),      # 鮮やかな緑
            ([0, 40, 0], [100, 255, 100]),      # 薄い緑
            ([0, 100, 0], [180, 255, 180])      # 濃い緑
        ]
        
        for lower, upper in bgr_green_ranges:
            green_masks.append(cv2.inRange(original_image, np.array(lower), np.array(upper)))
        
        # LABでの緑検出
        lab_green_lower = np.array([0, 0, 130])
        lab_green_upper = np.array([255, 120, 255])
        green_masks.append(cv2.inRange(lab, lab_green_lower, lab_green_upper))
        
        # 全ての緑マスクを統合
        green_mask = green_masks[0]
        for mask in green_masks[1:]:
            green_mask = cv2.bitwise_or(green_mask, mask)
        
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_light)
        
        green_lines = cv2.HoughLinesP(
            green_mask, rho=1, theta=np.pi/180, threshold=15,
            minLineLength=10, maxLineGap=15
        )
        
        if green_lines is not None:
            filtered_green = self.smart_line_filtering(green_lines)
            lines_by_color['green'] = [(x1, y1, x2, y2, 'green') for x1, y1, x2, y2 in filtered_green]
        else:
            lines_by_color['green'] = []
        
        # 🔴 赤い線の包括的検出（手書き文字対応）
        red_masks = []
        
        # HSVでの赤検出（手書き対応）
        red_hsv_ranges = [
            ([0, 40, 40], [15, 255, 255]),      # 明るい赤
            ([165, 40, 40], [180, 255, 255]),   # 赤の上位域
            ([0, 30, 30], [20, 255, 255]),      # 薄い赤
            ([160, 30, 30], [180, 255, 255])    # 暗い赤
        ]
        
        for lower, upper in red_hsv_ranges:
            red_masks.append(cv2.inRange(hsv, np.array(lower), np.array(upper)))
        
        # BGRでの赤検出
        bgr_red_ranges = [
            ([0, 0, 60], [120, 120, 255]),      # 基本赤
            ([0, 0, 40], [100, 100, 255]),      # 薄い赤
            ([0, 0, 80], [150, 150, 255]),      # 濃い赤
            ([0, 0, 50], [130, 130, 255])       # 中間赤
        ]
        
        for lower, upper in bgr_red_ranges:
            red_masks.append(cv2.inRange(original_image, np.array(lower), np.array(upper)))
        
        # LABでの赤検出
        lab_red_lower = np.array([0, 130, 130])
        lab_red_upper = np.array([255, 255, 255])
        red_masks.append(cv2.inRange(lab, lab_red_lower, lab_red_upper))
        
        # 全ての赤マスクを統合
        red_mask = red_masks[0]
        for mask in red_masks[1:]:
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_light)
        
        red_lines = cv2.HoughLinesP(
            red_mask, rho=1, theta=np.pi/180, threshold=10,
            minLineLength=8, maxLineGap=20
        )
        
        if red_lines is not None:
            filtered_red = self.smart_line_filtering(red_lines)
            lines_by_color['red'] = [(x1, y1, x2, y2, 'red') for x1, y1, x2, y2 in filtered_red]
        else:
            lines_by_color['red'] = []
        
        # デバッグ用マスク保存
        cv2.imwrite('debug_ultra_blue_mask.png', blue_mask)
        cv2.imwrite('debug_ultra_green_mask.png', green_mask)
        cv2.imwrite('debug_ultra_red_mask.png', red_mask)
        
        print(f"🔍 包括的色線検出: 青={len(lines_by_color['blue'])}本, "
              f"緑={len(lines_by_color['green'])}本, 赤={len(lines_by_color['red'])}本")
        
        return lines_by_color
    
    def smart_line_filtering(self, lines):
        """賢い線分フィルタリング（手書き図面専用）"""
        if lines is None or len(lines) == 0:
            return []
        
        lines_reshaped = lines.reshape(-1, 4)
        filtered_lines = []
        
        for line in lines_reshaped:
            x1, y1, x2, y2 = line
            
            # 線の長さと角度を計算
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # 手書き図面に適した緩い条件
            if length >= 5:  # 5ピクセル以上の線分を採用
                # 水平・垂直・斜め線を幅広く受け入れ
                if (abs(angle) < 15 or abs(angle - 90) < 15 or 
                    abs(angle - 180) < 15 or abs(angle + 90) < 15 or
                    abs(angle - 45) < 20 or abs(angle + 45) < 20 or
                    length > 25):  # 長い線は角度に関係なく採用
                    filtered_lines.append(line)
        
        return filtered_lines
    
    def extract_structure_lines(self, binary):
        """構造線抽出（手書き図面最適化）"""
        # 複数の閾値で線分検出を実行
        all_lines = []
        
        # 厳格な検出
        lines_strict = cv2.HoughLinesP(
            binary, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        if lines_strict is not None:
            all_lines.extend(lines_strict.reshape(-1, 4))
        
        # 中程度の検出
        lines_moderate = cv2.HoughLinesP(
            binary, rho=1, theta=np.pi/180, threshold=30,
            minLineLength=20, maxLineGap=15
        )
        if lines_moderate is not None:
            all_lines.extend(lines_moderate.reshape(-1, 4))
        
        # 緩い検出（細かい線対応）
        lines_loose = cv2.HoughLinesP(
            binary, rho=1, theta=np.pi/180, threshold=15,
            minLineLength=10, maxLineGap=20
        )
        if lines_loose is not None:
            all_lines.extend(lines_loose.reshape(-1, 4))
        
        if not all_lines:
            return []
        
        # 重複除去とクラスタリング
        unique_lines = self.remove_duplicate_lines(all_lines)
        
        return unique_lines
    
    def remove_duplicate_lines(self, lines):
        """重複線分の除去（改良版）"""
        if len(lines) == 0:
            return []
        
        # 線分の特徴量を作成（中点、角度、長さ）
        features = []
        for x1, y1, x2, y2 in lines:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            features.append([mid_x, mid_y, angle * 100, length])  # 角度に重みを付加
        
        features = np.array(features)
        
        # DBSCANで類似線分をクラスタリング
        if len(features) > 1:
            clustering = DBSCAN(eps=25, min_samples=1).fit(features)
            labels = clustering.labels_
            
            # 各クラスターから最適な線分を選択
            unique_lines = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                cluster_lines = [lines[i] for i in cluster_indices]
                
                if len(cluster_lines) == 1:
                    unique_lines.extend(cluster_lines)
                else:
                    # 最長の線分を選択
                    best_line = max(cluster_lines, 
                                  key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))
                    unique_lines.append(best_line)
            
            return unique_lines
        else:
            return lines
    
    def enhanced_text_recognition(self, image):
        """強化テキスト認識（手書き特化）"""
        if not self.ocr_reader:
            return []
        
        try:
            print("🔤 手書き特化テキスト認識処理中...")
            
            # 複数の前処理でOCR実行
            text_regions = []
            seen_texts = set()
            
            # 1. オリジナル画像でOCR
            results_original = self.ocr_reader.readtext(image)
            
            # 2. コントラスト強化画像でOCR
            enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
            results_enhanced = self.ocr_reader.readtext(enhanced)
            
            # 3. ガンマ補正画像でOCR
            gamma = 0.7
            look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(image, look_up_table)
            results_gamma = self.ocr_reader.readtext(gamma_corrected)
            
            # 全ての結果を統合
            all_results = results_original + results_enhanced + results_gamma
            
            # 色別テキスト検出
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 赤いテキストの特別処理
            red_lower1 = np.array([0, 30, 30])
            red_upper1 = np.array([15, 255, 255])
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([165, 30, 30])
            red_upper2 = np.array([180, 255, 255])
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            for (bbox, text, confidence) in all_results:
                if confidence > 0.3:  # 手書きなので信頼度を下げる
                    cleaned_text = text.strip()
                    if (len(cleaned_text) > 0 and not cleaned_text.isspace() 
                        and cleaned_text not in seen_texts):
                        seen_texts.add(cleaned_text)
                        
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))
                        
                        # 色判定
                        text_center_x = (x1 + x2) // 2
                        text_center_y = (y1 + y2) // 2
                        is_red_text = False
                        
                        if (0 <= text_center_y < red_mask.shape[0] and 
                            0 <= text_center_x < red_mask.shape[1]):
                            is_red_text = red_mask[text_center_y, text_center_x] > 0
                        
                        text_regions.append({
                            'bbox': (x1, y1, x2, y2),
                            'text': cleaned_text,
                            'confidence': confidence,
                            'color': 'red' if is_red_text else 'black'
                        })
            
            red_text_count = sum(1 for t in text_regions if t.get('color') == 'red')
            print(f"✅ 手書き特化テキスト認識完了: {len(text_regions)}個 (赤文字: {red_text_count}個)")
            return text_regions
            
        except Exception as e:
            print(f"⚠️ テキスト認識エラー: {e}")
            return []
    
    def process_pdf_ultra_precision(self):
        """超高精度PDFプロセシング"""
        try:
            print("🚀 超高精度手書き図面処理開始")
            
            all_elements = {
                'blue_lines': [],
                'green_lines': [],
                'red_lines': [],
                'main_lines': [],
                'text_regions': []
            }
            
            for i, image in enumerate(self.images):
                print(f"🔍 ページ {i+1} を超高精度処理中...")
                
                # 1. 高度な前処理
                original, gray, binary = self.advanced_preprocessing(image)
                
                # 2. 包括的色線検出
                color_lines = self.comprehensive_color_detection(original)
                all_elements['blue_lines'].extend(color_lines.get('blue', []))
                all_elements['green_lines'].extend(color_lines.get('green', []))
                all_elements['red_lines'].extend(color_lines.get('red', []))
                
                # 3. 構造線抽出
                main_lines = self.extract_structure_lines(binary)
                all_elements['main_lines'].extend(main_lines)
                
                # 4. 強化テキスト認識
                text_regions = self.enhanced_text_recognition(original)
                all_elements['text_regions'].extend(text_regions)
                
                print(f"✅ ページ {i+1} 超高精度処理完了:")
                print(f"   🔵 青線: {len(color_lines.get('blue', []))}本")
                print(f"   🟢 緑線: {len(color_lines.get('green', []))}本")
                print(f"   🔴 赤線: {len(color_lines.get('red', []))}本")
                print(f"   📏 主要線: {len(main_lines)}本")
                print(f"   📝 テキスト: {len(text_regions)}個")
            
            return all_elements
            
        except Exception as e:
            print(f"❌ 処理エラー: {str(e)}")
            return None
    
    def create_visualization(self, elements, output_path):
        """可視化画像作成"""
        try:
            if not self.images:
                return
            
            # 最初のページをベースに可視化
            base_image = self.images[0].copy()
            height, width = base_image.shape[:2]
            
            # 白い背景を作成
            vis_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # 元の画像を薄く重ねる
            alpha = 0.3
            vis_image = cv2.addWeighted(vis_image, 1-alpha, base_image, alpha, 0)
            
            # 線分を描画
            # 黒い線（主要構造線）
            for line in elements['main_lines']:
                x1, y1, x2, y2 = line[:4]
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
            
            # 青い線
            for line in elements['blue_lines']:
                x1, y1, x2, y2 = line[:4]
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # 緑の線
            for line in elements['green_lines']:
                x1, y1, x2, y2 = line[:4]
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 赤い線
            for line in elements['red_lines']:
                x1, y1, x2, y2 = line[:4]
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # テキスト領域を描画
            for text_region in elements['text_regions']:
                x1, y1, x2, y2 = text_region['bbox']
                color = (0, 0, 255) if text_region.get('color') == 'red' else (255, 0, 255)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 1)
                
                # テキストを描画
                cv2.putText(vis_image, text_region['text'], (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 可視化画像を保存
            cv2.imwrite(output_path, vis_image)
            print(f"📸 可視化画像を保存: {output_path}")
            
        except Exception as e:
            print(f"⚠️ 可視化作成エラー: {e}")


def convert_pdf_ultra_precision(input_path, output_path, scale=100, visualization=True):
    """超高精度PDF変換"""
    try:
        print("=" * 70)
        print("🚀 超高精度手書き図面専用 PDF to DXF 変換ツール 🚀")
        print("=" * 70)
        
        # 1. 変換システム初期化
        converter = UltraHighPrecisionConverter(input_path)
        
        # 2. 超高精度処理
        elements = converter.process_pdf_ultra_precision()
        
        if elements is None:
            return False
        
        # 3. DXF生成
        print("📐 超高精度DXF生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        # 超高精度要素をDXFに追加
        dxf_elements = {
            'lines': elements['main_lines'],
            'blue_lines': elements['blue_lines'],
            'green_lines': elements['green_lines'],
            'red_lines': elements['red_lines'],
            'text_regions': elements['text_regions']
        }
        
        dxf_writer.add_elements(dxf_elements)
        dxf_writer.save(output_path)
        
        # 4. 可視化画像生成
        if visualization:
            vis_path = output_path.replace('.dxf', '_ultra_precision_vis.png')
            converter.create_visualization(elements, vis_path)
        
        # 5. 結果表示
        total_lines = (len(elements['main_lines']) + len(elements['blue_lines']) + 
                      len(elements['green_lines']) + len(elements['red_lines']))
        
        print("=" * 70)
        print("🎉 超高精度変換完了 🎉")
        print("=" * 70)
        print(f"📁 入力: {input_path}")
        print(f"📁 出力: {output_path}")
        print(f"📊 超高精度検出結果:")
        print(f"   🔵 青線: {len(elements['blue_lines'])}本")
        print(f"   🟢 緑線: {len(elements['green_lines'])}本")
        print(f"   🔴 赤線: {len(elements['red_lines'])}本")
        print(f"   📏 主要線: {len(elements['main_lines'])}本")
        print(f"   📝 テキスト: {len(elements['text_regions'])}個")
        print(f"   🎯 総線分数: {total_lines}本")
        print("🚀 手書き図面に特化した超高精度変換完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🚀 超高精度手書き図面専用 PDF変換ツール 🚀')
    parser.add_argument('--input', '-i', required=True, help='入力PDFファイル')
    parser.add_argument('--output', '-o', help='出力DXFファイル')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール')
    parser.add_argument('--visualization', '-v', action='store_true', help='可視化画像を生成')
    
    args = parser.parse_args()
    
    # 出力ファイル名の生成
    if args.output:
        output_path = f"output/{args.output}"
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")
        output_path = f"output/{timestamp}_{base_name}_ultra_precision.dxf"
    
    # 変換実行
    success = convert_pdf_ultra_precision(
        args.input,
        output_path,
        args.scale,
        args.visualization
    )
    
    if success:
        print("✅ 超高精度変換が正常に完了しました！")
        sys.exit(0)
    else:
        print("❌ 変換に失敗しました。")
        sys.exit(1)
