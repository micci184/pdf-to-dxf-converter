#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最適化超高精度 PDF to DXF コンバーター
超高精度検出 + 重複除去最適化 + 実用性重視
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


class OptimizedUltraConverter:
    """最適化超高精度変換システム"""
    
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
            print("🔤 最適化OCRモデル初期化中...")
            self.ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
            print("✅ 最適化OCRモデル初期化完了")
        except Exception as e:
            print(f"⚠️ OCR初期化エラー: {e}")
            self.ocr_reader = None
    
    def load_pdf(self):
        """最適化高解像度PDFロード"""
        try:
            print("📖 最適化高解像度PDF読み込み中...")
            # 500dpiで品質と処理速度のバランスを重視
            images = pdf2image.convert_from_path(self.pdf_path, dpi=500)
            
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
            
            print(f"✅ 最適化高解像度PDF読み込み完了: {len(self.images)}ページ (500dpi)")
        except Exception as e:
            raise Exception(f"PDF読み込み失敗: {str(e)}")
    
    def optimized_preprocessing(self, image):
        """最適化前処理"""
        original = image.copy()
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 非局所平均デノイジング
        denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
        
        # 2. ガンマ補正
        gamma = 0.85
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, look_up_table)
        
        # 3. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gamma_corrected)
        
        # 4. 適応的二値化
        binary = cv2.adaptiveThreshold(
            clahe_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 5, 2
        )
        
        return original, gray, binary
    
    def optimized_color_detection(self, original_image):
        """最適化色線検出"""
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        lines_by_color = {'blue': [], 'green': [], 'red': []}
        
        # 青線検出（最適化）
        blue_masks = []
        blue_hsv_ranges = [
            ([90, 30, 30], [140, 255, 255]),
            ([80, 20, 20], [150, 255, 255]),
            ([95, 40, 40], [135, 200, 200])
        ]
        
        for lower, upper in blue_hsv_ranges:
            blue_masks.append(cv2.inRange(hsv, np.array(lower), np.array(upper)))
        
        bgr_blue_ranges = [
            ([60, 0, 0], [255, 120, 120]),
            ([40, 0, 0], [255, 100, 100])
        ]
        
        for lower, upper in bgr_blue_ranges:
            blue_masks.append(cv2.inRange(original_image, np.array(lower), np.array(upper)))
        
        blue_mask = blue_masks[0]
        for mask in blue_masks[1:]:
            blue_mask = cv2.bitwise_or(blue_mask, mask)
        
        blue_lines = cv2.HoughLinesP(
            blue_mask, rho=1, theta=np.pi/180, threshold=20,
            minLineLength=12, maxLineGap=18
        )
        
        if blue_lines is not None:
            filtered_blue = self.optimized_line_filtering(blue_lines)
            lines_by_color['blue'] = [(x1, y1, x2, y2, 'blue') for x1, y1, x2, y2 in filtered_blue]
        
        # 緑線検出（最適化）
        green_masks = []
        green_hsv_ranges = [
            ([30, 40, 40], [90, 255, 255]),
            ([35, 60, 60], [85, 255, 255]),
            ([40, 80, 80], [80, 255, 255])
        ]
        
        for lower, upper in green_hsv_ranges:
            green_masks.append(cv2.inRange(hsv, np.array(lower), np.array(upper)))
        
        bgr_green_ranges = [
            ([0, 60, 0], [120, 255, 120]),
            ([0, 80, 0], [150, 255, 150])
        ]
        
        for lower, upper in bgr_green_ranges:
            green_masks.append(cv2.inRange(original_image, np.array(lower), np.array(upper)))
        
        green_mask = green_masks[0]
        for mask in green_masks[1:]:
            green_mask = cv2.bitwise_or(green_mask, mask)
        
        green_lines = cv2.HoughLinesP(
            green_mask, rho=1, theta=np.pi/180, threshold=20,
            minLineLength=12, maxLineGap=18
        )
        
        if green_lines is not None:
            filtered_green = self.optimized_line_filtering(green_lines)
            lines_by_color['green'] = [(x1, y1, x2, y2, 'green') for x1, y1, x2, y2 in filtered_green]
        
        # 赤線検出（最適化）
        red_masks = []
        red_hsv_ranges = [
            ([0, 40, 40], [15, 255, 255]),
            ([165, 40, 40], [180, 255, 255])
        ]
        
        for lower, upper in red_hsv_ranges:
            red_masks.append(cv2.inRange(hsv, np.array(lower), np.array(upper)))
        
        bgr_red_ranges = [
            ([0, 0, 60], [120, 120, 255]),
            ([0, 0, 40], [100, 100, 255])
        ]
        
        for lower, upper in bgr_red_ranges:
            red_masks.append(cv2.inRange(original_image, np.array(lower), np.array(upper)))
        
        red_mask = red_masks[0]
        for mask in red_masks[1:]:
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        red_lines = cv2.HoughLinesP(
            red_mask, rho=1, theta=np.pi/180, threshold=15,
            minLineLength=10, maxLineGap=25
        )
        
        if red_lines is not None:
            filtered_red = self.optimized_line_filtering(red_lines)
            lines_by_color['red'] = [(x1, y1, x2, y2, 'red') for x1, y1, x2, y2 in filtered_red]
        
        print(f"🔍 最適化色線検出: 青={len(lines_by_color['blue'])}本, "
              f"緑={len(lines_by_color['green'])}本, 赤={len(lines_by_color['red'])}本")
        
        return lines_by_color
    
    def optimized_line_filtering(self, lines):
        """最適化線分フィルタリング"""
        if lines is None or len(lines) == 0:
            return []
        
        lines_reshaped = lines.reshape(-1, 4)
        filtered_lines = []
        
        for line in lines_reshaped:
            x1, y1, x2, y2 = line
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 長さによる基本フィルタリング
            if length >= 8:  # 8ピクセル以上
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # 角度による判定（緩い条件）
                if (abs(angle) < 12 or abs(angle - 90) < 12 or 
                    abs(angle - 180) < 12 or abs(angle + 90) < 12 or
                    abs(angle - 45) < 18 or abs(angle + 45) < 18 or
                    length > 25):
                    filtered_lines.append(line)
        
        return filtered_lines
    
    def extract_optimized_structure_lines(self, binary):
        """最適化構造線抽出"""
        all_lines = []
        
        # 複数の閾値で検出
        thresholds = [60, 40, 25]
        min_lengths = [35, 25, 15]
        max_gaps = [8, 15, 25]
        
        for threshold, min_length, max_gap in zip(thresholds, min_lengths, max_gaps):
            lines = cv2.HoughLinesP(
                binary, rho=1, theta=np.pi/180, threshold=threshold,
                minLineLength=min_length, maxLineGap=max_gap
            )
            if lines is not None:
                all_lines.extend(lines.reshape(-1, 4))
        
        if not all_lines:
            return []
        
        # 重複除去と最適化
        optimized_lines = self.advanced_duplicate_removal(all_lines)
        
        return optimized_lines
    
    def advanced_duplicate_removal(self, lines):
        """高度な重複除去（修正版）"""
        if len(lines) == 0:
            return []
        
        # 基本的な重複除去
        unique_lines = []
        tolerance = 20  # ピクセル単位の許容範囲
        
        for current_line in lines:
            x1, y1, x2, y2 = current_line
            is_duplicate = False
            
            for existing_line in unique_lines[:]:  # コピーを作成して安全に削除
                ex1, ey1, ex2, ey2 = existing_line
                
                # 端点間の距離を計算
                dist1 = ((x1-ex1)**2 + (y1-ey1)**2)**0.5
                dist2 = ((x2-ex2)**2 + (y2-ey2)**2)**0.5
                dist3 = ((x1-ex2)**2 + (y1-ey2)**2)**0.5
                dist4 = ((x2-ex1)**2 + (y2-ey1)**2)**0.5
                
                # 順方向または逆方向で類似していれば重複とみなす
                if (dist1 < tolerance and dist2 < tolerance) or (dist3 < tolerance and dist4 < tolerance):
                    is_duplicate = True
                    # より長い線分を選択
                    current_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    existing_length = ((ex2-ex1)**2 + (ey2-ey1)**2)**0.5
                    if current_length > existing_length:
                        # 既存の線分を置き換え
                        try:
                            unique_lines.remove(existing_line)
                            unique_lines.append(current_line.tolist() if hasattr(current_line, 'tolist') else current_line)
                        except ValueError:
                            pass
                    break
            
            if not is_duplicate:
                unique_lines.append(current_line.tolist() if hasattr(current_line, 'tolist') else current_line)
        
        # 長さによる最終フィルタリング
        final_lines = []
        for line in unique_lines:
            x1, y1, x2, y2 = line
            length = ((x2-x1)**2 + (y2-y1)**2)**0.5
            if length >= 8:  # 8ピクセル以上の線分のみ
                final_lines.append(line)
        
        return final_lines
    
    def optimized_text_recognition(self, image):
        """最適化テキスト認識"""
        if not self.ocr_reader:
            return []
        
        try:
            print("🔤 最適化テキスト認識処理中...")
            
            # 複数の前処理でOCR実行
            text_regions = []
            seen_texts = set()
            
            # オリジナル画像でOCR
            results = self.ocr_reader.readtext(image)
            
            # コントラスト強化画像でOCR
            enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=15)
            results_enhanced = self.ocr_reader.readtext(enhanced)
            
            all_results = results + results_enhanced
            
            for (bbox, text, confidence) in all_results:
                if confidence > 0.4:
                    cleaned_text = text.strip()
                    if (len(cleaned_text) > 0 and not cleaned_text.isspace() 
                        and cleaned_text not in seen_texts):
                        seen_texts.add(cleaned_text)
                        
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))
                        
                        text_regions.append({
                            'bbox': (x1, y1, x2, y2),
                            'text': cleaned_text,
                            'confidence': confidence,
                            'color': 'black'
                        })
            
            print(f"✅ 最適化テキスト認識完了: {len(text_regions)}個")
            return text_regions
            
        except Exception as e:
            print(f"⚠️ テキスト認識エラー: {e}")
            return []
    
    def process_pdf_optimized_ultra(self):
        """最適化超高精度PDFプロセシング"""
        try:
            print("⚡ 最適化超高精度処理開始")
            
            all_elements = {
                'blue_lines': [],
                'green_lines': [],
                'red_lines': [],
                'main_lines': [],
                'text_regions': []
            }
            
            for i, image in enumerate(self.images):
                print(f"🔍 ページ {i+1} を最適化超高精度処理中...")
                
                # 1. 最適化前処理
                original, gray, binary = self.optimized_preprocessing(image)
                
                # 2. 最適化色線検出
                color_lines = self.optimized_color_detection(original)
                all_elements['blue_lines'].extend(color_lines.get('blue', []))
                all_elements['green_lines'].extend(color_lines.get('green', []))
                all_elements['red_lines'].extend(color_lines.get('red', []))
                
                # 3. 最適化構造線抽出
                main_lines = self.extract_optimized_structure_lines(binary)
                all_elements['main_lines'].extend(main_lines)
                
                # 4. 最適化テキスト認識
                text_regions = self.optimized_text_recognition(original)
                all_elements['text_regions'].extend(text_regions)
                
                print(f"✅ ページ {i+1} 最適化超高精度処理完了:")
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
            
            base_image = self.images[0].copy()
            height, width = base_image.shape[:2]
            
            vis_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            alpha = 0.3
            vis_image = cv2.addWeighted(vis_image, 1-alpha, base_image, alpha, 0)
            
            # 線分を描画
            for line in elements['main_lines']:
                x1, y1, x2, y2 = line[:4]
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
            
            for line in elements['blue_lines']:
                x1, y1, x2, y2 = line[:4]
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            for line in elements['green_lines']:
                x1, y1, x2, y2 = line[:4]
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            for line in elements['red_lines']:
                x1, y1, x2, y2 = line[:4]
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # テキスト領域を描画
            for text_region in elements['text_regions']:
                x1, y1, x2, y2 = text_region['bbox']
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.putText(vis_image, text_region['text'], (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            cv2.imwrite(output_path, vis_image)
            print(f"📸 可視化画像を保存: {output_path}")
            
        except Exception as e:
            print(f"⚠️ 可視化作成エラー: {e}")


def convert_pdf_optimized_ultra(input_path, output_path, scale=100, visualization=True):
    """最適化超高精度PDF変換"""
    try:
        print("=" * 70)
        print("⚡ 最適化超高精度 PDF to DXF 変換ツール ⚡")
        print("=" * 70)
        
        # 1. 変換システム初期化
        converter = OptimizedUltraConverter(input_path)
        
        # 2. 最適化超高精度処理
        elements = converter.process_pdf_optimized_ultra()
        
        if elements is None:
            return False
        
        # 3. DXF生成
        print("📐 最適化超高精度DXF生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
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
            vis_path = output_path.replace('.dxf', '_optimized_ultra_vis.png')
            converter.create_visualization(elements, vis_path)
        
        # 5. 結果表示
        total_lines = (len(elements['main_lines']) + len(elements['blue_lines']) + 
                      len(elements['green_lines']) + len(elements['red_lines']))
        
        print("=" * 70)
        print("🎉 最適化超高精度変換完了 🎉")
        print("=" * 70)
        print(f"📁 入力: {input_path}")
        print(f"📁 出力: {output_path}")
        print(f"📊 最適化超高精度検出結果:")
        print(f"   🔵 青線: {len(elements['blue_lines'])}本")
        print(f"   🟢 緑線: {len(elements['green_lines'])}本")
        print(f"   🔴 赤線: {len(elements['red_lines'])}本")
        print(f"   📏 主要線: {len(elements['main_lines'])}本")
        print(f"   📝 テキスト: {len(elements['text_regions'])}個")
        print(f"   🎯 総線分数: {total_lines}本（重複除去最適化済み）")
        print("⚡ 最適化超高精度変換完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='⚡ 最適化超高精度 PDF変換ツール ⚡')
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
        output_path = f"output/{timestamp}_{base_name}_optimized_ultra.dxf"
    
    # 変換実行
    success = convert_pdf_optimized_ultra(
        args.input,
        output_path,
        args.scale,
        args.visualization
    )
    
    if success:
        print("✅ 最適化超高精度変換が正常に完了しました！")
        sys.exit(0)
    else:
        print("❌ 変換に失敗しました。")
        sys.exit(1)
