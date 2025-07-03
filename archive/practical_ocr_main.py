#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
実用的OCR対応版 PDF to DXF コンバーター
EasyOCRによる文字認識と高度な画像処理を組み合わせた実用版
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

# EasyOCRのインポート（オプション）
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCRが利用できません。文字認識機能は無効化されます。")


class PracticalOCRConverter:
    """実用的OCR対応変換システム"""
    
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
            # 450dpiで高解像度変換（バランス重視）
            images = pdf2image.convert_from_path(self.pdf_path, dpi=450)
            
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
            
            print(f"✅ 高品質PDF読み込み完了: {len(self.images)}ページ (450dpi)")
        except Exception as e:
            raise Exception(f"PDF読み込み失敗: {str(e)}")
    
    def advanced_preprocessing(self, image):
        """高度な前処理"""
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Non-local Means Denoising（高品質ノイズ除去）
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. シャープニング
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        
        # 3. CLAHE（コントラスト制限適応ヒストグラム均等化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        
        # 4. 適応的二値化
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 9, 2
        )
        
        # 5. モルフォロジー演算
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return enhanced, binary
    
    def ocr_text_recognition(self, image):
        """OCR文字認識"""
        if not self.ocr_reader:
            return []
        
        try:
            print("🔤 文字認識処理中...")
            results = self.ocr_reader.readtext(image)
            
            text_regions = []
            for (bbox, text, confidence) in results:
                if confidence > 0.4:  # 信頼度40%以上
                    # バウンディングボックスを取得
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    text_regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': text.strip(),
                        'confidence': confidence
                    })
            
            print(f"✅ 文字認識完了: {len(text_regions)}個のテキスト")
            return text_regions
        except Exception as e:
            print(f"⚠️ 文字認識エラー: {e}")
            return []
    
    def advanced_line_detection(self, binary_image, text_regions):
        """高度な線分検出"""
        # テキスト領域をマスク
        text_mask = np.zeros(binary_image.shape, dtype=np.uint8)
        for region in text_regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(text_mask, (x1-3, y1-3), (x2+3, y2+3), 255, -1)
        
        # テキスト領域を除外
        masked_binary = cv2.bitwise_and(binary_image, cv2.bitwise_not(text_mask))
        
        lines = []
        
        # 1. 水平線検出（強化版）
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
        
        h_lines = cv2.HoughLinesP(
            horizontal_lines, rho=1, theta=np.pi/180, threshold=60,
            minLineLength=30, maxLineGap=8
        )
        
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 20:
                    lines.append((x1, y1, x2, y2, 'horizontal'))
        
        # 2. 垂直線検出（強化版）
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
        
        v_lines = cv2.HoughLinesP(
            vertical_lines, rho=1, theta=np.pi/180, threshold=60,
            minLineLength=30, maxLineGap=8
        )
        
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 20:
                    lines.append((x1, y1, x2, y2, 'vertical'))
        
        # 3. 斜め線検出
        diagonal_lines = cv2.HoughLinesP(
            masked_binary, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=25, maxLineGap=12
        )
        
        if diagonal_lines is not None:
            for line in diagonal_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                
                # 水平・垂直でない線のみ
                if length > 25 and not (abs(angle) < 15 or abs(angle - 90) < 15):
                    lines.append((x1, y1, x2, y2, 'diagonal'))
        
        # 4. 線分のクリーンアップ
        cleaned_lines = self._cleanup_lines(lines)
        
        return cleaned_lines
    
    def _cleanup_lines(self, lines):
        """線分のクリーンアップ"""
        if not lines:
            return []
        
        # 重複除去
        unique_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            
            is_duplicate = False
            for existing in unique_lines:
                ex1, ey1, ex2, ey2 = existing[:4]
                
                # 端点間の距離をチェック
                dist1 = np.sqrt((x1 - ex1)**2 + (y1 - ey1)**2)
                dist2 = np.sqrt((x2 - ex2)**2 + (y2 - ey2)**2)
                dist3 = np.sqrt((x1 - ex2)**2 + (y1 - ey2)**2)
                dist4 = np.sqrt((x2 - ex1)**2 + (y2 - ey1)**2)
                
                # 非常に近い線分は重複とみなす
                if (dist1 < 8 and dist2 < 8) or (dist3 < 8 and dist4 < 8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        return unique_lines
    
    def detect_architectural_elements(self, binary_image, lines):
        """建築要素の検出"""
        elements = {
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': []
        }
        
        # 壁の検出（平行線ペア）
        long_lines = [line for line in lines if np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) > 50]
        
        for i, line1 in enumerate(long_lines):
            x1a, y1a, x2a, y2a = line1[:4]
            angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a)) % 180
            
            for j, line2 in enumerate(long_lines[i+1:], i+1):
                x1b, y1b, x2b, y2b = line2[:4]
                angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b)) % 180
                
                # 平行線判定
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                if angle_diff < 8:
                    # 距離計算
                    center1 = ((x1a + x2a) / 2, (y1a + y2a) / 2)
                    center2 = ((x1b + x2b) / 2, (y1b + y2b) / 2)
                    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    # 壁として認識
                    if 15 < dist < 80:
                        elements['walls'].append({
                            'line1': line1[:4],
                            'line2': line2[:4],
                            'thickness': dist,
                            'angle': angle1
                        })
        
        # 円形要素の検出（設備等）
        circles = cv2.HoughCircles(
            binary_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,
            param1=50,
            param2=25,
            minRadius=8,
            maxRadius=60
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            elements['circles'] = [(x, y, r) for (x, y, r) in circles if r > 8]
        else:
            elements['circles'] = []
        
        return elements
    
    def process_pdf_practical_ocr(self):
        """実用的OCR処理でPDFを処理"""
        try:
            print("🚀 実用的OCR処理開始")
            
            all_elements = {
                'lines': [],
                'walls': [],
                'circles': [],
                'text_regions': []
            }
            
            for i, image in enumerate(self.images):
                print(f"🔍 ページ {i+1} を実用的OCR処理中...")
                
                # 1. 高度な前処理
                enhanced, binary = self.advanced_preprocessing(image)
                
                # 2. OCR文字認識
                text_regions = self.ocr_text_recognition(enhanced)
                all_elements['text_regions'].extend(text_regions)
                
                # 3. 高度な線分検出
                lines = self.advanced_line_detection(binary, text_regions)
                all_elements['lines'].extend([line[:4] for line in lines])
                
                # 4. 建築要素検出
                arch_elements = self.detect_architectural_elements(binary, lines)
                all_elements['walls'].extend(arch_elements['walls'])
                all_elements['circles'].extend(arch_elements['circles'])
                
                print(f"✅ ページ {i+1} 完了:")
                print(f"   📏 線分: {len(lines)}本")
                print(f"   🏠 壁: {len(arch_elements['walls'])}個")
                print(f"   ⭕ 円: {len(arch_elements['circles'])}個")
                print(f"   📝 テキスト: {len(text_regions)}個")
            
            return all_elements
            
        except Exception as e:
            print(f"❌ 処理エラー: {str(e)}")
            return None
    
    def create_practical_visualization(self, elements, output_path):
        """実用的可視化"""
        if not self.images:
            return
        
        # 高解像度可視化画像を作成
        vis_image = self.images[0].copy()
        
        # 線分描画（青、細線）
        for line in elements['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 壁描画（緑、太線）
        for wall in elements['walls']:
            line1 = wall['line1']
            line2 = wall['line2']
            x1, y1, x2, y2 = line1
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            x1, y1, x2, y2 = line2
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 円描画（赤）
        for circle in elements['circles']:
            x, y, r = circle
            cv2.circle(vis_image, (x, y), r, (0, 0, 255), 2)
        
        # テキスト領域描画（黄色）
        for text_region in elements['text_regions']:
            x1, y1, x2, y2 = text_region['bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # テキスト表示
            font_scale = 0.6
            cv2.putText(vis_image, text_region['text'], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
        
        cv2.imwrite(output_path, vis_image)
        print(f"🎨 実用的可視化保存: {output_path}")


def create_output_filename(base_name):
    """出力ファイル名を生成（日付時間プレフィックス付き）"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    return f"output/{timestamp}_{base_name}"


def convert_pdf_practical_ocr(input_path, output_path, scale=100, visualization=True):
    """実用的OCR対応PDF変換"""
    try:
        print("=" * 60)
        print("🔤 実用的OCR対応版 AI変換ツール 🔤")
        print("=" * 60)
        
        # 1. 変換システム初期化
        converter = PracticalOCRConverter(input_path)
        
        # 2. 実用的OCR処理
        elements = converter.process_pdf_practical_ocr()
        
        if elements is None:
            return False
        
        # 3. 可視化生成
        if visualization:
            vis_path = output_path.replace('.dxf', '_practical_ocr_vis.png')
            converter.create_practical_visualization(elements, vis_path)
        
        # 4. DXF生成
        print("📐 実用的OCR対応DXF生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        # テキスト情報も含めてDXFに追加
        dxf_elements = {
            'lines': elements['lines'],
            'walls': elements['walls'],
            'circles': elements['circles'],
            'text_regions': elements['text_regions']
        }
        
        dxf_writer.add_elements(dxf_elements)
        dxf_writer.save(output_path)
        
        # 5. 結果表示
        print("=" * 60)
        print("🎉 実用的OCR対応変換完了 🎉")
        print("=" * 60)
        print(f"📁 入力: {input_path}")
        print(f"📁 出力: {output_path}")
        print(f"📊 検出結果:")
        print(f"   📏 線分: {len(elements['lines'])}本")
        print(f"   🏠 壁: {len(elements['walls'])}個")
        print(f"   ⭕ 円: {len(elements['circles'])}個")
        print(f"   📝 テキスト: {len(elements['text_regions'])}個")
        print("🔤 OCR対応による超実用的変換完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🔤 実用的OCR対応版 AI変換ツール 🔤')
    parser.add_argument('--input', '-i', required=True, help='入力PDFファイル')
    parser.add_argument('--output', '-o', help='出力DXFファイル（省略時は自動生成）')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール')
    parser.add_argument('--no-visualization', action='store_true', help='可視化を無効化')
    
    args = parser.parse_args()
    
    # 出力ファイル名の生成
    if args.output:
        output_path = create_output_filename(os.path.basename(args.output))
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = create_output_filename(f"{base_name}_practical_ocr.dxf")
    
    # 変換実行
    success = convert_pdf_practical_ocr(
        args.input,
        output_path,
        args.scale,
        not args.no_visualization
    )
    
    if success:
        print("✅ 実用的OCR対応変換が正常に完了しました！")
        sys.exit(0)
    else:
        print("❌ 変換に失敗しました。")
        sys.exit(1)
