#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
実用レベル PDF to DXF コンバーター
ノイズ除去と構造理解に重点を置いた実用版
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
from high_precision_dxf_writer import HighPrecisionDXFWriter, create_high_precision_dxf

# EasyOCRのインポート（オプション）
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class ProductionReadyConverter:
    """実用レベル変換システム"""
    
    def __init__(self, pdf_path):
        """初期化"""
        self.pdf_path = pdf_path
        self.images = []
        self.ocr_reader = None
        self.initialize_ocr()
        self.load_pdf()
    
    def initialize_ocr(self):
        """OCR初期化（軽量化）"""
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
        """最適化PDFロード"""
        try:
            print("📖 PDF読み込み中...")
            # 300dpiで適度な解像度（バランス重視）
            images = pdf2image.convert_from_path(self.pdf_path, dpi=300)
            
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
            
            print(f"✅ PDF読み込み完了: {len(self.images)}ページ (300dpi)")
        except Exception as e:
            raise Exception(f"PDF読み込み失敗: {str(e)}")
    
    def production_preprocessing(self, image):
        """実用レベル前処理"""
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. ガウシアンブラー（ノイズ軽減）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. 適応的二値化（シンプル）
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 3. ノイズ除去（モルフォロジー）
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return gray, binary
    
    def smart_text_recognition(self, image):
        """スマート文字認識"""
        if not self.ocr_reader:
            return []
        
        try:
            print("🔤 文字認識処理中...")
            results = self.ocr_reader.readtext(image)
            
            text_regions = []
            for (bbox, text, confidence) in results:
                if confidence > 0.6:  # 信頼度60%以上（厳格化）
                    # バウンディングボックスを取得
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    # 小さすぎるテキストは除外
                    if (x2 - x1) > 20 and (y2 - y1) > 10:
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
    
    def smart_line_detection(self, binary_image, text_regions):
        """スマート線分検出（ノイズ除去重視）"""
        # テキスト領域をマスク（拡張）
        text_mask = np.zeros(binary_image.shape, dtype=np.uint8)
        for region in text_regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(text_mask, (x1-10, y1-10), (x2+10, y2+10), 255, -1)
        
        # テキスト領域を除外
        masked_binary = cv2.bitwise_and(binary_image, cv2.bitwise_not(text_mask))
        
        lines = []
        
        # 1. 水平線検出（厳格化）
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        h_lines = cv2.HoughLinesP(
            horizontal_lines, rho=1, theta=np.pi/180, threshold=80,
            minLineLength=50, maxLineGap=5
        )
        
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 40:  # 長い線のみ
                    lines.append((x1, y1, x2, y2, 'horizontal'))
        
        # 2. 垂直線検出（厳格化）
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, vertical_kernel)
        
        v_lines = cv2.HoughLinesP(
            vertical_lines, rho=1, theta=np.pi/180, threshold=80,
            minLineLength=50, maxLineGap=5
        )
        
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 40:  # 長い線のみ
                    lines.append((x1, y1, x2, y2, 'vertical'))
        
        # 3. 線分のクリーンアップ（厳格化）
        cleaned_lines = self._aggressive_cleanup_lines(lines)
        
        return cleaned_lines
    
    def _aggressive_cleanup_lines(self, lines):
        """厳格な線分クリーンアップ"""
        if not lines:
            return []
        
        # 1. 長さによるフィルタリング
        long_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 30:  # 30ピクセル以上の線のみ
                long_lines.append(line)
        
        if not long_lines:
            return []
        
        # 2. クラスタリングによる重複除去
        line_centers = []
        for line in long_lines:
            x1, y1, x2, y2 = line[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            line_centers.append([center_x, center_y])
        
        if len(line_centers) > 1:
            clustering = DBSCAN(eps=20, min_samples=1).fit(line_centers)
            labels = clustering.labels_
            
            # 各クラスターから代表線を選択
            unique_lines = []
            for label in set(labels):
                cluster_lines = [long_lines[i] for i, l in enumerate(labels) if l == label]
                # 最も長い線を選択
                longest_line = max(cluster_lines, key=lambda l: np.sqrt((l[2] - l[0])**2 + (l[3] - l[1])**2))
                unique_lines.append(longest_line)
            
            return unique_lines
        else:
            return long_lines
    
    def detect_architectural_structure(self, binary_image, lines):
        """建築構造の検出（実用重視）"""
        elements = {
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': []
        }
        
        # 壁の検出（厳格化）
        if len(lines) < 2:
            return elements
        
        # 長い線のみを対象
        structural_lines = [line for line in lines 
                          if np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) > 80]
        
        for i, line1 in enumerate(structural_lines):
            x1a, y1a, x2a, y2a = line1[:4]
            angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a)) % 180
            
            for j, line2 in enumerate(structural_lines[i+1:], i+1):
                x1b, y1b, x2b, y2b = line2[:4]
                angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b)) % 180
                
                # 平行線判定（厳格化）
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                if angle_diff < 5:  # より厳格な平行判定
                    # 距離計算
                    center1 = ((x1a + x2a) / 2, (y1a + y2a) / 2)
                    center2 = ((x1b + x2b) / 2, (y1b + y2b) / 2)
                    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    # 適切な壁厚のみ
                    if 20 < dist < 60:
                        elements['walls'].append({
                            'line1': line1[:4],
                            'line2': line2[:4],
                            'thickness': dist,
                            'angle': angle1
                        })
        
        # 円形要素の検出（厳格化）
        circles = cv2.HoughCircles(
            binary_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,  # 最小距離を増加
            param1=60,   # 閾値を上げる
            param2=35,   # 閾値を上げる
            minRadius=10,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # 重複除去
            unique_circles = []
            for (x, y, r) in circles:
                is_duplicate = False
                for (ex, ey, er) in unique_circles:
                    if np.sqrt((x - ex)**2 + (y - ey)**2) < 30:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_circles.append((x, y, r))
            
            elements['circles'] = unique_circles
        else:
            elements['circles'] = []
        
        return elements
    
    def process_pdf_production_ready(self):
        """実用レベル処理でPDFを処理"""
        try:
            print("🚀 実用レベル処理開始")
            
            all_elements = {
                'lines': [],
                'walls': [],
                'circles': [],
                'text_regions': []
            }
            
            for i, image in enumerate(self.images):
                print(f"🔍 ページ {i+1} を実用レベル処理中...")
                
                # 1. 実用レベル前処理
                enhanced, binary = self.production_preprocessing(image)
                
                # 2. スマート文字認識
                text_regions = self.smart_text_recognition(enhanced)
                all_elements['text_regions'].extend(text_regions)
                
                # 3. スマート線分検出
                lines = self.smart_line_detection(binary, text_regions)
                all_elements['lines'].extend([line[:4] for line in lines])
                
                # 4. 建築構造検出
                arch_elements = self.detect_architectural_structure(binary, lines)
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
    
    def create_production_visualization(self, elements, output_path):
        """実用レベル可視化"""
        if not self.images:
            return
        
        # 元画像をベースに可視化
        vis_image = self.images[0].copy()
        
        # 線分描画（青、適度な太さ）
        for line in elements['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 壁描画（緑、太線）
        for wall in elements['walls']:
            line1 = wall['line1']
            line2 = wall['line2']
            x1, y1, x2, y2 = line1
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
            x1, y1, x2, y2 = line2
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        # 円描画（赤、適度な太さ）
        for circle in elements['circles']:
            x, y, r = circle
            cv2.circle(vis_image, (x, y), r, (0, 0, 255), 3)
        
        # テキスト領域描画（黄色、枠のみ）
        for text_region in elements['text_regions']:
            x1, y1, x2, y2 = text_region['bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # テキスト表示（読みやすく）
            cv2.putText(vis_image, text_region['text'], (x1, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imwrite(output_path, vis_image)
        print(f"🎨 実用レベル可視化保存: {output_path}")


def create_output_filename(base_name):
    """出力ファイル名を生成（日付時間プレフィックス付き）"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    return f"output/{timestamp}_{base_name}"


def convert_pdf_production_ready(input_path, output_path, scale=100, visualization=True):
    """実用レベルPDF変換"""
    try:
        print("=" * 60)
        print("🏭 実用レベル PDF変換ツール 🏭")
        print("=" * 60)
        
        # 1. 変換システム初期化
        converter = ProductionReadyConverter(input_path)
        
        # 2. 実用レベル処理
        elements = converter.process_pdf_production_ready()
        
        if elements is None:
            return False
        
        # 3. 可視化生成
        if visualization:
            vis_path = output_path.replace('.dxf', '_production_vis.png')
            converter.create_production_visualization(elements, vis_path)
        
        # 4. DXF生成
        print("📐 高精度DXF生成中...")
        
        # 厳選された要素のみDXFに追加
        dxf_elements = {
            'lines': elements['lines'],
            'walls': elements['walls'],
            'circles': elements['circles'],
            'text_regions': elements['text_regions']
        }
        
        # 高精度DXF作成
        success = create_high_precision_dxf(dxf_elements, output_path, scale)
        
        # 5. 結果表示
        print("=" * 60)
        print("🎉 実用レベル変換完了 🎉")
        print("=" * 60)
        print(f"📁 入力: {input_path}")
        print(f"📁 出力: {output_path}")
        print(f"📊 厳選された検出結果:")
        print(f"   📏 線分: {len(elements['lines'])}本")
        print(f"   🏠 壁: {len(elements['walls'])}個")
        print(f"   ⭕ 円: {len(elements['circles'])}個")
        print(f"   📝 テキスト: {len(elements['text_regions'])}個")
        print("🏭 実用レベル品質での変換完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🏭 実用レベル PDF変換ツール 🏭')
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
        output_path = create_output_filename(f"{base_name}_production.dxf")
    
    # 変換実行
    success = convert_pdf_production_ready(
        args.input,
        output_path,
        args.scale,
        not args.no_visualization
    )
    
    if success:
        print("✅ 実用レベル変換が正常に完了しました！")
        sys.exit(0)
    else:
        print("❌ 変換に失敗しました。")
        sys.exit(1)
