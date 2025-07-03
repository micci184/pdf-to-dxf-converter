#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
究極精度 PDF to DXF コンバーター
青い線、緑の蛍光線、全ての文字を完璧に取り込む究極版
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


class UltimatePrecisionConverter:
    """究極精度変換システム"""
    
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
        """超高解像度PDFロード"""
        try:
            print("📖 超高解像度PDF読み込み中...")
            # 600dpiで超高解像度変換（究極品質）
            images = pdf2image.convert_from_path(self.pdf_path, dpi=600)
            
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
            
            print(f"✅ 超高解像度PDF読み込み完了: {len(self.images)}ページ (600dpi)")
        except Exception as e:
            raise Exception(f"PDF読み込み失敗: {str(e)}")
    
    def ultimate_preprocessing(self, image):
        """究極前処理"""
        # カラー情報を保持
        original = image.copy()
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 高品質ノイズ除去
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. エッジ保持シャープニング
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        
        # 3. 適応的二値化
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 7, 2
        )
        
        return original, gray, binary
    
    def detect_color_lines(self, original_image):
        """色別線分検出（改良版）"""
        # HSV変換
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        
        lines_by_color = {}
        
        # 青い線の検出（範囲を拡大）
        blue_lower1 = np.array([100, 30, 30])
        blue_upper1 = np.array([140, 255, 255])
        blue_mask1 = cv2.inRange(hsv, blue_lower1, blue_upper1)
        
        # 暗い青も検出
        blue_lower2 = np.array([90, 50, 20])
        blue_upper2 = np.array([120, 255, 200])
        blue_mask2 = cv2.inRange(hsv, blue_lower2, blue_upper2)
        
        blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
        
        # BGRでも青を検出
        bgr_blue_lower = np.array([50, 0, 0])
        bgr_blue_upper = np.array([255, 100, 100])
        bgr_blue_mask = cv2.inRange(original_image, bgr_blue_lower, bgr_blue_upper)
        
        blue_mask = cv2.bitwise_or(blue_mask, bgr_blue_mask)
        
        blue_lines = cv2.HoughLinesP(
            blue_mask, rho=1, theta=np.pi/180, threshold=20,
            minLineLength=15, maxLineGap=8
        )
        
        if blue_lines is not None:
            lines_by_color['blue'] = [(x1, y1, x2, y2, 'blue') for x1, y1, x2, y2 in blue_lines.reshape(-1, 4)]
        else:
            lines_by_color['blue'] = []
        
        # 緑の線の検出（蛍光緑を含む範囲拡大）
        green_lower1 = np.array([35, 40, 40])
        green_upper1 = np.array([85, 255, 255])
        green_mask1 = cv2.inRange(hsv, green_lower1, green_upper1)
        
        # 明るい緑（蛍光緑）
        green_lower2 = np.array([40, 100, 100])
        green_upper2 = np.array([80, 255, 255])
        green_mask2 = cv2.inRange(hsv, green_lower2, green_upper2)
        
        green_mask = cv2.bitwise_or(green_mask1, green_mask2)
        
        # BGRでも緑を検出
        bgr_green_lower = np.array([0, 50, 0])
        bgr_green_upper = np.array([100, 255, 100])
        bgr_green_mask = cv2.inRange(original_image, bgr_green_lower, bgr_green_upper)
        
        green_mask = cv2.bitwise_or(green_mask, bgr_green_mask)
        
        green_lines = cv2.HoughLinesP(
            green_mask, rho=1, theta=np.pi/180, threshold=20,
            minLineLength=15, maxLineGap=8
        )
        
        if green_lines is not None:
            lines_by_color['green'] = [(x1, y1, x2, y2, 'green') for x1, y1, x2, y2 in green_lines.reshape(-1, 4)]
        else:
            lines_by_color['green'] = []
        
        # デバッグ用マスク保存
        if lines_by_color['blue'] or lines_by_color['green']:
            cv2.imwrite('debug_blue_mask.png', blue_mask)
            cv2.imwrite('debug_green_mask.png', green_mask)
            print(f"🔍 デバッグ: 青マスク・緑マスクを保存しました")
        
        return lines_by_color
    
    def ultimate_text_recognition(self, image):
        """究極文字認識"""
        if not self.ocr_reader:
            return []
        
        try:
            print("🔤 究極文字認識処理中...")
            results = self.ocr_reader.readtext(image)
            
            text_regions = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # 信頼度30%以上（寛容に）
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    text_regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': text.strip(),
                        'confidence': confidence
                    })
            
            print(f"✅ 究極文字認識完了: {len(text_regions)}個のテキスト")
            return text_regions
        except Exception as e:
            print(f"⚠️ 文字認識エラー: {e}")
            return []
    
    def process_pdf_ultimate_precision(self):
        """究極精度でPDFを処理"""
        try:
            print("🚀 究極精度処理開始")
            
            all_elements = {
                'blue_lines': [],
                'green_lines': [],
                'all_lines': [],
                'walls': [],
                'circles': [],
                'text_regions': []
            }
            
            for i, image in enumerate(self.images):
                print(f"🔍 ページ {i+1} を究極精度処理中...")
                
                # 1. 究極前処理
                original, gray, binary = self.ultimate_preprocessing(image)
                
                # 2. 色別線分検出
                color_lines = self.detect_color_lines(original)
                all_elements['blue_lines'].extend(color_lines.get('blue', []))
                all_elements['green_lines'].extend(color_lines.get('green', []))
                
                # 3. 究極文字認識
                text_regions = self.ultimate_text_recognition(gray)
                all_elements['text_regions'].extend(text_regions)
                
                # 4. 全体線分検出
                all_lines = cv2.HoughLinesP(
                    binary, rho=1, theta=np.pi/180, threshold=20,
                    minLineLength=15, maxLineGap=8
                )
                
                if all_lines is not None:
                    all_elements['all_lines'].extend([
                        (x1, y1, x2, y2) for x1, y1, x2, y2 in all_lines.reshape(-1, 4)
                    ])
                
                print(f"✅ ページ {i+1} 完了:")
                print(f"   🔵 青い線: {len(color_lines.get('blue', []))}本")
                print(f"   🟢 緑の線: {len(color_lines.get('green', []))}本")
                print(f"   📏 全線分: {len(all_lines) if all_lines is not None else 0}本")
                print(f"   📝 テキスト: {len(text_regions)}個")
            
            return all_elements
            
        except Exception as e:
            print(f"❌ 処理エラー: {str(e)}")
            return None


def convert_pdf_ultimate_precision(input_path, output_path, scale=100, visualization=True):
    """究極精度PDF変換"""
    try:
        print("=" * 60)
        print("🌟 究極精度 PDF変換ツール 🌟")
        print("=" * 60)
        
        # 1. 変換システム初期化
        converter = UltimatePrecisionConverter(input_path)
        
        # 2. 究極精度処理
        elements = converter.process_pdf_ultimate_precision()
        
        if elements is None:
            return False
        
        # 3. DXF生成
        print("📐 究極精度DXF生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        # 全ての要素をDXFに追加
        dxf_elements = {
            'lines': elements['all_lines'],
            'blue_lines': elements['blue_lines'],
            'green_lines': elements['green_lines'],
            'text_regions': elements['text_regions']
        }
        
        dxf_writer.add_elements(dxf_elements)
        dxf_writer.save(output_path)
        
        # 4. 結果表示
        print("=" * 60)
        print("🎉 究極精度変換完了 🎉")
        print("=" * 60)
        print(f"📁 入力: {input_path}")
        print(f"📁 出力: {output_path}")
        print(f"📊 究極精度検出結果:")
        print(f"   🔵 青い線: {len(elements['blue_lines'])}本")
        print(f"   🟢 緑の線: {len(elements['green_lines'])}本")
        print(f"   📏 全線分: {len(elements['all_lines'])}本")
        print(f"   📝 テキスト: {len(elements['text_regions'])}個")
        print("🌟 究極精度での変換完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🌟 究極精度 PDF変換ツール 🌟')
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
        output_path = f"output/{timestamp}_{base_name}_ultimate.dxf"
    
    # 変換実行
    success = convert_pdf_ultimate_precision(
        args.input,
        output_path,
        args.scale
    )
    
    if success:
        print("✅ 究極精度変換が正常に完了しました！")
        sys.exit(0)
    else:
        print("❌ 変換に失敗しました。")
        sys.exit(1)
