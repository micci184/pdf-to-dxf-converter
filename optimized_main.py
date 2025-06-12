#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最適化版 PDF to DXF コンバーター
処理速度と精度のバランスを取った実用的な変換ツール
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pdf2image
import pytesseract
from sklearn.cluster import DBSCAN
from enhanced_dxf_writer import EnhancedDXFWriter


class OptimizedPDFProcessor:
    """最適化されたPDFプロセッサー"""
    
    def __init__(self, pdf_path):
        """
        初期化
        
        Args:
            pdf_path (str): PDFファイルのパス
        """
        self.pdf_path = pdf_path
        self.images = []
        self.load_pdf()
    
    def load_pdf(self):
        """PDFファイルを読み込み、画像に変換"""
        try:
            # PDFを適切な解像度で画像に変換（400dpiで高品質かつ高速）
            images = pdf2image.convert_from_path(self.pdf_path, dpi=400)
            
            # OpenCV形式に変換
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
                
            print(f"PDFから{len(self.images)}ページを読み込みました（400dpi）")
        except Exception as e:
            raise Exception(f"PDFの読み込みに失敗しました: {str(e)}")
    
    def preprocess_image_optimized(self, image):
        """
        最適化された画像前処理
        
        Args:
            image (numpy.ndarray): 入力画像
            
        Returns:
            tuple: (preprocessed, binary) 前処理画像と二値化画像
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. CLAHE（コントラスト制限適応ヒストグラム均等化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # 3. 適応的二値化
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 4. モルフォロジー演算でノイズ除去
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return enhanced, binary
    
    def detect_text_regions(self, image):
        """
        テキスト領域を検出してマスクを作成
        
        Args:
            image (numpy.ndarray): 入力画像
            
        Returns:
            numpy.ndarray: テキストマスク
        """
        text_mask = np.zeros(image.shape, dtype=np.uint8)
        
        try:
            # Tesseractでテキスト領域を検出
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 20:  # 信頼度20以上
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # テキスト領域をマスク（少し拡張）
                    cv2.rectangle(text_mask, (x-2, y-2), (x + w + 2, y + h + 2), 255, -1)
            
        except Exception as e:
            print(f"OCR処理でエラー（スキップ）: {e}")
        
        return text_mask
    
    def detect_lines_optimized(self, binary_image, text_mask):
        """
        最適化された線分検出
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            text_mask (numpy.ndarray): テキストマスク
            
        Returns:
            list: 検出された線分のリスト
        """
        # テキスト領域を除外
        line_mask = cv2.bitwise_not(text_mask)
        masked_binary = cv2.bitwise_and(binary_image, line_mask)
        
        # 1. 水平線と垂直線を強調
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        horizontal_lines = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # 2. 線分検出
        lines = []
        
        # 水平線の検出
        h_lines = cv2.HoughLinesP(
            horizontal_lines, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                lines.append((x1, y1, x2, y2, 'horizontal'))
        
        # 垂直線の検出
        v_lines = cv2.HoughLinesP(
            vertical_lines, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                lines.append((x1, y1, x2, y2, 'vertical'))
        
        # 3. その他の線分検出
        other_lines = cv2.HoughLinesP(
            masked_binary, rho=1, theta=np.pi/180, threshold=30,
            minLineLength=20, maxLineGap=15
        )
        if other_lines is not None:
            for line in other_lines:
                x1, y1, x2, y2 = line[0]
                # 水平・垂直でない線のみ追加
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                if not (abs(angle) < 10 or abs(angle - 90) < 10):
                    lines.append((x1, y1, x2, y2, 'other'))
        
        # 4. 線分のクリーンアップ
        cleaned_lines = self._cleanup_lines(lines)
        
        return cleaned_lines
    
    def _cleanup_lines(self, lines):
        """
        線分のクリーンアップ（重複除去、短い線の除去など）
        
        Args:
            lines (list): 線分のリスト
            
        Returns:
            list: クリーンアップされた線分のリスト
        """
        if not lines:
            return []
        
        cleaned = []
        
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            
            # 線分の長さを計算
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 短すぎる線分は除外
            if length < 15:
                continue
            
            # 重複チェック
            is_duplicate = False
            for existing_line in cleaned:
                ex1, ey1, ex2, ey2 = existing_line[:4]
                
                # 端点間の距離をチェック
                dist1 = np.sqrt((x1 - ex1)**2 + (y1 - ey1)**2)
                dist2 = np.sqrt((x2 - ex2)**2 + (y2 - ey2)**2)
                dist3 = np.sqrt((x1 - ex2)**2 + (y1 - ey2)**2)
                dist4 = np.sqrt((x2 - ex1)**2 + (y2 - ey1)**2)
                
                # 近い線分は重複とみなす
                if (dist1 < 10 and dist2 < 10) or (dist3 < 10 and dist4 < 10):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                cleaned.append(line)
        
        return cleaned
    
    def detect_walls_optimized(self, lines):
        """
        最適化された壁検出
        
        Args:
            lines (list): 線分のリスト
            
        Returns:
            list: 検出された壁のリスト
        """
        walls = []
        
        # 平行線のペアを検出
        for i, line1 in enumerate(lines):
            x1a, y1a, x2a, y2a = line1[:4]
            angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a)) % 180
            length1 = np.sqrt((x2a - x1a)**2 + (y2a - y1a)**2)
            
            # 短い線分は壁として考慮しない
            if length1 < 50:
                continue
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                x1b, y1b, x2b, y2b = line2[:4]
                angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b)) % 180
                length2 = np.sqrt((x2b - x1b)**2 + (y2b - y1b)**2)
                
                # 短い線分は壁として考慮しない
                if length2 < 50:
                    continue
                
                # 角度の差が小さい（平行）
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                if angle_diff < 10:
                    # 距離を計算
                    center1 = ((x1a + x2a) / 2, (y1a + y2a) / 2)
                    center2 = ((x1b + x2b) / 2, (y1b + y2b) / 2)
                    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    # 適切な距離の平行線を壁とみなす
                    if 15 < dist < 80:  # 15-80ピクセルの厚みを壁とみなす
                        walls.append({
                            'line1': line1[:4],
                            'line2': line2[:4],
                            'thickness': dist,
                            'angle': angle1
                        })
        
        return walls
    
    def process_pdf(self):
        """
        PDFを処理して図形要素を検出
        
        Returns:
            dict: 検出された図形要素
        """
        all_elements = {
            'lines': [],
            'walls': [],
            'text_regions': []
        }
        
        for i, image in enumerate(self.images):
            print(f"ページ {i+1} を処理中...")
            
            # 1. 画像前処理
            enhanced, binary = self.preprocess_image_optimized(image)
            
            # 2. テキスト領域の検出
            text_mask = self.detect_text_regions(enhanced)
            
            # 3. 線分検出
            lines = self.detect_lines_optimized(binary, text_mask)
            all_elements['lines'].extend([line[:4] for line in lines])
            
            # 4. 壁検出
            walls = self.detect_walls_optimized(lines)
            all_elements['walls'].extend(walls)
            
            print(f"ページ {i+1} 完了: 線分{len(lines)}本, 壁{len(walls)}個")
        
        return all_elements
    
    def create_visualization(self, elements, output_path):
        """
        検出結果の可視化
        
        Args:
            elements (dict): 検出された要素
            output_path (str): 出力パス
        """
        if not self.images:
            return
        
        # 最初のページを使用
        vis_image = self.images[0].copy()
        
        # 線分の描画（青）
        for line in elements['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 壁の描画（緑、太線）
        for wall in elements['walls']:
            line1 = wall['line1']
            line2 = wall['line2']
            x1, y1, x2, y2 = line1
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
            x1, y1, x2, y2 = line2
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        cv2.imwrite(output_path, vis_image)
        print(f"可視化画像を保存: {output_path}")


def convert_pdf_to_dxf_optimized(input_path, output_path, scale=100, visualization=False):
    """
    PDFを最適化された方法でDXFに変換する
    
    Args:
        input_path (str): 入力PDFファイルのパス
        output_path (str): 出力DXFファイルのパス
        scale (int): スケール（1:scale）
        visualization (bool): 可視化画像を保存するかどうか
    
    Returns:
        bool: 変換が成功したかどうか
    """
    try:
        print("=== 最適化版 PDF to DXF 変換開始 ===")
        
        # 1. PDFの処理
        print("1. PDFを読み込み中...")
        pdf_processor = OptimizedPDFProcessor(input_path)
        
        # 2. 図形要素の検出
        print("2. 図形要素を検出中...")
        elements = pdf_processor.process_pdf()
        
        # 3. 可視化（オプション）
        if visualization:
            vis_path = output_path.replace('.dxf', '_visualization.png')
            print(f"3. 可視化画像を生成中: {vis_path}")
            pdf_processor.create_visualization(elements, vis_path)
        
        # 4. DXFファイルの作成
        print("4. DXFファイルを生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        # スケールの設定
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        # 要素の追加
        dxf_elements = {
            'lines': elements['lines'],
            'walls': elements['walls']
        }
        
        dxf_writer.add_elements(dxf_elements)
        dxf_writer.save(output_path)
        
        # 5. 結果の表示
        print("=== 変換完了 ===")
        print(f"入力: {input_path}")
        print(f"出力: {output_path}")
        print(f"検出結果:")
        print(f"  - 線分: {len(elements['lines'])}本")
        print(f"  - 壁: {len(elements['walls'])}個")
        
        return True
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {str(e)}")
        return False


if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='最適化版 PDF to DXF 変換ツール')
    parser.add_argument('--input', '-i', required=True, help='入力PDFファイルのパス')
    parser.add_argument('--output', '-o', required=True, help='出力DXFファイルのパス')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール（1:scale）')
    parser.add_argument('--visualization', '-v', action='store_true', help='可視化画像を生成する')
    
    args = parser.parse_args()
    
    # 変換実行
    success = convert_pdf_to_dxf_optimized(
        args.input, 
        args.output, 
        args.scale, 
        args.visualization
    )
    
    if success:
        print("変換が正常に完了しました。")
        sys.exit(0)
    else:
        print("変換に失敗しました。")
        sys.exit(1)
