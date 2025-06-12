#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
バランス調整版 PDF to DXF コンバーター
ノイズ除去と情報保持のバランスを最適化
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


class BalancedProcessor:
    """バランス調整プロセッサー"""
    
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
            # PDFを適切な解像度で画像に変換（350dpiでバランス）
            images = pdf2image.convert_from_path(self.pdf_path, dpi=350)
            
            # OpenCV形式に変換
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
                
            print(f"PDFから{len(self.images)}ページを読み込みました（350dpi）")
        except Exception as e:
            raise Exception(f"PDFの読み込みに失敗しました: {str(e)}")
    
    def preprocess_image_balanced(self, image):
        """
        バランス調整された画像前処理
        
        Args:
            image (numpy.ndarray): 入力画像
            
        Returns:
            numpy.ndarray: 前処理された二値化画像
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 軽いガウシアンブラー
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. 適応的二値化（バランス調整）
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 3
        )
        
        # 3. 軽いモルフォロジー演算
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 4. 極小ノイズのみ除去（20ピクセル未満）
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:  # 20ピクセル未満の極小領域のみ除去
                cv2.fillPoly(binary, [contour], 0)
        
        return binary
    
    def detect_lines_balanced(self, binary_image):
        """
        バランス調整された線分検出
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            
        Returns:
            list: 検出された線分のリスト
        """
        # 1. 水平線と垂直線を強調（中程度のカーネル）
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)
        
        # 2. 線分検出（バランス調整された閾値）
        lines = []
        
        # 水平線の検出
        h_lines = cv2.HoughLinesP(
            horizontal_lines, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=25, maxLineGap=10
        )
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 15:  # 15ピクセル以上
                    lines.append((x1, y1, x2, y2, 'horizontal'))
        
        # 垂直線の検出
        v_lines = cv2.HoughLinesP(
            vertical_lines, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=25, maxLineGap=10
        )
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 15:  # 15ピクセル以上
                    lines.append((x1, y1, x2, y2, 'vertical'))
        
        # 3. その他の線分（斜め線等）
        other_lines = cv2.HoughLinesP(
            binary_image, rho=1, theta=np.pi/180, threshold=40,
            minLineLength=20, maxLineGap=15
        )
        if other_lines is not None:
            for line in other_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                
                # 中程度の長さの線を採用
                if length > 20 and not (abs(angle) < 10 or abs(angle - 90) < 10):
                    lines.append((x1, y1, x2, y2, 'other'))
        
        # 4. 適度な線分統合
        cleaned_lines = self._moderate_line_cleanup(lines)
        
        return cleaned_lines
    
    def _moderate_line_cleanup(self, lines):
        """
        適度な線分クリーンアップ
        
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
            
            # 極短線のみ除外（10ピクセル未満）
            if length < 10:
                continue
            
            # 適度な重複チェック
            is_duplicate = False
            for existing_line in cleaned:
                ex1, ey1, ex2, ey2 = existing_line[:4]
                
                # 端点間の距離をチェック
                dist1 = np.sqrt((x1 - ex1)**2 + (y1 - ey1)**2)
                dist2 = np.sqrt((x2 - ex2)**2 + (y2 - ey2)**2)
                dist3 = np.sqrt((x1 - ex2)**2 + (y1 - ey2)**2)
                dist4 = np.sqrt((x2 - ex1)**2 + (y2 - ey1)**2)
                
                # 非常に近い線分のみ重複とみなす
                if (dist1 < 5 and dist2 < 5) or (dist3 < 5 and dist4 < 5):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                cleaned.append(line)
        
        return cleaned
    
    def detect_walls_moderate(self, lines):
        """
        適度な壁検出
        
        Args:
            lines (list): 線分のリスト
            
        Returns:
            list: 検出された壁のリスト
        """
        walls = []
        
        # 中程度の長さの線分を対象とする
        medium_lines = [line for line in lines if np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) > 40]
        
        # 平行線のペアを検出
        for i, line1 in enumerate(medium_lines):
            x1a, y1a, x2a, y2a = line1[:4]
            angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a)) % 180
            
            for j, line2 in enumerate(medium_lines[i+1:], i+1):
                x1b, y1b, x2b, y2b = line2[:4]
                angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b)) % 180
                
                # 角度の差が小さい（平行）
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                if angle_diff < 8:
                    # 距離を計算
                    center1 = ((x1a + x2a) / 2, (y1a + y2a) / 2)
                    center2 = ((x1b + x2b) / 2, (y1b + y2b) / 2)
                    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    # 適切な距離の平行線を壁とみなす
                    if 15 < dist < 80:
                        walls.append({
                            'line1': line1[:4],
                            'line2': line2[:4],
                            'thickness': dist,
                            'angle': angle1
                        })
        
        return walls
    
    def detect_circles_and_arcs(self, binary_image):
        """
        円と円弧の検出
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            
        Returns:
            list: 検出された円のリスト
        """
        circles = []
        
        # HoughCirclesで円を検出
        detected_circles = cv2.HoughCircles(
            binary_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )
        
        if detected_circles is not None:
            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                if r > 5:  # 5ピクセル以上の円のみ
                    circles.append((x, y, r))
        
        return circles
    
    def process_pdf_balanced(self):
        """
        バランス調整処理でPDFを処理
        
        Returns:
            dict: 検出された図形要素
        """
        all_elements = {
            'lines': [],
            'walls': [],
            'circles': []
        }
        
        for i, image in enumerate(self.images):
            print(f"ページ {i+1} を処理中（バランス調整）...")
            
            # 1. バランス調整前処理
            binary = self.preprocess_image_balanced(image)
            
            # 2. バランス調整線分検出
            lines = self.detect_lines_balanced(binary)
            all_elements['lines'].extend([line[:4] for line in lines])
            
            # 3. 適度な壁検出
            walls = self.detect_walls_moderate(lines)
            all_elements['walls'].extend(walls)
            
            # 4. 円・円弧検出
            circles = self.detect_circles_and_arcs(binary)
            all_elements['circles'].extend(circles)
            
            print(f"ページ {i+1} 完了: 線分{len(lines)}本, 壁{len(walls)}個, 円{len(circles)}個")
        
        return all_elements
    
    def create_balanced_visualization(self, elements, output_path):
        """
        バランス調整された可視化画像を作成
        
        Args:
            elements (dict): 検出された要素
            output_path (str): 出力パス
        """
        if not self.images:
            return
        
        # 元画像をベースに使用
        vis_image = self.images[0].copy()
        
        # 線分の描画（青、細線）
        for line in elements['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # 壁の描画（緑、太線）
        for wall in elements['walls']:
            line1 = wall['line1']
            line2 = wall['line2']
            x1, y1, x2, y2 = line1
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x1, y1, x2, y2 = line2
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 円の描画（赤）
        for circle in elements['circles']:
            x, y, r = circle
            cv2.circle(vis_image, (x, y), r, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, vis_image)
        print(f"バランス調整可視化画像を保存: {output_path}")


def convert_pdf_to_dxf_balanced(input_path, output_path, scale=100, visualization=False):
    """
    PDFをバランス調整版でDXFに変換する
    
    Args:
        input_path (str): 入力PDFファイルのパス
        output_path (str): 出力DXFファイルのパス
        scale (int): スケール（1:scale）
        visualization (bool): 可視化画像を保存するかどうか
    
    Returns:
        bool: 変換が成功したかどうか
    """
    try:
        print("=== バランス調整版 PDF to DXF 変換開始 ===")
        
        # 1. PDFの処理
        print("1. PDFを読み込み中...")
        pdf_processor = BalancedProcessor(input_path)
        
        # 2. バランス調整処理
        print("2. バランス調整処理で図形要素を検出中...")
        elements = pdf_processor.process_pdf_balanced()
        
        # 3. 可視化（オプション）
        if visualization:
            vis_path = output_path.replace('.dxf', '_balanced_vis.png')
            print(f"3. バランス調整可視化画像を生成中: {vis_path}")
            pdf_processor.create_balanced_visualization(elements, vis_path)
        
        # 4. DXFファイルの作成
        print("4. バランス調整DXFファイルを生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        # スケールの設定
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        # 要素の追加
        dxf_elements = {
            'lines': elements['lines'],
            'walls': elements['walls'],
            'circles': elements['circles']
        }
        
        dxf_writer.add_elements(dxf_elements)
        dxf_writer.save(output_path)
        
        # 5. 結果の表示
        print("=== バランス調整変換完了 ===")
        print(f"入力: {input_path}")
        print(f"出力: {output_path}")
        print(f"検出結果（バランス調整済み）:")
        print(f"  - 線分: {len(elements['lines'])}本")
        print(f"  - 壁: {len(elements['walls'])}個")
        print(f"  - 円: {len(elements['circles'])}個")
        print("※ ノイズ除去と情報保持のバランスを最適化")
        
        return True
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {str(e)}")
        return False


if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='バランス調整版 PDF to DXF 変換ツール')
    parser.add_argument('--input', '-i', required=True, help='入力PDFファイルのパス')
    parser.add_argument('--output', '-o', required=True, help='出力DXFファイルのパス')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール（1:scale）')
    parser.add_argument('--visualization', '-v', action='store_true', help='バランス調整可視化画像を生成する')
    
    args = parser.parse_args()
    
    # 日付時間プレフィックスを追加
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    
    # 出力ファイル名に日付時間を追加
    base_name = os.path.basename(args.output)
    dir_name = os.path.dirname(args.output)
    output_with_timestamp = os.path.join(dir_name, f"{timestamp}_{base_name}")
    
    # 変換実行
    success = convert_pdf_to_dxf_balanced(
        args.input, 
        output_with_timestamp, 
        args.scale, 
        args.visualization
    )
    
    if success:
        print("バランス調整変換が正常に完了しました。")
        sys.exit(0)
    else:
        print("変換に失敗しました。")
        sys.exit(1)
