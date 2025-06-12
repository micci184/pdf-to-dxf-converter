#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CAD専用最適化版 PDF to DXF コンバーター
ノイズ除去と線の統合に特化し、CADソフトでの表示に最適化
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pdf2image
from sklearn.cluster import DBSCAN
from enhanced_dxf_writer import EnhancedDXFWriter


class CADOptimizedProcessor:
    """CAD専用最適化プロセッサー"""
    
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
            # PDFを適切な解像度で画像に変換（300dpiで高速化）
            images = pdf2image.convert_from_path(self.pdf_path, dpi=300)
            
            # OpenCV形式に変換
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
                
            print(f"PDFから{len(self.images)}ページを読み込みました（300dpi）")
        except Exception as e:
            raise Exception(f"PDFの読み込みに失敗しました: {str(e)}")
    
    def preprocess_image_cad(self, image):
        """
        CAD専用の画像前処理
        
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
        
        # 1. ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. 適応的二値化（より厳しい閾値）
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 4
        )
        
        # 3. モルフォロジー演算でノイズ除去（より強力）
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 4. 小さなノイズを除去
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # 50ピクセル未満の小さな領域を除去
                cv2.fillPoly(binary, [contour], 0)
        
        return binary
    
    def detect_main_lines_only(self, binary_image):
        """
        主要な線分のみを検出（ノイズ除去重視）
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            
        Returns:
            list: 検出された主要線分のリスト
        """
        # 1. 水平線と垂直線を強調（より大きなカーネル）
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)
        
        # 2. 主要線分のみ検出（より厳しい閾値）
        lines = []
        
        # 水平線の検出（厳しい条件）
        h_lines = cv2.HoughLinesP(
            horizontal_lines, rho=1, theta=np.pi/180, threshold=80,
            minLineLength=50, maxLineGap=15
        )
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 30:  # 30ピクセル以上の線のみ
                    lines.append((x1, y1, x2, y2, 'horizontal'))
        
        # 垂直線の検出（厳しい条件）
        v_lines = cv2.HoughLinesP(
            vertical_lines, rho=1, theta=np.pi/180, threshold=80,
            minLineLength=50, maxLineGap=15
        )
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 30:  # 30ピクセル以上の線のみ
                    lines.append((x1, y1, x2, y2, 'vertical'))
        
        # 3. その他の主要線分（より厳しい条件）
        other_lines = cv2.HoughLinesP(
            binary_image, rho=1, theta=np.pi/180, threshold=60,
            minLineLength=40, maxLineGap=20
        )
        if other_lines is not None:
            for line in other_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                
                # 長い線のみ、かつ水平・垂直でない線
                if length > 40 and not (abs(angle) < 15 or abs(angle - 90) < 15):
                    lines.append((x1, y1, x2, y2, 'other'))
        
        # 4. 線分の統合とクリーンアップ
        cleaned_lines = self._merge_similar_lines(lines)
        
        return cleaned_lines
    
    def _merge_similar_lines(self, lines):
        """
        類似した線分を統合する
        
        Args:
            lines (list): 線分のリスト
            
        Returns:
            list: 統合された線分のリスト
        """
        if not lines:
            return []
        
        # 線分の特徴量を計算
        features = []
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            features.append([center_x, center_y, angle, length])
        
        features = np.array(features)
        
        # DBSCANクラスタリング（より厳しい条件）
        clustering = DBSCAN(eps=30, min_samples=2).fit(features)
        labels = clustering.labels_
        
        # クラスタごとに線分を統合
        merged_lines = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # ノイズは除外
                continue
            
            cluster_lines = [lines[i] for i in range(len(lines)) if labels[i] == label]
            
            if len(cluster_lines) == 1:
                merged_lines.append(cluster_lines[0][:4])
            else:
                # クラスタ内の線分を統合
                merged_line = self._merge_line_cluster(cluster_lines)
                if merged_line:
                    merged_lines.append(merged_line)
        
        return merged_lines
    
    def _merge_line_cluster(self, cluster_lines):
        """
        クラスタ内の線分を統合する
        
        Args:
            cluster_lines (list): クラスタ内の線分
            
        Returns:
            tuple: 統合された線分 (x1, y1, x2, y2)
        """
        # 全ての端点を収集
        points = []
        for line in cluster_lines:
            x1, y1, x2, y2 = line[:4]
            points.extend([(x1, y1), (x2, y2)])
        
        points = np.array(points)
        
        # 主成分分析で最適な直線を求める
        mean_point = np.mean(points, axis=0)
        centered_points = points - mean_point
        
        # 共分散行列の固有ベクトル
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 最大固有値に対応する固有ベクトル（主方向）
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # 主方向への射影で端点を求める
        projections = np.dot(centered_points, main_direction)
        min_proj_idx = np.argmin(projections)
        max_proj_idx = np.argmax(projections)
        
        # 端点の座標
        start_point = points[min_proj_idx]
        end_point = points[max_proj_idx]
        
        return (int(start_point[0]), int(start_point[1]), 
                int(end_point[0]), int(end_point[1]))
    
    def detect_walls_conservative(self, lines):
        """
        保守的な壁検出（主要な壁のみ）
        
        Args:
            lines (list): 線分のリスト
            
        Returns:
            list: 検出された主要壁のリスト
        """
        walls = []
        
        # 長い線分のみを対象とする
        long_lines = [line for line in lines if np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) > 80]
        
        # 平行線のペアを検出（より厳しい条件）
        for i, line1 in enumerate(long_lines):
            x1a, y1a, x2a, y2a = line1[:4]
            angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a)) % 180
            
            for j, line2 in enumerate(long_lines[i+1:], i+1):
                x1b, y1b, x2b, y2b = line2[:4]
                angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b)) % 180
                
                # 角度の差が小さい（平行）
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                if angle_diff < 5:
                    # 距離を計算
                    center1 = ((x1a + x2a) / 2, (y1a + y2a) / 2)
                    center2 = ((x1b + x2b) / 2, (y1b + y2b) / 2)
                    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    # 適切な距離の平行線を壁とみなす（より厳しい条件）
                    if 20 < dist < 60:
                        walls.append({
                            'line1': line1[:4],
                            'line2': line2[:4],
                            'thickness': dist,
                            'angle': angle1
                        })
        
        return walls
    
    def process_pdf_cad_optimized(self):
        """
        CAD最適化処理でPDFを処理
        
        Returns:
            dict: 検出された主要図形要素
        """
        all_elements = {
            'lines': [],
            'walls': []
        }
        
        for i, image in enumerate(self.images):
            print(f"ページ {i+1} を処理中（CAD最適化）...")
            
            # 1. CAD専用前処理
            binary = self.preprocess_image_cad(image)
            
            # 2. 主要線分のみ検出
            lines = self.detect_main_lines_only(binary)
            all_elements['lines'].extend([line[:4] for line in lines])
            
            # 3. 保守的な壁検出
            walls = self.detect_walls_conservative(lines)
            all_elements['walls'].extend(walls)
            
            print(f"ページ {i+1} 完了: 線分{len(lines)}本, 壁{len(walls)}個（ノイズ除去済み）")
        
        return all_elements
    
    def create_clean_visualization(self, elements, output_path):
        """
        クリーンな可視化画像を作成
        
        Args:
            elements (dict): 検出された要素
            output_path (str): 出力パス
        """
        if not self.images:
            return
        
        # 白い背景を作成
        vis_image = np.ones_like(self.images[0]) * 255
        
        # 線分の描画（黒、細線）
        for line in elements['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 0), 1)
        
        # 壁の描画（赤、太線）
        for wall in elements['walls']:
            line1 = wall['line1']
            line2 = wall['line2']
            x1, y1, x2, y2 = line1
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            x1, y1, x2, y2 = line2
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imwrite(output_path, vis_image)
        print(f"クリーンな可視化画像を保存: {output_path}")


def convert_pdf_to_dxf_cad_optimized(input_path, output_path, scale=100, visualization=False):
    """
    PDFをCAD最適化版でDXFに変換する
    
    Args:
        input_path (str): 入力PDFファイルのパス
        output_path (str): 出力DXFファイルのパス
        scale (int): スケール（1:scale）
        visualization (bool): 可視化画像を保存するかどうか
    
    Returns:
        bool: 変換が成功したかどうか
    """
    try:
        print("=== CAD最適化版 PDF to DXF 変換開始 ===")
        
        # 1. PDFの処理
        print("1. PDFを読み込み中...")
        pdf_processor = CADOptimizedProcessor(input_path)
        
        # 2. CAD最適化処理
        print("2. CAD最適化処理で図形要素を検出中...")
        elements = pdf_processor.process_pdf_cad_optimized()
        
        # 3. 可視化（オプション）
        if visualization:
            vis_path = output_path.replace('.dxf', '_cad_clean.png')
            print(f"3. クリーンな可視化画像を生成中: {vis_path}")
            pdf_processor.create_clean_visualization(elements, vis_path)
        
        # 4. DXFファイルの作成
        print("4. CAD最適化DXFファイルを生成中...")
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
        print("=== CAD最適化変換完了 ===")
        print(f"入力: {input_path}")
        print(f"出力: {output_path}")
        print(f"検出結果（ノイズ除去済み）:")
        print(f"  - 主要線分: {len(elements['lines'])}本")
        print(f"  - 主要壁: {len(elements['walls'])}個")
        print("※ CADソフト（JW-CAD、AutoCAD等）での表示に最適化されています")
        
        return True
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {str(e)}")
        return False


if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='CAD最適化版 PDF to DXF 変換ツール')
    parser.add_argument('--input', '-i', required=True, help='入力PDFファイルのパス')
    parser.add_argument('--output', '-o', required=True, help='出力DXFファイルのパス')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール（1:scale）')
    parser.add_argument('--visualization', '-v', action='store_true', help='クリーンな可視化画像を生成する')
    
    args = parser.parse_args()
    
    # 変換実行
    success = convert_pdf_to_dxf_cad_optimized(
        args.input, 
        args.output, 
        args.scale, 
        args.visualization
    )
    
    if success:
        print("CAD最適化変換が正常に完了しました。")
        sys.exit(0)
    else:
        print("変換に失敗しました。")
        sys.exit(1)
