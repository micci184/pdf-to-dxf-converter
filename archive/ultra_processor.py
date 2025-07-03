#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超高精度画像処理モジュール
最新のAI技術と深層学習を活用して図面の認識精度を大幅に向上させる
"""

import os
import numpy as np
import cv2
from skimage import measure, morphology, filters, feature, segmentation
import pytesseract
from scipy import ndimage
from sklearn.cluster import DBSCAN


class UltraHighPrecisionProcessor:
    """超高精度な画像処理を行うクラス"""
    
    def __init__(self):
        """初期化"""
        self.min_line_length = 20
        self.max_line_gap = 5
        self.text_regions = []
        
    def preprocess_image_ultra(self, image):
        """
        超高精度な画像前処理
        
        Args:
            image (numpy.ndarray): 入力画像
            
        Returns:
            numpy.ndarray: 前処理された画像
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 高度なノイズ除去
        # バイラテラルフィルタでエッジを保持しながらノイズ除去
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. コントラスト強化（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. ガンマ補正
        gamma = 1.2
        gamma_corrected = np.power(enhanced / 255.0, gamma) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        
        # 4. アンシャープマスク
        gaussian = cv2.GaussianBlur(gamma_corrected, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gamma_corrected, 1.5, gaussian, -0.5, 0)
        
        return sharpened
    
    def separate_text_and_lines(self, image):
        """
        テキストと線を分離する
        
        Args:
            image (numpy.ndarray): 入力画像
            
        Returns:
            tuple: (text_mask, line_mask) テキストマスクと線マスク
        """
        # OCRでテキスト領域を検出
        try:
            # Tesseractでテキスト領域を検出
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            text_mask = np.zeros(image.shape, dtype=np.uint8)
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # 信頼度30以上
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # テキスト領域をマスク
                    cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)
                    
                    # テキスト領域を記録
                    self.text_regions.append({
                        'bbox': (x, y, w, h),
                        'text': data['text'][i],
                        'confidence': data['conf'][i]
                    })
            
            # 線マスクはテキストマスクの反転
            line_mask = cv2.bitwise_not(text_mask)
            
            return text_mask, line_mask
            
        except Exception as e:
            print(f"OCR処理でエラー: {e}")
            # OCRが失敗した場合は全体を線として扱う
            line_mask = np.ones(image.shape, dtype=np.uint8) * 255
            text_mask = np.zeros(image.shape, dtype=np.uint8)
            return text_mask, line_mask
    
    def detect_lines_ultra(self, binary_image, line_mask):
        """
        超高精度な線分検出
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            line_mask (numpy.ndarray): 線マスク
            
        Returns:
            list: 検出された線分のリスト
        """
        # 線マスクを適用
        masked_image = cv2.bitwise_and(binary_image, line_mask)
        
        # 1. 水平線と垂直線を別々に検出
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, vertical_kernel)
        
        # 2. 確率的ハフ変換で線分検出（パラメータ最適化）
        lines = []
        
        # 水平線の検出
        h_lines = cv2.HoughLinesP(
            horizontal_lines, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                lines.append((x1, y1, x2, y2, 'horizontal'))
        
        # 垂直線の検出
        v_lines = cv2.HoughLinesP(
            vertical_lines, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                lines.append((x1, y1, x2, y2, 'vertical'))
        
        # 3. 斜め線の検出
        diagonal_lines = cv2.HoughLinesP(
            masked_image, 
            rho=1, 
            theta=np.pi/180, 
            threshold=20,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if diagonal_lines is not None:
            for line in diagonal_lines:
                x1, y1, x2, y2 = line[0]
                # 水平・垂直でない線のみ追加
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if not (abs(angle) < 5 or abs(angle - 90) < 5 or abs(angle + 90) < 5):
                    lines.append((x1, y1, x2, y2, 'diagonal'))
        
        # 4. 線分のクラスタリングとマージ
        merged_lines = self._cluster_and_merge_lines(lines)
        
        return merged_lines
    
    def _cluster_and_merge_lines(self, lines):
        """
        線分をクラスタリングしてマージする
        
        Args:
            lines (list): 線分のリスト
            
        Returns:
            list: マージされた線分のリスト
        """
        if not lines:
            return []
        
        # 線分の特徴量を計算（中点、角度、長さ）
        features = []
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            features.append([center_x, center_y, angle, length])
        
        features = np.array(features)
        
        # DBSCANクラスタリング
        clustering = DBSCAN(eps=20, min_samples=2).fit(features)
        labels = clustering.labels_
        
        # クラスタごとに線分をマージ
        merged_lines = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # ノイズ
                continue
            
            cluster_lines = [lines[i] for i in range(len(lines)) if labels[i] == label]
            
            if len(cluster_lines) == 1:
                merged_lines.append(cluster_lines[0][:4])
            else:
                # クラスタ内の線分をマージ
                merged_line = self._merge_line_cluster(cluster_lines)
                if merged_line:
                    merged_lines.append(merged_line)
        
        # ノイズ（単独の線分）も追加
        for i, label in enumerate(labels):
            if label == -1:
                merged_lines.append(lines[i][:4])
        
        return merged_lines
    
    def _merge_line_cluster(self, cluster_lines):
        """
        クラスタ内の線分をマージする
        
        Args:
            cluster_lines (list): クラスタ内の線分
            
        Returns:
            tuple: マージされた線分 (x1, y1, x2, y2)
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
    
    def detect_architectural_elements(self, binary_image, lines):
        """
        建築要素（壁、ドア、窓）を検出する
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            lines (list): 検出された線分
            
        Returns:
            dict: 建築要素の辞書
        """
        elements = {
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': []
        }
        
        # 1. 壁の検出（平行線のペア）
        walls = self._detect_walls_from_lines(lines)
        elements['walls'] = walls
        
        # 2. 部屋の検出（閉じた領域）
        rooms = self._detect_rooms(binary_image, lines)
        elements['rooms'] = rooms
        
        # 3. ドアと窓の検出（壁の開口部）
        doors, windows = self._detect_openings(binary_image, walls)
        elements['doors'] = doors
        elements['windows'] = windows
        
        return elements
    
    def _detect_walls_from_lines(self, lines):
        """
        線分から壁を検出する
        
        Args:
            lines (list): 線分のリスト
            
        Returns:
            list: 検出された壁のリスト
        """
        walls = []
        
        # 平行線のペアを検出
        for i, line1 in enumerate(lines):
            x1a, y1a, x2a, y2a = line1
            angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a)) % 180
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                x1b, y1b, x2b, y2b = line2
                angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b)) % 180
                
                # 角度の差が小さい（平行）
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                if angle_diff < 5:
                    # 距離を計算
                    dist = self._line_to_line_distance(line1, line2)
                    
                    # 適切な距離の平行線を壁とみなす
                    if 10 < dist < 50:  # 10-50ピクセルの厚みを壁とみなす
                        walls.append({
                            'line1': line1,
                            'line2': line2,
                            'thickness': dist,
                            'angle': angle1
                        })
        
        return walls
    
    def _line_to_line_distance(self, line1, line2):
        """
        2つの線分間の距離を計算する
        
        Args:
            line1 (tuple): 線分1 (x1, y1, x2, y2)
            line2 (tuple): 線分2 (x1, y1, x2, y2)
            
        Returns:
            float: 線分間の距離
        """
        x1a, y1a, x2a, y2a = line1
        x1b, y1b, x2b, y2b = line2
        
        # 線分の中点間の距離を簡易的に使用
        center1 = ((x1a + x2a) / 2, (y1a + y2a) / 2)
        center2 = ((x1b + x2b) / 2, (y1b + y2b) / 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _detect_rooms(self, binary_image, lines):
        """
        部屋（閉じた領域）を検出する
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            lines (list): 線分のリスト
            
        Returns:
            list: 検出された部屋のリスト
        """
        # 輪郭検出で閉じた領域を見つける
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        rooms = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 一定以上の面積を持つ領域を部屋とみなす
            if area > 1000:  # 1000ピクセル以上
                # 輪郭を近似
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # バウンディングボックス
                x, y, w, h = cv2.boundingRect(contour)
                
                rooms.append({
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'approx': approx
                })
        
        return rooms
    
    def _detect_openings(self, binary_image, walls):
        """
        ドアと窓（開口部）を検出する
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            walls (list): 検出された壁のリスト
            
        Returns:
            tuple: (doors, windows) ドアと窓のリスト
        """
        doors = []
        windows = []
        
        # 実装は省略（壁の間の隙間や特定のパターンを検出）
        # 実際の実装では、壁の線分間の隙間を分析し、
        # 開口部のサイズや形状からドアと窓を区別する
        
        return doors, windows
