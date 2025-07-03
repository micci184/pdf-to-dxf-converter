#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
拡張画像処理モジュール
高度な画像処理とAI技術を活用して図面の認識精度を向上させる
"""

import os
import numpy as np
import cv2
from skimage import measure, morphology, filters, feature
import matplotlib.pyplot as plt


class EnhancedImageProcessor:
    """高度な画像処理を行うクラス"""
    
    def __init__(self):
        """初期化"""
        pass
    
    def enhance_image(self, image):
        """
        画像の品質を向上させる
        
        Args:
            image (numpy.ndarray): 入力画像
            
        Returns:
            numpy.ndarray: 品質向上後の画像
        """
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # ノイズ除去（非局所平均法）
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # アンシャープマスク（シャープネス向上）
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return sharpened
    
    def adaptive_threshold(self, image, block_size=11, c=2):
        """
        適応的二値化処理
        
        Args:
            image (numpy.ndarray): 入力画像
            block_size (int): ブロックサイズ
            c (int): 定数C
            
        Returns:
            numpy.ndarray: 二値化画像
        """
        # 適応的二値化
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            block_size, 
            c
        )
        
        # モルフォロジー演算でノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def detect_lines_advanced(self, binary_image):
        """
        高度な線分検出
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            
        Returns:
            list: 検出された線分のリスト [(x1, y1, x2, y2), ...]
        """
        # 確率的ハフ変換で線分検出（パラメータ調整）
        lines = cv2.HoughLinesP(
            binary_image, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,  # 閾値を下げて検出感度を上げる
            minLineLength=20,  # 最小線分長を短くして検出感度を上げる
            maxLineGap=20  # ギャップを大きくして途切れた線分も検出
        )
        
        result = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                result.append((x1, y1, x2, y2))
        
        # 短い線分をフィルタリング
        filtered_lines = []
        for x1, y1, x2, y2 in result:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 10:  # 10ピクセル以上の線分のみ採用
                filtered_lines.append((x1, y1, x2, y2))
        
        return self._merge_collinear_lines(filtered_lines)
    
    def _merge_collinear_lines(self, lines, max_angle_diff=5, max_distance=10):
        """
        同一直線上にある線分をマージする
        
        Args:
            lines (list): 線分のリスト [(x1, y1, x2, y2), ...]
            max_angle_diff (float): 許容される最大角度差（度）
            max_distance (float): 許容される最大距離
            
        Returns:
            list: マージされた線分のリスト
        """
        if not lines:
            return []
        
        # 線分の角度を計算
        angles = []
        for x1, y1, x2, y2 in lines:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angles.append(angle)
        
        # 類似した角度の線分をグループ化
        groups = []
        used = [False] * len(lines)
        
        for i in range(len(lines)):
            if used[i]:
                continue
                
            group = [i]
            used[i] = True
            
            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue
                    
                # 角度の差を計算
                angle_diff = min(abs(angles[i] - angles[j]), 180 - abs(angles[i] - angles[j]))
                
                if angle_diff < max_angle_diff:
                    # 距離をチェック
                    x1i, y1i, x2i, y2i = lines[i]
                    x1j, y1j, x2j, y2j = lines[j]
                    
                    # 簡易的な距離チェック（厳密には点と線分の距離を計算すべき）
                    dist1 = min(
                        np.sqrt((x1i - x1j)**2 + (y1i - y1j)**2),
                        np.sqrt((x1i - x2j)**2 + (y1i - y2j)**2),
                        np.sqrt((x2i - x1j)**2 + (y2i - y1j)**2),
                        np.sqrt((x2i - x2j)**2 + (y2i - y2j)**2)
                    )
                    
                    if dist1 < max_distance:
                        group.append(j)
                        used[j] = True
            
            groups.append(group)
        
        # 各グループ内の線分をマージ
        merged_lines = []
        for group in groups:
            if len(group) == 1:
                merged_lines.append(lines[group[0]])
            else:
                # グループ内の全ての点を集める
                points = []
                for idx in group:
                    x1, y1, x2, y2 = lines[idx]
                    points.append((x1, y1))
                    points.append((x2, y2))
                
                # 主成分分析で最適な直線を見つける（簡易実装）
                points = np.array(points)
                mean_point = np.mean(points, axis=0)
                
                # 共分散行列の固有ベクトルを計算
                x_centered = points[:, 0] - mean_point[0]
                y_centered = points[:, 1] - mean_point[1]
                cov = np.cov(x_centered, y_centered)
                
                # 最大固有値に対応する固有ベクトルが直線の方向
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                sort_indices = np.argsort(eigenvalues)[::-1]
                direction_vector = eigenvectors[:, sort_indices[0]]
                
                # 点群の両端を見つける
                min_projection = float('inf')
                max_projection = float('-inf')
                min_point = None
                max_point = None
                
                for point in points:
                    # 方向ベクトルへの射影
                    projection = np.dot(point - mean_point, direction_vector)
                    
                    if projection < min_projection:
                        min_projection = projection
                        min_point = point
                    
                    if projection > max_projection:
                        max_projection = projection
                        max_point = point
                
                # マージされた線分
                if min_point is not None and max_point is not None:
                    merged_lines.append((
                        int(min_point[0]), int(min_point[1]),
                        int(max_point[0]), int(max_point[1])
                    ))
        
        return merged_lines
    
    def detect_circles_advanced(self, binary_image, original_image):
        """
        高度な円検出
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            original_image (numpy.ndarray): 元の画像
            
        Returns:
            list: 検出された円のリスト [(x, y, radius), ...]
        """
        # 二値化画像の輪郭を検出
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        circles = []
        for contour in contours:
            # 輪郭の面積と周囲長を計算
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 面積が小さすぎる輪郭は無視
            if area < 50:
                continue
            
            # 円形度を計算（1に近いほど円に近い）
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 円形度が高い輪郭を円として検出
            if circularity > 0.7:  # 0.7以上を円とみなす
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # 半径が大きすぎる/小さすぎる円は除外
                if 5 <= radius <= 100:
                    circles.append((center[0], center[1], radius))
        
        # ハフ変換による円検出も併用
        hough_circles = cv2.HoughCircles(
            cv2.GaussianBlur(original_image, (5, 5), 0),
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )
        
        if hough_circles is not None:
            hough_circles = np.uint16(np.around(hough_circles))
            for circle in hough_circles[0, :]:
                x, y, r = circle
                circles.append((x, y, r))
        
        # 重複する円を除去
        filtered_circles = []
        for i, (x1, y1, r1) in enumerate(circles):
            is_duplicate = False
            for j, (x2, y2, r2) in enumerate(filtered_circles):
                # 中心点間の距離
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # 中心が近く、半径も近い円は重複とみなす
                if dist < max(r1, r2) * 0.5 and abs(r1 - r2) < max(r1, r2) * 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_circles.append((x1, y1, r1))
        
        return filtered_circles
    
    def detect_rectangles_advanced(self, binary_image):
        """
        高度な長方形検出
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            
        Returns:
            list: 検出された長方形のリスト [(x, y, w, h, angle), ...]
        """
        # 輪郭検出
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        rectangles = []
        for contour in contours:
            # 輪郭の面積
            area = cv2.contourArea(contour)
            
            # 小さすぎる輪郭は無視
            if area < 100:
                continue
            
            # 輪郭を近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 4点で構成される輪郭を長方形とみなす
            if len(approx) == 4:
                # 凸性チェック
                if cv2.isContourConvex(approx):
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # 回転した長方形を取得
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    
                    # 中心点、幅、高さ、角度
                    center = rect[0]
                    width, height = rect[1]
                    angle = rect[2]
                    
                    rectangles.append((
                        int(center[0]), int(center[1]),
                        int(width), int(height),
                        angle
                    ))
        
        return rectangles
    
    def detect_walls(self, binary_image, min_length=50):
        """
        壁を検出する（太い線や平行線のペアを検出）
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            min_length (int): 最小壁長さ
            
        Returns:
            list: 検出された壁のリスト [(x1, y1, x2, y2, thickness), ...]
        """
        # 距離変換で線の太さを測定
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        # 線分検出
        lines = self.detect_lines_advanced(binary_image)
        
        walls = []
        for x1, y1, x2, y2 in lines:
            # 線分の長さを計算
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 短すぎる線分は壁として扱わない
            if length < min_length:
                continue
            
            # 線分上の点をサンプリングして平均の太さを計算
            num_samples = 10
            thickness_sum = 0
            count = 0
            
            for i in range(num_samples):
                t = i / (num_samples - 1)
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                
                # 画像の範囲内かチェック
                if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
                    thickness = dist_transform[y, x] * 2  # 半径から直径へ
                    thickness_sum += thickness
                    count += 1
            
            # 平均の太さを計算
            avg_thickness = thickness_sum / count if count > 0 else 1
            
            # 太さが一定以上の線分を壁として検出
            if avg_thickness > 3:  # 3ピクセル以上の太さを持つ線分を壁とみなす
                walls.append((x1, y1, x2, y2, avg_thickness))
        
        return walls
    
    def detect_doors_and_windows(self, binary_image, walls):
        """
        ドアと窓を検出する
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            walls (list): 検出された壁のリスト
            
        Returns:
            tuple: (doors, windows) 検出されたドアと窓のリスト
        """
        # 実装は省略（実際には壁の間の隙間や特定のパターンを検出）
        return [], []
    
    def visualize_detection(self, image, elements):
        """
        検出結果を可視化する
        
        Args:
            image (numpy.ndarray): 入力画像
            elements (dict): 検出された要素
            
        Returns:
            numpy.ndarray: 可視化された画像
        """
        # 可視化用の画像をコピー
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        # 線分の描画
        for x1, y1, x2, y2 in elements.get('lines', []):
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 円の描画
        for x, y, r in elements.get('circles', []):
            cv2.circle(vis_image, (x, y), r, (0, 255, 0), 2)
        
        # 長方形の描画
        for x, y, w, h, angle in elements.get('rectangles', []):
            box = cv2.boxPoints(((x, y), (w, h), angle))
            box = np.int32(box)
            cv2.drawContours(vis_image, [box], 0, (255, 0, 0), 2)
        
        # 壁の描画
        for x1, y1, x2, y2, thickness in elements.get('walls', []):
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 255, 0), max(2, int(thickness)))
        
        return vis_image
