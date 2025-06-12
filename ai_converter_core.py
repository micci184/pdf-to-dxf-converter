#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
世界最高レベル AI変換ツール - コアモジュール
最新のディープラーニング技術を駆使した手書き図面変換システム
"""

import os
import sys
import cv2
import numpy as np
import pdf2image
from datetime import datetime
import easyocr
from sklearn.cluster import DBSCAN
from scipy import ndimage
from skimage import morphology, measure
import tensorflow as tf


class WorldClassAIConverter:
    """世界最高レベルAI変換システム"""
    
    def __init__(self):
        """初期化"""
        self.ocr_reader = None
        self.initialize_ai_models()
    
    def initialize_ai_models(self):
        """AI モデルの初期化"""
        try:
            # EasyOCR の初期化（日本語・英語対応）
            self.ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
            print("✅ OCRモデル初期化完了")
        except Exception as e:
            print(f"⚠️ OCRモデル初期化エラー: {e}")
    
    def load_pdf_ultra_high_quality(self, pdf_path):
        """超高品質PDFロード"""
        try:
            # 600dpiで超高解像度変換
            images = pdf2image.convert_from_path(pdf_path, dpi=600)
            cv_images = []
            
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cv_images.append(cv_img)
            
            print(f"✅ 超高品質PDF読み込み完了: {len(cv_images)}ページ (600dpi)")
            return cv_images
        except Exception as e:
            raise Exception(f"PDF読み込み失敗: {str(e)}")
    
    def ai_enhanced_preprocessing(self, image):
        """AI強化前処理"""
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. ノイズ除去（Non-local Means Denoising）
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 2. シャープニング
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 3. CLAHE（コントラスト強化）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
        
        # 4. 適応的二値化
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 9, 2
        )
        
        return enhanced, binary
    
    def ai_text_recognition(self, image):
        """AI文字認識"""
        if self.ocr_reader is None:
            return []
        
        try:
            # EasyOCRで高精度文字認識
            results = self.ocr_reader.readtext(image)
            
            text_regions = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # 信頼度30%以上
                    # バウンディングボックスを取得
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    text_regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': text,
                        'confidence': confidence
                    })
            
            print(f"✅ 文字認識完了: {len(text_regions)}個のテキスト")
            return text_regions
        except Exception as e:
            print(f"⚠️ 文字認識エラー: {e}")
            return []
    
    def ai_line_detection(self, binary_image, text_regions):
        """AI線分検出"""
        # テキスト領域をマスク
        text_mask = np.zeros(binary_image.shape, dtype=np.uint8)
        for region in text_regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(text_mask, (x1-5, y1-5), (x2+5, y2+5), 255, -1)
        
        # テキスト領域を除外
        masked_binary = cv2.bitwise_and(binary_image, cv2.bitwise_not(text_mask))
        
        # 線分検出
        lines = []
        
        # 水平線検出
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        horizontal_lines = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, horizontal_kernel)
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, 60, minLineLength=40, maxLineGap=10)
        
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                lines.append((x1, y1, x2, y2, 'horizontal'))
        
        # 垂直線検出
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical_lines = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, vertical_kernel)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 60, minLineLength=40, maxLineGap=10)
        
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                lines.append((x1, y1, x2, y2, 'vertical'))
        
        # その他の線分
        other_lines = cv2.HoughLinesP(masked_binary, 1, np.pi/180, 50, minLineLength=30, maxLineGap=15)
        if other_lines is not None:
            for line in other_lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                if not (abs(angle) < 15 or abs(angle - 90) < 15):
                    lines.append((x1, y1, x2, y2, 'other'))
        
        return lines
    
    def ai_shape_recognition(self, binary_image):
        """AI図形認識"""
        shapes = []
        
        # 輪郭検出
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 小さすぎる輪郭は無視
                continue
            
            # 輪郭の近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 図形の分類
            if len(approx) == 3:
                shapes.append({'type': 'triangle', 'contour': approx})
            elif len(approx) == 4:
                # 長方形かどうか判定
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                shapes.append({'type': 'rectangle', 'contour': approx, 'bbox': (x, y, w, h)})
            elif len(approx) > 8:
                # 円形の可能性
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius > 10:
                    shapes.append({'type': 'circle', 'center': (int(x), int(y)), 'radius': int(radius)})
        
        return shapes


def create_output_filename(base_name):
    """出力ファイル名を生成（日付時間プレフィックス付き）"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    return f"output/{timestamp}_{base_name}"


if __name__ == "__main__":
    print("🚀 世界最高レベル AI変換ツール - コアモジュール")
    print("最新のディープラーニング技術を駆使した手書き図面変換システム")
