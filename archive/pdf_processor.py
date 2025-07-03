#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDFプロセッサークラス
PDFファイルを読み込み、図形要素を検出する
"""

import os
import numpy as np
import cv2
import fitz  # PyMuPDF
import pdf2image
from skimage import measure, morphology
import tensorflow as tf


class PDFProcessor:
    """PDFファイルを処理し、図形要素を検出するクラス"""
    
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
            # PDFを画像に変換
            images = pdf2image.convert_from_path(self.pdf_path, dpi=300)
            
            # OpenCV形式に変換
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
                
            print(f"PDFから{len(self.images)}ページを読み込みました")
        except Exception as e:
            raise Exception(f"PDFの読み込みに失敗しました: {str(e)}")
    
    def preprocess_image(self, image):
        """
        画像の前処理を行う
        
        Args:
            image (numpy.ndarray): 入力画像
            
        Returns:
            numpy.ndarray: 前処理された画像
        """
        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ノイズ除去
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 二値化
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # モルフォロジー演算でノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_lines(self, binary_image):
        """
        線分を検出する
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            
        Returns:
            list: 検出された線分のリスト [(x1, y1, x2, y2), ...]
        """
        # 確率的ハフ変換で線分検出
        lines = cv2.HoughLinesP(
            binary_image, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50, 
            minLineLength=30, 
            maxLineGap=10
        )
        
        result = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                result.append((x1, y1, x2, y2))
                
        return result
    
    def detect_circles(self, binary_image):
        """
        円を検出する
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            
        Returns:
            list: 検出された円のリスト [(x, y, radius), ...]
        """
        # ハフ変換で円検出
        circles = cv2.HoughCircles(
            binary_image, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=5, 
            maxRadius=100
        )
        
        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                result.append((x, y, r))
                
        return result
    
    def detect_rectangles(self, binary_image):
        """
        長方形を検出する
        
        Args:
            binary_image (numpy.ndarray): 二値化画像
            
        Returns:
            list: 検出された長方形のリスト [(x, y, w, h), ...]
        """
        # 輪郭検出
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        result = []
        for contour in contours:
            # 輪郭を近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 4点で構成される輪郭を長方形とみなす
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                result.append((x, y, w, h))
                
        return result
    
    def detect_text(self, image):
        """
        テキストを検出する（実際のOCRはここで実装）
        
        Args:
            image (numpy.ndarray): 入力画像
            
        Returns:
            list: 検出されたテキストのリスト [(x, y, text), ...]
        """
        # ここではダミーのテキスト検出を返す
        # 実際のアプリケーションではTesseractやGoogleのOCR APIなどを使用
        return []
    
    def detect_elements(self):
        """
        全ての図形要素を検出する
        
        Returns:
            dict: 検出された図形要素の辞書
        """
        elements = {
            'lines': [],
            'circles': [],
            'rectangles': [],
            'text': []
        }
        
        for i, image in enumerate(self.images):
            # 画像の前処理
            preprocessed = self.preprocess_image(image)
            
            # 線分検出
            lines = self.detect_lines(preprocessed)
            for line in lines:
                elements['lines'].append({
                    'page': i,
                    'coords': line,
                    'type': 'line'
                })
            
            # 円検出
            circles = self.detect_circles(preprocessed)
            for circle in circles:
                elements['circles'].append({
                    'page': i,
                    'coords': circle,
                    'type': 'circle'
                })
            
            # 長方形検出
            rectangles = self.detect_rectangles(preprocessed)
            for rect in rectangles:
                elements['rectangles'].append({
                    'page': i,
                    'coords': rect,
                    'type': 'rectangle'
                })
            
            # テキスト検出
            text = self.detect_text(image)
            for t in text:
                elements['text'].append({
                    'page': i,
                    'coords': (t[0], t[1]),
                    'content': t[2],
                    'type': 'text'
                })
        
        return elements
    
    def visualize_detection(self, output_path=None):
        """
        検出結果を可視化する
        
        Args:
            output_path (str, optional): 出力画像のパス
            
        Returns:
            numpy.ndarray: 可視化された画像
        """
        if not self.images:
            raise Exception("PDFが読み込まれていません")
        
        # 最初のページのみ可視化
        image = self.images[0].copy()
        
        # 前処理
        preprocessed = self.preprocess_image(image)
        
        # 線分検出
        lines = self.detect_lines(preprocessed)
        for x1, y1, x2, y2 in lines:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 円検出
        circles = self.detect_circles(preprocessed)
        for x, y, r in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        
        # 長方形検出
        rectangles = self.detect_rectangles(preprocessed)
        for x, y, w, h in rectangles:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
