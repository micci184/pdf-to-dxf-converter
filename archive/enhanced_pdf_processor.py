#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
拡張PDFプロセッサークラス
PDFファイルを読み込み、高度な画像処理とAI技術を活用して図形要素を検出する
"""

import os
import numpy as np
import cv2
import fitz  # PyMuPDF
import pdf2image
from skimage import measure, morphology
from enhanced_image_processor import EnhancedImageProcessor


class EnhancedPDFProcessor:
    """PDFファイルを処理し、図形要素を検出するクラス"""
    
    def __init__(self, pdf_path):
        """
        初期化
        
        Args:
            pdf_path (str): PDFファイルのパス
        """
        self.pdf_path = pdf_path
        self.images = []
        self.image_processor = EnhancedImageProcessor()
        self.load_pdf()
    
    def load_pdf(self):
        """PDFファイルを読み込み、画像に変換"""
        try:
            # PDFを画像に変換（高解像度）
            images = pdf2image.convert_from_path(self.pdf_path, dpi=600)
            
            # OpenCV形式に変換
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
                
            print(f"PDFから{len(self.images)}ページを読み込みました")
        except Exception as e:
            raise Exception(f"PDFの読み込みに失敗しました: {str(e)}")
    
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
            'walls': []
        }
        
        for i, image in enumerate(self.images):
            # 画像の品質向上
            enhanced = self.image_processor.enhance_image(image)
            
            # 適応的二値化
            binary = self.image_processor.adaptive_threshold(enhanced)
            
            # 線分検出（高度なアルゴリズム）
            lines = self.image_processor.detect_lines_advanced(binary)
            for line in lines:
                elements['lines'].append(line)
            
            # 円検出（高度なアルゴリズム）
            circles = self.image_processor.detect_circles_advanced(binary, enhanced)
            for circle in circles:
                elements['circles'].append(circle)
            
            # 長方形検出（高度なアルゴリズム）
            rectangles = self.image_processor.detect_rectangles_advanced(binary)
            for rect in rectangles:
                elements['rectangles'].append(rect)
            
            # 壁検出
            walls = self.image_processor.detect_walls(binary)
            for wall in walls:
                elements['walls'].append(wall)
        
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
        
        # 最初のページのみ処理
        image = self.images[0].copy()
        
        # 画像の品質向上
        enhanced = self.image_processor.enhance_image(image)
        
        # 適応的二値化
        binary = self.image_processor.adaptive_threshold(enhanced)
        
        # 要素検出
        elements = {}
        
        # 線分検出
        elements['lines'] = self.image_processor.detect_lines_advanced(binary)
        
        # 円検出
        elements['circles'] = self.image_processor.detect_circles_advanced(binary, enhanced)
        
        # 長方形検出
        elements['rectangles'] = self.image_processor.detect_rectangles_advanced(binary)
        
        # 壁検出
        elements['walls'] = self.image_processor.detect_walls(binary)
        
        # 検出結果の可視化
        vis_image = self.image_processor.visualize_detection(image, elements)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image
