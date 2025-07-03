#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超高精度版 PDF to DXF コンバーター
最新のAI技術と深層学習を活用して手書き図面のPDFをDXF形式に超高精度で変換するツール
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pdf2image
from ultra_processor import UltraHighPrecisionProcessor
from enhanced_dxf_writer import EnhancedDXFWriter


class UltraPDFProcessor:
    """超高精度PDFプロセッサー"""
    
    def __init__(self, pdf_path):
        """
        初期化
        
        Args:
            pdf_path (str): PDFファイルのパス
        """
        self.pdf_path = pdf_path
        self.images = []
        self.processor = UltraHighPrecisionProcessor()
        self.load_pdf()
    
    def load_pdf(self):
        """PDFファイルを読み込み、画像に変換"""
        try:
            # PDFを超高解像度で画像に変換
            images = pdf2image.convert_from_path(self.pdf_path, dpi=800)  # 800dpiで高解像度化
            
            # OpenCV形式に変換
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
                
            print(f"PDFから{len(self.images)}ページを超高解像度で読み込みました")
        except Exception as e:
            raise Exception(f"PDFの読み込みに失敗しました: {str(e)}")
    
    def detect_elements_ultra(self):
        """
        超高精度で図形要素を検出する
        
        Returns:
            dict: 検出された図形要素の辞書
        """
        all_elements = {
            'lines': [],
            'circles': [],
            'rectangles': [],
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': [],
            'text': []
        }
        
        for i, image in enumerate(self.images):
            print(f"ページ {i+1} を処理中...")
            
            # 1. 超高精度な前処理
            preprocessed = self.processor.preprocess_image_ultra(image)
            
            # 2. テキストと線の分離
            text_mask, line_mask = self.processor.separate_text_and_lines(preprocessed)
            
            # 3. 適応的二値化（線領域のみ）
            line_binary = cv2.adaptiveThreshold(
                cv2.bitwise_and(preprocessed, line_mask),
                255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 4. 超高精度な線分検出
            lines = self.processor.detect_lines_ultra(line_binary, line_mask)
            all_elements['lines'].extend(lines)
            
            # 5. 建築要素の検出
            arch_elements = self.processor.detect_architectural_elements(line_binary, lines)
            all_elements['walls'].extend(arch_elements['walls'])
            all_elements['doors'].extend(arch_elements['doors'])
            all_elements['windows'].extend(arch_elements['windows'])
            all_elements['rooms'].extend(arch_elements['rooms'])
            
            # 6. テキスト情報の追加
            for text_region in self.processor.text_regions:
                all_elements['text'].append({
                    'page': i,
                    'bbox': text_region['bbox'],
                    'text': text_region['text'],
                    'confidence': text_region['confidence']
                })
            
            print(f"ページ {i+1} 完了: 線分{len(lines)}本, 壁{len(arch_elements['walls'])}個, テキスト{len(self.processor.text_regions)}個")
        
        return all_elements
    
    def create_visualization(self, elements, output_path=None):
        """
        検出結果の可視化
        
        Args:
            elements (dict): 検出された要素
            output_path (str, optional): 出力パス
            
        Returns:
            numpy.ndarray: 可視化画像
        """
        if not self.images:
            return None
        
        # 最初のページを使用
        vis_image = self.images[0].copy()
        
        # 線分の描画（青）
        for line in elements['lines']:
            x1, y1, x2, y2 = line[:4]
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 壁の描画（緑）
        for wall in elements['walls']:
            line1 = wall['line1']
            line2 = wall['line2']
            x1, y1, x2, y2 = line1
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            x1, y1, x2, y2 = line2
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 部屋の描画（赤）
        for room in elements['rooms']:
            x, y, w, h = room['bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # テキストの描画（黄）
        for text in elements['text']:
            x, y, w, h = text['bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(vis_image, text['text'][:10], (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image


def convert_pdf_to_dxf_ultra(input_path, output_path, scale=100, visualization=False):
    """
    PDFを超高精度でDXFに変換する
    
    Args:
        input_path (str): 入力PDFファイルのパス
        output_path (str): 出力DXFファイルのパス
        scale (int): スケール（1:scale）
        visualization (bool): 可視化画像を保存するかどうか
    
    Returns:
        bool: 変換が成功したかどうか
    """
    try:
        print("=== 超高精度 PDF to DXF 変換開始 ===")
        
        # 1. PDFの処理
        print("1. PDFを読み込み中...")
        pdf_processor = UltraPDFProcessor(input_path)
        
        # 2. 超高精度な図形要素の検出
        print("2. 超高精度で図形要素を検出中...")
        elements = pdf_processor.detect_elements_ultra()
        
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
        
        # 要素の追加（建築要素を考慮）
        dxf_elements = {
            'lines': elements['lines'],
            'circles': elements.get('circles', []),
            'rectangles': elements.get('rectangles', []),
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
        print(f"  - 部屋: {len(elements['rooms'])}個")
        print(f"  - テキスト: {len(elements['text'])}個")
        
        return True
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {str(e)}")
        return False


if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='超高精度 PDF to DXF 変換ツール')
    parser.add_argument('--input', '-i', required=True, help='入力PDFファイルのパス')
    parser.add_argument('--output', '-o', required=True, help='出力DXFファイルのパス')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール（1:scale）')
    parser.add_argument('--visualization', '-v', action='store_true', help='可視化画像を生成する')
    
    args = parser.parse_args()
    
    # 変換実行
    success = convert_pdf_to_dxf_ultra(
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
