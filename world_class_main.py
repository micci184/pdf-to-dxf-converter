#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
世界最高レベル AI変換ツール - メインプログラム
手書き図面を最高精度でDXF/JWW形式に変換
"""

import os
import sys
import argparse
from datetime import datetime
from ai_converter_core import WorldClassAIConverter, create_output_filename
from enhanced_dxf_writer import EnhancedDXFWriter
import cv2
import numpy as np


class WorldClassConverter:
    """世界最高レベル変換システム"""
    
    def __init__(self, pdf_path):
        """初期化"""
        self.pdf_path = pdf_path
        self.ai_converter = WorldClassAIConverter()
        self.images = []
    
    def process_pdf_world_class(self):
        """世界最高レベルPDF処理"""
        try:
            print("🚀 世界最高レベル AI変換開始")
            
            # 1. 超高品質PDF読み込み
            print("📖 超高品質PDF読み込み中...")
            self.images = self.ai_converter.load_pdf_ultra_high_quality(self.pdf_path)
            
            all_elements = {
                'lines': [],
                'walls': [],
                'circles': [],
                'rectangles': [],
                'text_regions': []
            }
            
            for i, image in enumerate(self.images):
                print(f"🔍 ページ {i+1} を世界最高レベル処理中...")
                
                # 2. AI強化前処理
                enhanced, binary = self.ai_converter.ai_enhanced_preprocessing(image)
                
                # 3. AI文字認識
                text_regions = self.ai_converter.ai_text_recognition(enhanced)
                all_elements['text_regions'].extend(text_regions)
                
                # 4. AI線分検出
                lines = self.ai_converter.ai_line_detection(binary, text_regions)
                all_elements['lines'].extend([line[:4] for line in lines])
                
                # 5. AI図形認識
                shapes = self.ai_converter.ai_shape_recognition(binary)
                
                for shape in shapes:
                    if shape['type'] == 'circle':
                        all_elements['circles'].append(shape)
                    elif shape['type'] == 'rectangle':
                        all_elements['rectangles'].append(shape)
                
                # 6. AI壁検出
                walls = self._detect_walls_ai(lines)
                all_elements['walls'].extend(walls)
                
                print(f"✅ ページ {i+1} 完了:")
                print(f"   📏 線分: {len(lines)}本")
                print(f"   🏠 壁: {len(walls)}個")
                print(f"   ⭕ 円: {len([s for s in shapes if s['type'] == 'circle'])}個")
                print(f"   📝 テキスト: {len(text_regions)}個")
            
            return all_elements
            
        except Exception as e:
            print(f"❌ 処理エラー: {str(e)}")
            return None
    
    def _detect_walls_ai(self, lines):
        """AI壁検出"""
        walls = []
        
        # 長い線分のみを対象
        long_lines = [line for line in lines if np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) > 50]
        
        for i, line1 in enumerate(long_lines):
            x1a, y1a, x2a, y2a = line1[:4]
            angle1 = np.degrees(np.arctan2(y2a - y1a, x2a - x1a)) % 180
            
            for j, line2 in enumerate(long_lines[i+1:], i+1):
                x1b, y1b, x2b, y2b = line2[:4]
                angle2 = np.degrees(np.arctan2(y2b - y1b, x2b - x1b)) % 180
                
                # 平行線判定
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                if angle_diff < 10:
                    # 距離計算
                    center1 = ((x1a + x2a) / 2, (y1a + y2a) / 2)
                    center2 = ((x1b + x2b) / 2, (y1b + y2b) / 2)
                    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    # 壁として認識
                    if 20 < dist < 100:
                        walls.append({
                            'line1': line1[:4],
                            'line2': line2[:4],
                            'thickness': dist,
                            'angle': angle1
                        })
        
        return walls
    
    def create_world_class_visualization(self, elements, output_path):
        """世界最高レベル可視化"""
        if not self.images:
            return
        
        # 高解像度可視化画像を作成
        vis_image = self.images[0].copy()
        
        # 線分描画（青、細線）
        for line in elements['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 壁描画（緑、太線）
        for wall in elements['walls']:
            line1 = wall['line1']
            line2 = wall['line2']
            x1, y1, x2, y2 = line1
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            x1, y1, x2, y2 = line2
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 円描画（赤）
        for circle in elements['circles']:
            center = circle['center']
            radius = circle['radius']
            cv2.circle(vis_image, center, radius, (0, 0, 255), 2)
        
        # テキスト領域描画（黄色）
        for text_region in elements['text_regions']:
            x1, y1, x2, y2 = text_region['bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # テキスト表示
            cv2.putText(vis_image, text_region['text'], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imwrite(output_path, vis_image)
        print(f"🎨 世界最高レベル可視化保存: {output_path}")


def convert_pdf_world_class(input_path, output_path, scale=100, visualization=True):
    """世界最高レベルPDF変換"""
    try:
        print("=" * 60)
        print("🌟 世界最高レベル AI変換ツール 🌟")
        print("=" * 60)
        
        # 1. 変換システム初期化
        converter = WorldClassConverter(input_path)
        
        # 2. 世界最高レベル処理
        elements = converter.process_pdf_world_class()
        
        if elements is None:
            return False
        
        # 3. 可視化生成
        if visualization:
            vis_path = output_path.replace('.dxf', '_world_class_vis.png')
            converter.create_world_class_visualization(elements, vis_path)
        
        # 4. DXF生成
        print("📐 世界最高レベルDXF生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        # テキスト情報も含めてDXFに追加
        dxf_elements = {
            'lines': elements['lines'],
            'walls': elements['walls'],
            'circles': elements['circles'],
            'text_regions': elements['text_regions']
        }
        
        dxf_writer.add_elements(dxf_elements)
        dxf_writer.save(output_path)
        
        # 5. 結果表示
        print("=" * 60)
        print("🎉 世界最高レベル変換完了 🎉")
        print("=" * 60)
        print(f"📁 入力: {input_path}")
        print(f"📁 出力: {output_path}")
        print(f"📊 検出結果:")
        print(f"   📏 線分: {len(elements['lines'])}本")
        print(f"   🏠 壁: {len(elements['walls'])}個")
        print(f"   ⭕ 円: {len(elements['circles'])}個")
        print(f"   📝 テキスト: {len(elements['text_regions'])}個")
        print("🚀 最新AI技術による超高精度変換完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🌟 世界最高レベル AI変換ツール 🌟')
    parser.add_argument('--input', '-i', required=True, help='入力PDFファイル')
    parser.add_argument('--output', '-o', help='出力DXFファイル（省略時は自動生成）')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール')
    parser.add_argument('--no-visualization', action='store_true', help='可視化を無効化')
    
    args = parser.parse_args()
    
    # 出力ファイル名の生成
    if args.output:
        output_path = create_output_filename(os.path.basename(args.output))
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = create_output_filename(f"{base_name}_world_class.dxf")
    
    # 変換実行
    success = convert_pdf_world_class(
        args.input,
        output_path,
        args.scale,
        not args.no_visualization
    )
    
    if success:
        print("✅ 世界最高レベル変換が正常に完了しました！")
        sys.exit(0)
    else:
        print("❌ 変換に失敗しました。")
        sys.exit(1)
