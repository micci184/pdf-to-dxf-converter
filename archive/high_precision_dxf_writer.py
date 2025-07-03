#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高精度DXF書き出しモジュール
可視化と同じ品質でDXFファイルを生成
"""

import ezdxf
import numpy as np
from datetime import datetime


class HighPrecisionDXFWriter:
    """高精度DXF書き出しクラス"""
    
    def __init__(self):
        """初期化"""
        self.doc = ezdxf.new('R2010')
        self.msp = self.doc.modelspace()
        self.scale_factor = 1.0
        self.elements = {}
        self.setup_layers()
    
    def setup_layers(self):
        """レイヤー設定"""
        layers = [
            ('Lines', 1),      # 青
            ('Walls', 3),      # 緑
            ('Circles', 1),    # 赤
            ('Text', 6),       # 黄
            ('Structure', 2),  # 構造線
        ]
        
        for layer_name, color in layers:
            if layer_name not in self.doc.layers:
                layer = self.doc.layers.new(layer_name)
                layer.color = color
    
    def set_scale(self, scale):
        """スケール設定"""
        self.scale_factor = scale
    
    def add_elements(self, detected_elements):
        """検出された図形要素を追加"""
        self.elements = detected_elements
    
    def _transform_point(self, x, y):
        """座標変換（高精度）"""
        # スケール適用
        tx = float(x) * self.scale_factor
        ty = float(y) * self.scale_factor
        return tx, ty
    
    def _add_high_precision_lines(self):
        """高精度線分追加"""
        if 'lines' not in self.elements:
            return
        
        print("📏 高精度線分書き出し中...")
        line_count = 0
        
        for line in self.elements['lines']:
            if len(line) >= 4:
                x1, y1, x2, y2 = line[:4]
                
                # 座標変換
                tx1, ty1 = self._transform_point(x1, y1)
                tx2, ty2 = self._transform_point(x2, y2)
                
                # 線分の長さチェック
                length = np.sqrt((tx2 - tx1)**2 + (ty2 - ty1)**2)
                if length > 1.0:  # 1mm以上の線のみ
                    # Y座標を反転（DXF座標系）
                    self.msp.add_line(
                        (tx1, -ty1),
                        (tx2, -ty2),
                        dxfattribs={
                            'layer': 'Lines',
                            'lineweight': 25  # 0.25mm
                        }
                    )
                    line_count += 1
        
        print(f"✅ 線分 {line_count}本を高精度で書き出し")
    
    def _add_high_precision_walls(self):
        """高精度壁追加"""
        if 'walls' not in self.elements:
            return
        
        print("🏠 高精度壁書き出し中...")
        wall_count = 0
        
        for wall in self.elements['walls']:
            if isinstance(wall, dict) and 'line1' in wall and 'line2' in wall:
                # 壁の両側線を描画
                for line_key in ['line1', 'line2']:
                    x1, y1, x2, y2 = wall[line_key]
                    
                    # 座標変換
                    tx1, ty1 = self._transform_point(x1, y1)
                    tx2, ty2 = self._transform_point(x2, y2)
                    
                    # 線分の長さチェック
                    length = np.sqrt((tx2 - tx1)**2 + (ty2 - ty1)**2)
                    if length > 2.0:  # 2mm以上の線のみ
                        # Y座標を反転（DXF座標系）
                        self.msp.add_line(
                            (tx1, -ty1),
                            (tx2, -ty2),
                            dxfattribs={
                                'layer': 'Walls',
                                'lineweight': 50  # 0.5mm（太線）
                            }
                        )
                
                # 壁の厚みを表現（ハッチング）
                if 'thickness' in wall and wall['thickness'] > 5:
                    self._add_wall_hatch(wall)
                
                wall_count += 1
        
        print(f"✅ 壁 {wall_count}個を高精度で書き出し")
    
    def _add_wall_hatch(self, wall):
        """壁のハッチング追加"""
        try:
            line1 = wall['line1']
            line2 = wall['line2']
            
            # 4つの頂点を計算
            x1a, y1a, x2a, y2a = line1
            x1b, y1b, x2b, y2b = line2
            
            # 座標変換
            tx1a, ty1a = self._transform_point(x1a, y1a)
            tx2a, ty2a = self._transform_point(x2a, y2a)
            tx1b, ty1b = self._transform_point(x1b, y1b)
            tx2b, ty2b = self._transform_point(x2b, y2b)
            
            # 壁の領域をポリラインで作成
            points = [
                (tx1a, -ty1a),
                (tx2a, -ty2a),
                (tx2b, -ty2b),
                (tx1b, -ty1b)
            ]
            
            # ポリライン追加
            self.msp.add_lwpolyline(
                points,
                dxfattribs={
                    'layer': 'Walls',
                    'closed': True,
                    'lineweight': 25
                }
            )
            
        except Exception as e:
            print(f"⚠️ 壁ハッチングエラー: {e}")
    
    def _add_high_precision_circles(self):
        """高精度円追加"""
        if 'circles' not in self.elements:
            return
        
        print("⭕ 高精度円書き出し中...")
        circle_count = 0
        
        for circle in self.elements['circles']:
            if len(circle) >= 3:
                x, y, r = circle[:3]
                
                # 座標変換
                tx, ty = self._transform_point(x, y)
                tr = float(r) * self.scale_factor
                
                # 半径チェック
                if tr > 1.0:  # 1mm以上の円のみ
                    # Y座標を反転（DXF座標系）
                    self.msp.add_circle(
                        (tx, -ty),
                        tr,
                        dxfattribs={
                            'layer': 'Circles',
                            'lineweight': 25
                        }
                    )
                    circle_count += 1
        
        print(f"✅ 円 {circle_count}個を高精度で書き出し")
    
    def _add_high_precision_text(self):
        """高精度テキスト追加"""
        if 'text_regions' not in self.elements:
            return
        
        print("📝 高精度テキスト書き出し中...")
        text_count = 0
        
        for text_region in self.elements['text_regions']:
            if 'bbox' in text_region and 'text' in text_region:
                x1, y1, x2, y2 = text_region['bbox']
                text = text_region['text']
                
                # テキストの中心座標
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 座標変換
                tx, ty = self._transform_point(center_x, center_y)
                
                # テキストサイズ計算
                text_height = y2 - y1
                text_size = max(text_height * self.scale_factor, 2.0)  # 最小2mm
                
                # テキスト追加（位置指定を修正）
                text_entity = self.msp.add_text(
                    text,
                    dxfattribs={
                        'layer': 'Text',
                        'height': text_size,
                        'style': 'Standard',
                        'insert': (tx, -ty),  # 位置を直接指定
                        'halign': 1,  # 中央揃え
                        'valign': 1   # 中央揃え
                    }
                )
                
                text_count += 1
        
        print(f"✅ テキスト {text_count}個を高精度で書き出し")
    
    def _add_construction_lines(self):
        """補助線追加（構造理解用）"""
        if 'lines' not in self.elements:
            return
        
        # 主要な構造線を抽出
        structural_lines = []
        for line in self.elements['lines']:
            if len(line) >= 4:
                x1, y1, x2, y2 = line[:4]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # 長い線（100ピクセル以上）を構造線として扱う
                if length > 100:
                    structural_lines.append(line)
        
        print(f"🏗️ 構造線 {len(structural_lines)}本を追加")
        
        for line in structural_lines:
            x1, y1, x2, y2 = line[:4]
            tx1, ty1 = self._transform_point(x1, y1)
            tx2, ty2 = self._transform_point(x2, y2)
            
            self.msp.add_line(
                (tx1, -ty1),
                (tx2, -ty2),
                dxfattribs={
                    'layer': 'Structure',
                    'lineweight': 13,  # 0.13mm（細線）
                    'linetype': 'DASHED'
                }
            )
    
    def save(self, filename):
        """高精度DXFファイル保存"""
        try:
            print("💾 高精度DXF生成中...")
            
            # 各要素を高精度で追加
            self._add_high_precision_lines()
            self._add_high_precision_walls()
            self._add_high_precision_circles()
            self._add_high_precision_text()
            self._add_construction_lines()
            
            # ファイル保存
            self.doc.saveas(filename)
            
            # ファイル情報表示
            import os
            file_size = os.path.getsize(filename)
            print(f"✅ 高精度DXFファイル保存完了")
            print(f"📁 ファイル: {filename}")
            print(f"📊 サイズ: {file_size:,} bytes")
            
            return True
            
        except Exception as e:
            print(f"❌ DXF保存エラー: {str(e)}")
            return False
    
    def get_statistics(self):
        """統計情報取得"""
        stats = {
            'lines': len(self.elements.get('lines', [])),
            'walls': len(self.elements.get('walls', [])),
            'circles': len(self.elements.get('circles', [])),
            'text_regions': len(self.elements.get('text_regions', [])),
            'scale_factor': self.scale_factor
        }
        return stats


def create_high_precision_dxf(elements, output_path, scale=100):
    """高精度DXF作成関数"""
    writer = HighPrecisionDXFWriter()
    writer.set_scale(1.0 / scale)
    writer.add_elements(elements)
    
    success = writer.save(output_path)
    
    if success:
        stats = writer.get_statistics()
        print("📊 高精度DXF統計:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    return success
