#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
拡張DXFライタークラス
検出された図形要素をDXF形式に変換する
"""

import os
import numpy as np
import ezdxf
from ezdxf.math import Vec2


class EnhancedDXFWriter:
    """検出された図形要素をDXF形式に変換するクラス"""
    
    def __init__(self):
        """初期化"""
        self.elements = {}
        self.scale_factor = 1.0  # スケールファクター
        self.origin_x = 0  # 原点X座標
        self.origin_y = 0  # 原点Y座標
        self.auto_cleanup = True  # 自動クリーンアップ
    
    def set_scale(self, scale_factor):
        """
        スケールを設定
        
        Args:
            scale_factor (float): スケールファクター
        """
        self.scale_factor = scale_factor
    
    def set_origin(self, x, y):
        """
        原点を設定
        
        Args:
            x (float): X座標
            y (float): Y座標
        """
        self.origin_x = x
        self.origin_y = y
    
    def add_elements(self, detected_elements):
        """
        検出された図形要素を追加
        
        Args:
            detected_elements (dict): 検出された図形要素の辞書
        """
        self.elements = detected_elements
    
    def _transform_point(self, x, y):
        """
        座標変換（スケールと原点の適用）
        
        Args:
            x (float): X座標
            y (float): Y座標
            
        Returns:
            tuple: 変換後の座標 (x, y)
        """
        return (
            (x - self.origin_x) * self.scale_factor,
            (y - self.origin_y) * self.scale_factor
        )
    
    def _cleanup_elements(self):
        """
        要素のクリーンアップ（重複除去、整列など）
        
        Returns:
            dict: クリーンアップされた要素
        """
        if not self.auto_cleanup:
            return self.elements
        
        cleaned = {
            'lines': [],
            'circles': [],
            'rectangles': [],
            'walls': [],
            'doors': [],
            'windows': []
        }
        
        # 線分の重複除去
        if 'lines' in self.elements:
            seen_lines = set()
            for line in self.elements['lines']:
                x1, y1, x2, y2 = line
                # 線分の正規化（始点と終点を常に一定の順序に）
                if (x1, y1) > (x2, y2):
                    x1, y1, x2, y2 = x2, y2, x1, y1
                
                # 近似的な重複チェック（小数点以下を丸める）
                key = (round(x1), round(y1), round(x2), round(y2))
                if key not in seen_lines:
                    seen_lines.add(key)
                    cleaned['lines'].append(line)
        
        # 円の重複除去
        if 'circles' in self.elements:
            seen_circles = set()
            for circle in self.elements['circles']:
                x, y, r = circle
                key = (round(x), round(y), round(r))
                if key not in seen_circles:
                    seen_circles.add(key)
                    cleaned['circles'].append(circle)
        
        # 長方形の重複除去
        if 'rectangles' in self.elements:
            seen_rects = set()
            for rect in self.elements['rectangles']:
                x, y, w, h, angle = rect
                key = (round(x), round(y), round(w), round(h), round(angle))
                if key not in seen_rects:
                    seen_rects.add(key)
                    cleaned['rectangles'].append(rect)
        
        # その他の要素をコピー
        for key in ['walls', 'doors', 'windows']:
            if key in self.elements:
                cleaned[key] = self.elements[key]
        
        return cleaned
    
    def save(self, output_path):
        """
        DXF形式でファイルを保存
        
        Args:
            output_path (str): 出力ファイルのパス
            
        Returns:
            bool: 保存が成功したかどうか
        """
        try:
            # 新しいDXFドキュメントを作成（AutoCAD 2010形式）
            doc = ezdxf.new('R2010')
            
            # レイヤーの作成
            doc.layers.new(name='Lines', dxfattribs={'color': 7})  # 白
            doc.layers.new(name='Circles', dxfattribs={'color': 1})  # 赤
            doc.layers.new(name='Rectangles', dxfattribs={'color': 5})  # 青
            doc.layers.new(name='Walls', dxfattribs={'color': 2})  # 黄
            doc.layers.new(name='Doors', dxfattribs={'color': 3})  # 緑
            doc.layers.new(name='Windows', dxfattribs={'color': 4})  # シアン
            
            # モデルスペースの取得
            msp = doc.modelspace()
            
            # 要素のクリーンアップ
            elements = self._cleanup_elements()
            
            # 線分の追加
            if 'lines' in elements:
                for line in elements['lines']:
                    x1, y1, x2, y2 = line
                    tx1, ty1 = self._transform_point(x1, y1)
                    tx2, ty2 = self._transform_point(x2, y2)
                    msp.add_line(
                        (tx1, -ty1),  # Y座標を反転（DXFの座標系に合わせる）
                        (tx2, -ty2),
                        dxfattribs={'layer': 'Lines'}
                    )
            
            # 円の追加
            if 'circles' in elements:
                for circle in elements['circles']:
                    x, y, r = circle
                    tx, ty = self._transform_point(x, y)
                    tr = r * self.scale_factor  # 半径もスケール
                    msp.add_circle(
                        (tx, -ty),  # Y座標を反転
                        tr,
                        dxfattribs={'layer': 'Circles'}
                    )
            
            # 長方形の追加
            if 'rectangles' in elements:
                for rect in elements['rectangles']:
                    x, y, w, h, angle = rect
                    tx, ty = self._transform_point(x, y)
                    tw = w * self.scale_factor
                    th = h * self.scale_factor
                    
                    # 回転した長方形の頂点を計算
                    c = np.cos(np.radians(angle))
                    s = np.sin(np.radians(angle))
                    
                    half_w = tw / 2
                    half_h = th / 2
                    
                    points = [
                        (tx + c * half_w - s * half_h, -(ty + s * half_w + c * half_h)),
                        (tx - c * half_w - s * half_h, -(ty - s * half_w + c * half_h)),
                        (tx - c * half_w + s * half_h, -(ty - s * half_w - c * half_h)),
                        (tx + c * half_w + s * half_h, -(ty + s * half_w - c * half_h)),
                    ]
                    
                    msp.add_lwpolyline(
                        points,
                        dxfattribs={'layer': 'Rectangles', 'closed': True}
                    )
            
            # 壁の追加
            if 'walls' in elements:
                for wall in elements['walls']:
                    if isinstance(wall, dict) and 'line1' in wall and 'line2' in wall:
                        # 辞書形式の壁データ
                        x1, y1, x2, y2 = wall['line1']
                        tx1, ty1 = self._transform_point(x1, y1)
                        tx2, ty2 = self._transform_point(x2, y2)
                        
                        msp.add_line(
                            (tx1, -ty1),
                            (tx2, -ty2),
                            dxfattribs={'layer': 'Walls'}
                        )
                        
                        x1, y1, x2, y2 = wall['line2']
                        tx1, ty1 = self._transform_point(x1, y1)
                        tx2, ty2 = self._transform_point(x2, y2)
                        
                        msp.add_line(
                            (tx1, -ty1),
                            (tx2, -ty2),
                            dxfattribs={'layer': 'Walls'}
                        )
                    else:
                        # タプル形式の壁データ（従来の形式）
                        if len(wall) >= 5:
                            x1, y1, x2, y2, thickness = wall
                        else:
                            x1, y1, x2, y2 = wall[:4]
                            thickness = 0
                        
                        tx1, ty1 = self._transform_point(x1, y1)
                        tx2, ty2 = self._transform_point(x2, y2)
                        
                        # 壁の厚みを考慮したポリライン
                        if thickness > 0:
                            # 壁の方向ベクトル
                            dx = tx2 - tx1
                            dy = ty2 - ty1
                            length = np.sqrt(dx*dx + dy*dy)
                            
                            if length > 0:
                                # 単位法線ベクトル
                                nx = -dy / length
                                ny = dx / length
                                
                                # 壁の厚みの半分
                                half_thick = thickness * self.scale_factor / 2
                                
                                # 壁の4つの頂点
                                points = [
                                    (tx1 + nx * half_thick, -(ty1 + ny * half_thick)),
                                    (tx2 + nx * half_thick, -(ty2 + ny * half_thick)),
                                    (tx2 - nx * half_thick, -(ty2 - ny * half_thick)),
                                    (tx1 - nx * half_thick, -(ty1 - ny * half_thick)),
                                ]
                                
                                msp.add_lwpolyline(
                                    points,
                                    dxfattribs={'layer': 'Walls', 'closed': True}
                                )
                        else:
                            # 厚みが指定されていない場合は線分として描画
                            msp.add_line(
                                (tx1, -ty1),
                                (tx2, -ty2),
                                dxfattribs={'layer': 'Walls'}
                            )
            
            # ドアの追加
            if 'doors' in elements:
                for door in elements['doors']:
                    # ドアの描画処理（実装省略）
                    pass
            
            # 窓の追加
            if 'windows' in elements:
                for window in elements['windows']:
                    # 窓の描画処理（実装省略）
                    pass
            
            # DXFファイルとして保存
            doc.saveas(output_path)
            print(f"DXFファイルを保存しました: {output_path}")
            return True
        
        except Exception as e:
            print(f"DXFファイルの保存に失敗しました: {str(e)}")
            raise e
