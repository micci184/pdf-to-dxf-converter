#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorFlow強化版 PDF to DXF コンバーター
TensorFlowを活用した高度な画像処理と図形認識
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pdf2image
from datetime import datetime
import tensorflow as tf
from sklearn.cluster import DBSCAN
from enhanced_dxf_writer import EnhancedDXFWriter


class TensorFlowEnhancedConverter:
    """TensorFlow強化変換システム"""
    
    def __init__(self, pdf_path):
        """初期化"""
        self.pdf_path = pdf_path
        self.images = []
        self.tf_session = None
        self.initialize_tensorflow()
        self.load_pdf()
    
    def initialize_tensorflow(self):
        """TensorFlow初期化"""
        try:
            print("🧠 TensorFlow初期化中...")
            # TensorFlowの設定
            tf.config.set_visible_devices([], 'GPU')  # CPU使用を強制
            print(f"✅ TensorFlow {tf.__version__} 初期化完了")
        except Exception as e:
            print(f"⚠️ TensorFlow初期化エラー: {e}")
    
    def load_pdf(self):
        """超高品質PDFロード"""
        try:
            print("📖 超高品質PDF読み込み中...")
            # 500dpiで超高解像度変換
            images = pdf2image.convert_from_path(self.pdf_path, dpi=500)
            
            for img in images:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.images.append(cv_img)
            
            print(f"✅ 超高品質PDF読み込み完了: {len(self.images)}ページ (500dpi)")
        except Exception as e:
            raise Exception(f"PDF読み込み失敗: {str(e)}")
    
    def tensorflow_enhanced_preprocessing(self, image):
        """TensorFlow強化前処理"""
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # TensorFlowテンソルに変換
        tf_image = tf.constant(gray, dtype=tf.float32)
        tf_image = tf.expand_dims(tf_image, axis=0)  # バッチ次元追加
        tf_image = tf.expand_dims(tf_image, axis=-1)  # チャンネル次元追加
        
        # 1. TensorFlowガウシアンフィルタ
        kernel_size = 5
        sigma = 1.0
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        tf_image = tf.nn.conv2d(tf_image, kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # 2. TensorFlowエッジ強調
        edge_kernel = tf.constant([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ], dtype=tf.float32)
        edge_kernel = tf.reshape(edge_kernel, [3, 3, 1, 1])
        
        edges = tf.nn.conv2d(tf_image, edge_kernel, strides=[1, 1, 1, 1], padding='SAME')
        enhanced = tf_image + 0.3 * edges
        
        # NumPy配列に戻す
        enhanced_np = enhanced.numpy()[0, :, :, 0].astype(np.uint8)
        
        # 3. 適応的二値化
        binary = cv2.adaptiveThreshold(
            enhanced_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 3
        )
        
        return enhanced_np, binary
    
    def _create_gaussian_kernel(self, size, sigma):
        """ガウシアンカーネル作成"""
        coords = tf.cast(tf.range(size), tf.float32)
        coords -= size // 2
        
        g = tf.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / tf.reduce_sum(g)
        
        # tf.outerの代わりにmatmulを使用
        g_col = tf.expand_dims(g, axis=1)
        g_row = tf.expand_dims(g, axis=0)
        kernel = tf.matmul(g_col, g_row)
        
        kernel = tf.expand_dims(kernel, axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)
        
        return kernel
    
    def tensorflow_line_detection(self, binary_image):
        """TensorFlow強化線分検出"""
        lines = []
        
        # TensorFlowテンソルに変換
        tf_binary = tf.constant(binary_image, dtype=tf.float32)
        tf_binary = tf.expand_dims(tf_binary, axis=0)
        tf_binary = tf.expand_dims(tf_binary, axis=-1)
        
        # 1. 水平線検出カーネル
        h_kernel = tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        ], dtype=tf.float32)
        h_kernel = tf.reshape(h_kernel, [1, 10, 1, 1])
        
        h_response = tf.nn.conv2d(tf_binary, h_kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # 2. 垂直線検出カーネル
        v_kernel = tf.constant([
            [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
        ], dtype=tf.float32)
        v_kernel = tf.reshape(v_kernel, [10, 1, 1, 1])
        
        v_response = tf.nn.conv2d(tf_binary, v_kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # NumPy配列に変換して処理
        h_enhanced = h_response.numpy()[0, :, :, 0]
        v_enhanced = v_response.numpy()[0, :, :, 0]
        
        # OpenCVのHoughLinesP で精密検出
        h_lines = cv2.HoughLinesP(
            (h_enhanced > 8.0).astype(np.uint8) * 255,
            rho=1, theta=np.pi/180, threshold=50,
            minLineLength=40, maxLineGap=10
        )
        
        v_lines = cv2.HoughLinesP(
            (v_enhanced > 8.0).astype(np.uint8) * 255,
            rho=1, theta=np.pi/180, threshold=50,
            minLineLength=40, maxLineGap=10
        )
        
        # 結果をまとめる
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                lines.append((x1, y1, x2, y2, 'horizontal'))
        
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                lines.append((x1, y1, x2, y2, 'vertical'))
        
        return lines
    
    def tensorflow_shape_recognition(self, binary_image):
        """TensorFlow図形認識"""
        shapes = []
        
        # TensorFlowテンソルに変換
        tf_binary = tf.constant(binary_image, dtype=tf.float32)
        tf_binary = tf.expand_dims(tf_binary, axis=0)
        tf_binary = tf.expand_dims(tf_binary, axis=-1)
        
        # 円検出用カーネル（簡易版）
        circle_kernel = self._create_circle_kernel(15)
        circle_response = tf.nn.conv2d(tf_binary, circle_kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # 長方形検出用カーネル
        rect_kernel = self._create_rectangle_kernel(20, 10)
        rect_response = tf.nn.conv2d(tf_binary, rect_kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # 円の検出
        circle_peaks = tf.where(circle_response > tf.reduce_max(circle_response, axis=None) * 0.8)
        if len(circle_peaks) > 0:
            for peak in circle_peaks.numpy():
                y, x = peak[1], peak[2]  # バッチ次元を除く
                shapes.append({'type': 'circle', 'center': (x, y), 'radius': 15})
        
        # 長方形の検出
        rect_peaks = tf.where(rect_response > tf.reduce_max(rect_response, axis=None) * 0.8)
        if len(rect_peaks) > 0:
            for peak in rect_peaks.numpy():
                y, x = peak[1], peak[2]
                shapes.append({'type': 'rectangle', 'center': (x, y), 'size': (20, 10)})
        
        return shapes
    
    def _create_circle_kernel(self, radius):
        """円検出カーネル作成"""
        size = radius * 2 + 1
        center = radius
        
        y, x = tf.meshgrid(tf.range(size, dtype=tf.float32), tf.range(size, dtype=tf.float32))
        dist = tf.sqrt((x - center)**2 + (y - center)**2)
        
        # 円周部分を強調
        kernel = tf.where(
            tf.abs(dist - radius) < 2.0,
            tf.ones_like(dist),
            tf.zeros_like(dist)
        )
        
        kernel = tf.reshape(kernel, [size, size, 1, 1])
        
        return kernel
    
    def _create_rectangle_kernel(self, width, height):
        """長方形検出カーネル作成"""
        kernel = tf.zeros((height + 4, width + 4), dtype=tf.float32)
        
        # 外枠を1に設定
        kernel = tf.tensor_scatter_nd_update(
            kernel,
            [[0, i] for i in range(width + 4)] +
            [[height + 3, i] for i in range(width + 4)] +
            [[i, 0] for i in range(height + 4)] +
            [[i, width + 3] for i in range(height + 4)],
            tf.ones(2 * (width + 4) + 2 * (height + 4) - 4)
        )
        
        kernel = tf.reshape(kernel, [height + 4, width + 4, 1, 1])
        
        return kernel
    
    def detect_walls_tensorflow(self, lines):
        """TensorFlow壁検出"""
        walls = []
        
        if len(lines) < 2:
            return walls
        
        # 線分データをTensorFlowテンソルに変換
        line_coords = tf.constant([[line[0], line[1], line[2], line[3]] for line in lines], dtype=tf.float32)
        
        # 角度計算
        angles = tf.atan2(line_coords[:, 3] - line_coords[:, 1], line_coords[:, 2] - line_coords[:, 0])
        angles = angles * 180.0 / tf.constant(np.pi)
        angles = tf.abs(angles)
        
        # 中心点計算
        centers = tf.stack([
            (line_coords[:, 0] + line_coords[:, 2]) / 2.0,
            (line_coords[:, 1] + line_coords[:, 3]) / 2.0
        ], axis=1)
        
        # 平行線ペアの検出
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                angle_diff = tf.abs(angles[i] - angles[j])
                angle_diff = tf.minimum(angle_diff, 180.0 - angle_diff)
                
                if angle_diff < 10.0:  # 平行線判定
                    dist = tf.norm(centers[i] - centers[j])
                    
                    if 20.0 < dist < 100.0:  # 適切な距離
                        walls.append({
                            'line1': lines[i][:4],
                            'line2': lines[j][:4],
                            'thickness': float(dist.numpy()),
                            'angle': float(angles[i].numpy())
                        })
        
        return walls
    
    def process_pdf_tensorflow_enhanced(self):
        """TensorFlow強化処理でPDFを処理"""
        try:
            print("🚀 TensorFlow強化処理開始")
            
            all_elements = {
                'lines': [],
                'walls': [],
                'circles': [],
                'rectangles': []
            }
            
            for i, image in enumerate(self.images):
                print(f"🔍 ページ {i+1} をTensorFlow強化処理中...")
                
                # 1. TensorFlow強化前処理
                _, binary = self.tensorflow_enhanced_preprocessing(image)
                
                # 2. TensorFlow線分検出
                lines = self.tensorflow_line_detection(binary)
                all_elements['lines'].extend([line[:4] for line in lines])
                
                # 3. TensorFlow図形認識
                shapes = self.tensorflow_shape_recognition(binary)
                
                for shape in shapes:
                    if shape['type'] == 'circle':
                        all_elements['circles'].append(shape)
                    elif shape['type'] == 'rectangle':
                        all_elements['rectangles'].append(shape)
                
                # 4. TensorFlow壁検出
                walls = self.detect_walls_tensorflow(lines)
                all_elements['walls'].extend(walls)
                
                print(f"✅ ページ {i+1} 完了:")
                print(f"   📏 線分: {len(lines)}本")
                print(f"   🏠 壁: {len(walls)}個")
                print(f"   ⭕ 円: {len([s for s in shapes if s['type'] == 'circle'])}個")
                print(f"   📐 長方形: {len([s for s in shapes if s['type'] == 'rectangle'])}個")
            
            return all_elements
            
        except Exception as e:
            print(f"❌ TensorFlow処理エラー: {str(e)}")
            return None
    
    def create_tensorflow_visualization(self, elements, output_path):
        """TensorFlow強化可視化"""
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
        
        # 長方形描画（紫）
        for rect in elements['rectangles']:
            center = rect['center']
            size = rect['size']
            x1 = center[0] - size[0] // 2
            y1 = center[1] - size[1] // 2
            x2 = center[0] + size[0] // 2
            y2 = center[1] + size[1] // 2
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        cv2.imwrite(output_path, vis_image)
        print(f"🎨 TensorFlow強化可視化保存: {output_path}")


def create_output_filename(base_name):
    """出力ファイル名を生成（日付時間プレフィックス付き）"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    return f"output/{timestamp}_{base_name}"


def convert_pdf_tensorflow_enhanced(input_path, output_path, scale=100, visualization=True):
    """TensorFlow強化PDF変換"""
    try:
        print("=" * 60)
        print("🧠 TensorFlow強化版 AI変換ツール 🧠")
        print("=" * 60)
        
        # 1. 変換システム初期化
        converter = TensorFlowEnhancedConverter(input_path)
        
        # 2. TensorFlow強化処理
        elements = converter.process_pdf_tensorflow_enhanced()
        
        if elements is None:
            return False
        
        # 3. 可視化生成
        if visualization:
            vis_path = output_path.replace('.dxf', '_tensorflow_vis.png')
            converter.create_tensorflow_visualization(elements, vis_path)
        
        # 4. DXF生成
        print("📐 TensorFlow強化DXF生成中...")
        dxf_writer = EnhancedDXFWriter()
        
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        dxf_elements = {
            'lines': elements['lines'],
            'walls': elements['walls'],
            'circles': elements['circles']
        }
        
        dxf_writer.add_elements(dxf_elements)
        dxf_writer.save(output_path)
        
        # 5. 結果表示
        print("=" * 60)
        print("🎉 TensorFlow強化変換完了 🎉")
        print("=" * 60)
        print(f"📁 入力: {input_path}")
        print(f"📁 出力: {output_path}")
        print(f"📊 検出結果:")
        print(f"   📏 線分: {len(elements['lines'])}本")
        print(f"   🏠 壁: {len(elements['walls'])}個")
        print(f"   ⭕ 円: {len(elements['circles'])}個")
        print(f"   📐 長方形: {len(elements['rectangles'])}個")
        print("🧠 TensorFlowによる超高精度変換完了！")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🧠 TensorFlow強化版 AI変換ツール 🧠')
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
        output_path = create_output_filename(f"{base_name}_tensorflow.dxf")
    
    # 変換実行
    success = convert_pdf_tensorflow_enhanced(
        args.input,
        output_path,
        args.scale,
        not args.no_visualization
    )
    
    if success:
        print("✅ TensorFlow強化変換が正常に完了しました！")
        sys.exit(0)
    else:
        print("❌ 変換に失敗しました。")
        sys.exit(1)
