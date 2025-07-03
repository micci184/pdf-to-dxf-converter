#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF to DXF コンバーター
手書き図面のPDFをDXF形式に変換するツール
"""

import os
import sys
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QWidget, QFileDialog, QLabel, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pdf2image
import cv2
import numpy as np
import ezdxf
from skimage import measure, morphology


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
        
        return elements


class DXFWriter:
    """検出された図形要素をDXF形式に変換するクラス"""
    
    def __init__(self):
        """初期化"""
        self.elements = []
        
    def add_elements(self, detected_elements):
        """
        検出された図形要素を追加
        
        Args:
            detected_elements (dict): 検出された図形要素の辞書
        """
        # 線分の追加
        for line in detected_elements['lines']:
            self.elements.append({
                'type': 'line',
                'coords': line['coords'],
                'layer': 'Lines',
                'color': 7  # 白
            })
        
        # 円の追加
        for circle in detected_elements['circles']:
            self.elements.append({
                'type': 'circle',
                'coords': circle['coords'],
                'layer': 'Circles',
                'color': 1  # 赤
            })
        
        # 長方形の追加
        for rect in detected_elements['rectangles']:
            x, y, w, h = rect['coords']
            self.elements.append({
                'type': 'rectangle',
                'coords': (x, y, x + w, y + h),
                'layer': 'Rectangles',
                'color': 5  # 青
            })
    
    def save(self, output_path):
        """
        DXF形式でファイルを保存
        
        Args:
            output_path (str): 出力ファイルのパス
        """
        try:
            # 新しいDXFドキュメントを作成
            doc = ezdxf.new('R2010')
            
            # レイヤーの作成
            doc.layers.new(name='Lines', dxfattribs={'color': 7})
            doc.layers.new(name='Circles', dxfattribs={'color': 1})
            doc.layers.new(name='Rectangles', dxfattribs={'color': 5})
            
            # モデルスペースの取得
            msp = doc.modelspace()
            
            # 図形要素の追加
            for element in self.elements:
                if element['type'] == 'line':
                    x1, y1, x2, y2 = element['coords']
                    msp.add_line(
                        (x1, -y1), (x2, -y2),  # Y座標を反転（DXFの座標系に合わせる）
                        dxfattribs={'layer': element['layer']}
                    )
                
                elif element['type'] == 'circle':
                    x, y, r = element['coords']
                    msp.add_circle(
                        (x, -y), r,  # Y座標を反転
                        dxfattribs={'layer': element['layer']}
                    )
                
                elif element['type'] == 'rectangle':
                    x1, y1, x2, y2 = element['coords']
                    # 長方形をポリラインとして追加
                    msp.add_lwpolyline(
                        [(x1, -y1), (x2, -y1), (x2, -y2), (x1, -y2), (x1, -y1)],  # Y座標を反転
                        dxfattribs={'layer': element['layer'], 'closed': True}
                    )
            
            # DXFファイルとして保存
            doc.saveas(output_path)
            print(f"DXFファイルを保存しました: {output_path}")
            return True
        except Exception as e:
            raise Exception(f"DXFファイルの保存に失敗しました: {str(e)}")


class ConversionThread(QThread):
    """変換処理を行うスレッド"""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        
    def run(self):
        try:
            # PDFの処理
            self.progress_signal.emit(10)
            pdf_processor = PDFProcessor(self.input_path)
            self.progress_signal.emit(30)
            
            # 図形要素の検出
            elements = pdf_processor.detect_elements()
            self.progress_signal.emit(60)
            
            # DXFファイルの作成
            dxf_writer = DXFWriter()
            dxf_writer.add_elements(elements)
            dxf_writer.save(self.output_path)
            self.progress_signal.emit(100)
            
            self.finished_signal.emit(self.output_path)
        except Exception as e:
            self.error_signal.emit(f"変換中にエラーが発生しました: {str(e)}")


class MainWindow(QMainWindow):
    """メインウィンドウ"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """UIの初期化"""
        self.setWindowTitle("PDF to DXF コンバーター")
        self.setGeometry(100, 100, 600, 400)
        
        # メインウィジェットとレイアウト
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # ファイル選択部分
        file_layout = QHBoxLayout()
        self.file_label = QLabel("ファイルが選択されていません")
        self.select_button = QPushButton("PDFを選択")
        self.select_button.clicked.connect(self.select_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_button)
        
        # 変換ボタン
        self.convert_button = QPushButton("DXFに変換")
        self.convert_button.clicked.connect(self.convert_file)
        self.convert_button.setEnabled(False)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # レイアウトに追加
        main_layout.addLayout(file_layout)
        main_layout.addWidget(self.convert_button)
        main_layout.addWidget(self.progress_bar)
        
        # 説明テキスト
        info_label = QLabel(
            "このツールは手書き図面のPDFをDXF形式に変換します。\n"
            "AIを使用して図形要素を検出し、ベクターデータに変換します。\n"
            "変換精度は原図の品質に依存します。"
        )
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)
        
        # メインウィジェットの設定
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.input_path = None
        self.conversion_thread = None
    
    def select_file(self):
        """PDFファイルを選択"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "PDFファイルを選択", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            self.input_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.convert_button.setEnabled(True)
    
    def convert_file(self):
        """ファイルの変換を開始"""
        if not self.input_path:
            return
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存先を選択", "", "DXF Files (*.dxf)"
        )
        
        if output_path:
            # ボタンを無効化
            self.select_button.setEnabled(False)
            self.convert_button.setEnabled(False)
            
            # 変換スレッドの開始
            self.conversion_thread = ConversionThread(self.input_path, output_path)
            self.conversion_thread.progress_signal.connect(self.update_progress)
            self.conversion_thread.finished_signal.connect(self.conversion_finished)
            self.conversion_thread.error_signal.connect(self.conversion_error)
            self.conversion_thread.start()
    
    def update_progress(self, value):
        """進捗バーの更新"""
        self.progress_bar.setValue(value)
    
    def conversion_finished(self, output_path):
        """変換完了時の処理"""
        self.select_button.setEnabled(True)
        self.convert_button.setEnabled(True)
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"変換が完了しました。\n保存先: {output_path}")
        msg_box.setWindowTitle("変換完了")
        msg_box.exec_()
    
    def conversion_error(self, error_message):
        """エラー発生時の処理"""
        self.select_button.setEnabled(True)
        self.convert_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(error_message)
        msg_box.setWindowTitle("エラー")
        msg_box.exec_()


def convert_pdf_to_dxf(input_path, output_path):
    """
    PDFをDXFに変換する（コマンドライン用）
    
    Args:
        input_path (str): 入力PDFファイルのパス
        output_path (str): 出力DXFファイルのパス
    
    Returns:
        bool: 変換が成功したかどうか
    """
    try:
        # PDFの処理
        pdf_processor = PDFProcessor(input_path)
        
        # 図形要素の検出
        elements = pdf_processor.detect_elements()
        
        # DXFファイルの作成
        dxf_writer = DXFWriter()
        dxf_writer.add_elements(elements)
        dxf_writer.save(output_path)
        
        print(f"変換が完了しました: {output_path}")
        return True
    except Exception as e:
        print(f"変換中にエラーが発生しました: {str(e)}")
        return False


if __name__ == "__main__":
    # コマンドライン引数のチェック
    if len(sys.argv) > 2:
        # コマンドラインモード
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        convert_pdf_to_dxf(input_path, output_path)
    else:
        # GUIモード
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
