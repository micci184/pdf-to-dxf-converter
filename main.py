#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF to JWW コンバーター
手書き図面のPDFをJWW形式に変換するツール
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
from pdf_processor import PDFProcessor
from jww_writer import JWWWriter

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
            
            # JWWファイルの作成
            jww_writer = JWWWriter()
            jww_writer.add_elements(elements)
            jww_writer.save(self.output_path)
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
        self.setWindowTitle("PDF to JWW コンバーター")
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
        self.convert_button = QPushButton("JWWに変換")
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
            "このツールは手書き図面のPDFをJWW形式に変換します。\n"
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
            self, "保存先を選択", "", "JWW Files (*.jww)"
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
