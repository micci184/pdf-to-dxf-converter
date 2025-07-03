#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
拡張版 PDF to DXF コンバーター
高度な画像処理とAI技術を活用して手書き図面のPDFをDXF形式に高精度で変換するツール
"""

import os
import sys
import tempfile
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QWidget, QFileDialog, QLabel, QProgressBar, QMessageBox,
                            QSlider, QCheckBox, QComboBox, QGroupBox, QGridLayout, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

from enhanced_pdf_processor import EnhancedPDFProcessor
from enhanced_dxf_writer import EnhancedDXFWriter


class ConversionThread(QThread):
    """変換処理を行うスレッド"""
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    preview_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, input_path, output_path, settings):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.settings = settings
        
    def run(self):
        try:
            # PDFの処理
            self.status_signal.emit("PDFを読み込んでいます...")
            self.progress_signal.emit(10)
            
            pdf_processor = EnhancedPDFProcessor(self.input_path)
            
            # プレビュー画像の生成と送信
            self.status_signal.emit("プレビューを生成しています...")
            preview_image = pdf_processor.visualize_detection()
            self.preview_signal.emit(preview_image)
            self.progress_signal.emit(30)
            
            # 図形要素の検出
            self.status_signal.emit("図形要素を検出しています...")
            elements = pdf_processor.detect_elements()
            self.progress_signal.emit(60)
            
            # DXFファイルの作成
            self.status_signal.emit("DXFファイルを生成しています...")
            dxf_writer = EnhancedDXFWriter()
            
            # スケールの設定
            if self.settings['scale_factor'] != 1.0:
                dxf_writer.set_scale(self.settings['scale_factor'])
            
            # 原点の設定
            if self.settings['set_origin']:
                dxf_writer.set_origin(
                    self.settings['origin_x'],
                    self.settings['origin_y']
                )
            
            # 自動クリーンアップの設定
            dxf_writer.auto_cleanup = self.settings['auto_cleanup']
            
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
        self.setWindowTitle("拡張版 PDF to DXF コンバーター")
        self.setGeometry(100, 100, 800, 600)
        
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
        
        # 設定部分
        settings_group = QGroupBox("変換設定")
        settings_layout = QGridLayout()
        
        # スケール設定
        settings_layout.addWidget(QLabel("スケール:"), 0, 0)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["1:1", "1:50", "1:100", "1:200", "1:500", "カスタム"])
        self.scale_combo.setCurrentText("1:100")
        self.scale_combo.currentTextChanged.connect(self.update_scale)
        settings_layout.addWidget(self.scale_combo, 0, 1)
        
        self.custom_scale = QSpinBox()
        self.custom_scale.setRange(1, 1000)
        self.custom_scale.setValue(100)
        self.custom_scale.setEnabled(False)
        self.custom_scale.valueChanged.connect(self.update_custom_scale)
        settings_layout.addWidget(self.custom_scale, 0, 2)
        
        # 原点設定
        self.origin_check = QCheckBox("原点を設定")
        self.origin_check.setChecked(False)
        self.origin_check.stateChanged.connect(self.toggle_origin)
        settings_layout.addWidget(self.origin_check, 1, 0)
        
        settings_layout.addWidget(QLabel("X:"), 1, 1)
        self.origin_x = QSpinBox()
        self.origin_x.setRange(0, 10000)
        self.origin_x.setValue(0)
        self.origin_x.setEnabled(False)
        settings_layout.addWidget(self.origin_x, 1, 2)
        
        settings_layout.addWidget(QLabel("Y:"), 1, 3)
        self.origin_y = QSpinBox()
        self.origin_y.setRange(0, 10000)
        self.origin_y.setValue(0)
        self.origin_y.setEnabled(False)
        settings_layout.addWidget(self.origin_y, 1, 4)
        
        # 自動クリーンアップ
        self.cleanup_check = QCheckBox("自動クリーンアップ（重複除去）")
        self.cleanup_check.setChecked(True)
        settings_layout.addWidget(self.cleanup_check, 2, 0, 1, 3)
        
        settings_group.setLayout(settings_layout)
        
        # プレビュー部分
        preview_group = QGroupBox("プレビュー")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel("PDFを選択するとプレビューが表示されます")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet("background-color: #f0f0f0;")
        
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        
        # 変換ボタン
        self.convert_button = QPushButton("DXFに変換")
        self.convert_button.clicked.connect(self.convert_file)
        self.convert_button.setEnabled(False)
        
        # ステータスラベル
        self.status_label = QLabel("準備完了")
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # レイアウトに追加
        main_layout.addLayout(file_layout)
        main_layout.addWidget(settings_group)
        main_layout.addWidget(preview_group)
        main_layout.addWidget(self.convert_button)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.progress_bar)
        
        # メインウィジェットの設定
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.input_path = None
        self.conversion_thread = None
    
    def update_scale(self, text):
        """スケール設定の更新"""
        self.custom_scale.setEnabled(text == "カスタム")
    
    def update_custom_scale(self):
        """カスタムスケールの更新"""
        self.scale_combo.setCurrentText("カスタム")
    
    def toggle_origin(self, state):
        """原点設定の切り替え"""
        enabled = state == Qt.Checked
        self.origin_x.setEnabled(enabled)
        self.origin_y.setEnabled(enabled)
    
    def select_file(self):
        """PDFファイルを選択"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "PDFファイルを選択", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            self.input_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.convert_button.setEnabled(True)
            self.status_label.setText("PDFを読み込んでプレビューを生成中...")
            
            # プレビュー生成（バックグラウンド）
            self.generate_preview(file_path)
    
    def generate_preview(self, pdf_path):
        """プレビューを生成"""
        try:
            # 簡易的なプレビュー生成（本来はスレッド化すべき）
            pdf_processor = EnhancedPDFProcessor(pdf_path)
            preview_image = pdf_processor.visualize_detection()
            
            # プレビュー表示
            self.update_preview(preview_image)
            self.status_label.setText("プレビュー生成完了")
        except Exception as e:
            self.status_label.setText(f"プレビュー生成エラー: {str(e)}")
    
    def update_preview(self, image):
        """プレビュー画像を更新"""
        h, w = image.shape[:2]
        
        # 表示サイズに合わせてリサイズ
        max_height = 300
        if h > max_height:
            scale = max_height / h
            w = int(w * scale)
            h = max_height
            image = cv2.resize(image, (w, h))
        
        # OpenCV画像をQPixmapに変換
        bytes_per_line = 3 * w
        q_image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_image)
        self.preview_label.setPixmap(pixmap)
    
    def get_settings(self):
        """現在の設定を取得"""
        settings = {}
        
        # スケール設定
        scale_text = self.scale_combo.currentText()
        if scale_text == "カスタム":
            settings['scale_factor'] = 1.0 / self.custom_scale.value()
        else:
            try:
                scale_value = int(scale_text.split(':')[1])
                settings['scale_factor'] = 1.0 / scale_value
            except:
                settings['scale_factor'] = 1.0
        
        # 原点設定
        settings['set_origin'] = self.origin_check.isChecked()
        settings['origin_x'] = self.origin_x.value()
        settings['origin_y'] = self.origin_y.value()
        
        # クリーンアップ設定
        settings['auto_cleanup'] = self.cleanup_check.isChecked()
        
        return settings
    
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
            
            # 現在の設定を取得
            settings = self.get_settings()
            
            # 変換スレッドの開始
            self.conversion_thread = ConversionThread(self.input_path, output_path, settings)
            self.conversion_thread.progress_signal.connect(self.update_progress)
            self.conversion_thread.status_signal.connect(self.update_status)
            self.conversion_thread.preview_signal.connect(self.update_preview)
            self.conversion_thread.finished_signal.connect(self.conversion_finished)
            self.conversion_thread.error_signal.connect(self.conversion_error)
            self.conversion_thread.start()
    
    def update_progress(self, value):
        """進捗バーの更新"""
        self.progress_bar.setValue(value)
    
    def update_status(self, status):
        """ステータスの更新"""
        self.status_label.setText(status)
    
    def conversion_finished(self, output_path):
        """変換完了時の処理"""
        self.select_button.setEnabled(True)
        self.convert_button.setEnabled(True)
        self.status_label.setText("変換完了")
        
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
        self.status_label.setText("エラー発生")
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(error_message)
        msg_box.setWindowTitle("エラー")
        msg_box.exec_()


def convert_pdf_to_dxf(input_path, output_path, scale=100, set_origin=False, origin_x=0, origin_y=0):
    """
    PDFをDXFに変換する（コマンドライン用）
    
    Args:
        input_path (str): 入力PDFファイルのパス
        output_path (str): 出力DXFファイルのパス
        scale (int): スケール（1:scale）
        set_origin (bool): 原点を設定するかどうか
        origin_x (int): 原点のX座標
        origin_y (int): 原点のY座標
    
    Returns:
        bool: 変換が成功したかどうか
    """
    try:
        # PDFの処理
        pdf_processor = EnhancedPDFProcessor(input_path)
        
        # 図形要素の検出
        elements = pdf_processor.detect_elements()
        
        # DXFファイルの作成
        dxf_writer = EnhancedDXFWriter()
        
        # スケールの設定
        if scale != 1:
            dxf_writer.set_scale(1.0 / scale)
        
        # 原点の設定
        if set_origin:
            dxf_writer.set_origin(origin_x, origin_y)
        
        dxf_writer.add_elements(elements)
        dxf_writer.save(output_path)
        
        print(f"変換が完了しました: {output_path}")
        return True
    except Exception as e:
        print(f"変換中にエラーが発生しました: {str(e)}")
        return False


if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='PDFをDXFに変換するツール')
    parser.add_argument('--input', '-i', help='入力PDFファイルのパス')
    parser.add_argument('--output', '-o', help='出力DXFファイルのパス')
    parser.add_argument('--scale', '-s', type=int, default=100, help='スケール（1:scale）')
    parser.add_argument('--origin', action='store_true', help='原点を設定する')
    parser.add_argument('--origin-x', type=int, default=0, help='原点のX座標')
    parser.add_argument('--origin-y', type=int, default=0, help='原点のY座標')
    
    args = parser.parse_args()
    
    if args.input and args.output:
        # コマンドラインモード
        convert_pdf_to_dxf(
            args.input, args.output, args.scale,
            args.origin, args.origin_x, args.origin_y
        )
    else:
        # GUIモード
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
