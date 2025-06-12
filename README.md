# PDF to DXF Converter

手書き図面のPDFファイルをDXF形式に変換するツールです。最新のAI技術と高度な画像処理を活用して、高精度な変換を実現します。

## 特徴

- **高精度な図形認識**: 線分、円、長方形、壁などの建築要素を正確に検出
- **最適化された処理速度**: 実用的な処理時間で高品質な変換を実現
- **複数のバージョン**: 基本版、拡張版、超高精度版、最適化版を提供
- **可視化機能**: 検出結果をプレビュー表示
- **GUI/CLIサポート**: グラフィカルインターフェースとコマンドライン両方に対応

## インストール

### 必要な環境

- Python 3.8以上
- pip

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 追加の依存関係（オプション）

OCR機能を使用する場合は、Tesseractをインストールしてください：

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# https://github.com/UB-Mannheim/tesseract/wiki からダウンロード
```

## 使用方法

### 1. 基本版（GUI）

```bash
python pdf_to_dxf.py
```

### 2. 拡張版（GUI）

```bash
python enhanced_main.py
```

### 3. 最適化版（推奨）

#### GUI版
```bash
python enhanced_main.py
```

#### コマンドライン版
```bash
python optimized_main.py --input input.pdf --output output.dxf --scale 100 --visualization
```

### パラメータ

- `--input, -i`: 入力PDFファイルのパス
- `--output, -o`: 出力DXFファイルのパス
- `--scale, -s`: スケール（1:scale、デフォルト: 100）
- `--visualization, -v`: 可視化画像を生成（オプション）

## ファイル構成

```
pdf_to_jww_converter/
├── README.md                    # このファイル
├── requirements.txt             # 依存関係
├── .gitignore                  # Git除外設定
├── pdf_to_dxf.py               # 基本版メインプログラム
├── enhanced_main.py            # 拡張版メインプログラム
├── optimized_main.py           # 最適化版メインプログラム（推奨）
├── ultra_main.py               # 超高精度版メインプログラム
├── enhanced_image_processor.py  # 拡張画像処理モジュール
├── enhanced_pdf_processor.py   # 拡張PDFプロセッサー
├── enhanced_dxf_writer.py      # 拡張DXFライター
├── ultra_processor.py          # 超高精度プロセッサー
└── ultra_high_precision_processor.py # 超高精度画像処理
```

## 変換プロセス

1. **PDF読み込み**: PDFファイルを高解像度画像に変換
2. **前処理**: ノイズ除去、コントラスト強化、二値化
3. **テキスト分離**: OCRによるテキスト領域の検出と分離
4. **図形検出**: 線分、円、長方形、壁などの建築要素を検出
5. **クリーンアップ**: 重複除去、ノイズ除去
6. **DXF変換**: 検出された要素をDXF形式に変換
7. **可視化**: 検出結果のプレビュー画像を生成

## 対応形式

### 入力
- PDF形式（手書き図面、建築図面）

### 出力
- DXF形式（AutoCAD、JW-CAD等で利用可能）
- PNG形式（可視化画像）

## 推奨使用方法

最も精度と速度のバランスが取れた **最適化版** の使用を推奨します：

```bash
python optimized_main.py --input your_drawing.pdf --output result.dxf --scale 100 --visualization
```

## 注意事項

- 手書き図面の品質により変換精度が左右されます
- 複雑な図面や文字が多い図面では処理時間が長くなる場合があります
- 変換後は必ず結果を確認し、必要に応じて手動で修正してください

## トラブルシューティング

### よくある問題

1. **Tesseractエラー**
   - Tesseractがインストールされていない場合、OCR機能はスキップされます
   - 図形検出には影響しません

2. **メモリ不足**
   - 大きなPDFファイルの場合、解像度を下げて試してください
   - DPIを400から300に変更することを検討してください

3. **処理時間が長い**
   - 最適化版（optimized_main.py）を使用してください
   - 超高精度版は処理時間が非常に長くなります

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能要望は、GitHubのIssuesでお知らせください。

## 更新履歴

- v1.0.0: 基本版リリース
- v1.1.0: 拡張版追加（高度な画像処理）
- v1.2.0: 超高精度版追加（AI技術活用）
- v1.3.0: 最適化版追加（速度と精度のバランス）
