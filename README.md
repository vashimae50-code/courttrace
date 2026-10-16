# コートトレース（CourtTrace）

バスケの試合動画を、作戦板（ボード）上のコマに変換するブラウザアプリと、PC 側の学習・解析ツールです。

- 公開版（インストール不要）: https://vashimae50-code.github.io/courttrace/
- このリポジトリの中身
  - `index.html` … アプリ本体（1 ファイル）
  - `model/`, `vendor/` … 検出モデルと JS ライブラリ（オフラインでも動くように同梱）
  - `courttrace_pc/` … PC 側ツール（教師データからの追加学習、YOLO/SAM2 追跡）
  - `serve.py`, `start.bat` … 手元でアプリを動かすための簡易サーバー

## 新しい PC（ミニ PC など）へのセットアップ

1. Git と Python 3.10〜3.12、Google Chrome を入れる。
2. リポジトリを取得する。

```bat
cd %USERPROFILE%\Documents
git clone https://github.com/vashimae50-code/courttrace.git
```

3. アプリを使う。どちらでも可。
   - 公開版: Chrome で https://vashimae50-code.github.io/courttrace/ を開く（更新が自動で反映される）。
   - 手元で動かす: `courttrace\start.bat` をダブルクリック → http://localhost:8765 が開く。
4. 学習・PC 解析を使う場合は `courttrace\courttrace_pc\setup.bat` を一度実行する（Python 仮想環境を作ります。数分かかります）。

### 前の PC から引き継ぐもの

| 何を | どうやって |
|---|---|
| 位置合わせ・名簿・記録 | 前の PC のアプリで「プロジェクト保存」→ 新しい PC で「プロジェクト読込」 |
| 教師データ（ラベル） | 前の PC で「教師データを書き出し(ZIP)」→ 新しい PC で「読み込み(ZIP)」 |
| 試合動画 | メモリカードの原本からコピー。ファイル名は変えない（ファイル名で位置合わせと結びつけています） |
| 学習済みモデル `model/custom.onnx` | 学習し直す方が確実。10 枚未満で作ったものは使わない |

### 24 時間稼働の設定（Windows）

- 電源オプションでスリープしない設定にする。
- Windows Update のアクティブ時間を設定し、解析中に再起動されないようにする。
- 外から操作するなら Chrome リモートデスクトップを入れる。

## 更新のしかた

```bat
cd %USERPROFILE%\Documents\courttrace
git pull
```

公開版を使っている場合は、ブラウザで Ctrl+F5（強制再読み込み）するだけです。
