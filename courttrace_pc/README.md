# コートトレース PC解析（SAM 2 追跡）

Webアプリで付けた「手がかり（クリック）」を出発点に、SAM 2 が選手とボールを全編追跡し、
結果をプロジェクトJSONに書き込みます。JSONをWebアプリで「プロジェクト読込」すると、
背番号付きのコマがボード上で動きます。

## 1. 準備（初回のみ）

```bat
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install ultralytics "git+https://github.com/facebookresearch/sam2.git"
```

チェックポイントを `courttrace_pc\ckpt\` に置きます（tiny を推奨、CPUで動きます）。

- https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
- https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt （精度重視・2倍遅い）

## 2. Webアプリ側でやること

1. 動画を開き、位置合わせ（4〜6点、確定）。
2. カメラが動く動画は「動画全体を自動解析」を一度実行（フレームごとの射影変換がJSONに入る）。
3. 「PC解析（SAM 2）用の手がかり」→「手がかりを付ける」を押し、名簿で選手を選んでから
   動画上のその選手をクリック。10人とボールに最低1回ずつ。途中で見失いそうな場面（交錯・画面外）では
   その後の時刻でもう一度クリック。
4. 「プロジェクト保存」でJSONを書き出す。

## 3. 実行

```bat
venv\Scripts\python courttrace_pc\track_sam2.py --video "試合.mp4" --project "試合.courttrace.json" --out "試合.tracked.json" --fps 6 --model tiny
```

- `--fps 6`: 1秒あたり6フレームを追跡（速度と滑らかさのバランス）。CPUでは 10分の動画で 1〜2時間が目安。
- `--start 120 --end 300`: 区間だけ処理したいとき（秒）。
- `--chunk 20`: メモリ節約の分割秒数。メモリ不足なら 10 に。

終わると `試合.tracked.json` ができ、見失った時刻が一覧表示されます。
Webアプリで読み込み、見失った箇所は手がかりを追加して再実行するか、ボード上でコマをドラッグして直します。

## 4. 共有

Webアプリ（GitHub Pages）でJSONを読み込んだ状態で「プロジェクト保存」しておけば、
同じJSONと動画URL（YouTube）を渡すだけで部内の誰でも同じボードを再生できます。
