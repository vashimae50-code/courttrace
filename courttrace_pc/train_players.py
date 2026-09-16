#!/usr/bin/env python
"""
CourtTrace 追加学習: アプリの「教師データを書き出し(ZIP)」で作った ZIP から、
人物(player) / ボール(ball) / 審判(referee) 検出モデルを追加学習し、ブラウザ用 ONNX を書き出します。

使い方（venv を有効にした状態で）:
  python train_players.py courttrace_labels_XXXX.zip [他の試合のZIP ...] [--epochs 60] [--imgsz 960] [--model yolov10s.pt]

出力:
  runs/courttrace/<日時>/weights/best.pt      … 学習済み PyTorch モデル（次回の追加学習の土台にも使える）
  runs/courttrace/<日時>/weights/best.onnx    … ブラウザ用（アプリの「カスタム検出モデル」で読み込む）
  model/custom.onnx                            … 同じものをコピー。courttrace リポジトリの model/ に置くと全員に配布される

ヒント:
  - 複数試合の ZIP をまとめて渡すほど汎用性が上がります（クラスは player/ball/referee で試合をまたいで共通）。
  - CPU でも動きます（100枚・60エポックで 30〜60分程度）。GPU があれば --device 0。
  - 学習済み best.pt を --model に渡せば、そこからさらに追加学習できます。
"""
import argparse, os, random, shutil, sys, time, zipfile
from pathlib import Path


class _Tee:
    """Mirror everything printed to the console into train_log.txt (so the result survives a closed window)."""
    def __init__(self, stream, path):
        self.stream = stream; self.f = open(path, 'a', encoding='utf-8', errors='replace')
    def write(self, s):
        try: self.stream.write(s)
        except Exception: pass
        self.f.write(s); self.f.flush()
    def flush(self):
        try: self.stream.flush()
        except Exception: pass
        self.f.flush()
    def isatty(self): return False
    def fileno(self): return self.stream.fileno()

_ENV0 = dict(os.environ)  # environment before ultralytics touches it (used for the export subprocess)
_LOG = Path(__file__).resolve().parent / 'train_log.txt'
sys.stdout = _Tee(sys.stdout, _LOG); sys.stderr = _Tee(sys.stderr, _LOG)
print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} train_players.py {' '.join(sys.argv[1:])}")
print(f"log: {_LOG}")


def unpack(zips, root: Path):
    img_dir = root / 'images'; lab_dir = root / 'labels'
    for d in (img_dir / 'train', img_dir / 'val', lab_dir / 'train', lab_dir / 'val'):
        d.mkdir(parents=True, exist_ok=True)
    pairs = []
    for zi, zp in enumerate(zips):
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            for n in names:
                if n.startswith('images/train/') and n.lower().endswith('.jpg'):
                    stem = Path(n).stem
                    lab = f'labels/train/{stem}.txt'
                    if lab not in names:
                        continue
                    out_stem = f'g{zi}_{stem}'
                    pairs.append((out_stem, z.read(n), z.read(lab)))
    if not pairs:
        sys.exit('ZIP の中に images/train/*.jpg と labels/train/*.txt が見つかりません')
    random.seed(0); random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.15)) if len(pairs) >= 5 else 0
    for i, (stem, jpg, lab) in enumerate(pairs):
        split = 'val' if i < n_val else 'train'
        (img_dir / split / f'{stem}.jpg').write_bytes(jpg)
        (lab_dir / split / f'{stem}.txt').write_bytes(lab)
    if n_val == 0:  # too few images: validate on the training set
        for f in (img_dir / 'train').iterdir():
            shutil.copy(f, img_dir / 'val' / f.name)
        for f in (lab_dir / 'train').iterdir():
            shutil.copy(f, lab_dir / 'val' / f.name)
    yaml = root / 'data.yaml'
    yaml.write_text(f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: player\n  1: ball\n  2: referee\n", encoding='utf-8')
    return yaml, len(pairs), n_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zips', nargs='+', help='アプリで書き出した教師データ ZIP（複数可）')
    ap.add_argument('--model', default='yolov10s.pt', help='土台のモデル（yolov10s.pt / yolov10n.pt / 前回の best.pt）')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=960, help='学習時の画像サイズ。ボールが小さいので 960〜1280 推奨（CPU なら 960）')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default='cpu', help="'cpu' または GPU 番号 '0'")
    ap.add_argument('--out', default='runs/courttrace')
    a = ap.parse_args()

    from ultralytics import YOLO
    work = Path('dataset_tmp'); shutil.rmtree(work, ignore_errors=True)
    yaml, n, n_val = unpack(a.zips, work)
    print(f'画像 {n} 枚（検証用 {n_val} 枚）を展開しました → {work}')

    name = time.strftime('%Y%m%d_%H%M')
    model = YOLO(a.model)
    results = model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, project=a.out, name=name,
                # small dataset: gentle augmentation, freeze nothing, keep pretrained "person"/"sports ball" knowledge
                lr0=0.002, warmup_epochs=2, close_mosaic=10, degrees=0, shear=0, perspective=0, flipud=0, fliplr=0.5,
                hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, mosaic=0.8, mixup=0.0, patience=30, workers=2, verbose=True)
    save_dir = Path(getattr(results, 'save_dir', None) or model.trainer.save_dir)
    best = save_dir / 'weights' / 'best.pt'
    if not best.exists():
        best = save_dir / 'weights' / 'last.pt'
    print('学習完了:', best)

    # export for the browser: 640 square, same output layout as the bundled yolov10 models ([1,300,6] x1,y1,x2,y2,score,class)
    # NOTE: exporting inside the training process yields the raw [1,7,8400] head; a fresh interpreter gives the
    # end-to-end [1,300,6] layout the app expects, so run the export in a subprocess.
    import subprocess
    code = ("from ultralytics import YOLO; import sys; "
            "print(YOLO(sys.argv[1]).export(format='onnx', imgsz=640, opset=17, simplify=True, dynamic=False))")
    proc = subprocess.run([sys.executable, '-c', code, str(best)], capture_output=True, text=True, encoding='utf-8', errors='replace', env=_ENV0)
    sys.stdout.write(proc.stdout); sys.stderr.write(proc.stderr)
    onnx = best.with_suffix('.onnx')
    if proc.returncode != 0 or not onnx.exists():
        sys.exit('ONNX への書き出しに失敗しました（上のメッセージを確認してください）')
    try:
        import numpy as np, onnxruntime as ort
        sess = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
        shape = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, 640, 640), np.float32)})[0].shape
        print('ONNX 出力の形:', shape, '（[1, 300, 6] が期待値。[1, 7, 8400] でもアプリは対応済み）')
    except Exception as e:
        print('ONNX の確認をスキップ:', e)
    dst = Path('model'); dst.mkdir(exist_ok=True)
    shutil.copy(onnx, dst / 'custom.onnx')
    print('=' * 60); print('DONE  ブラウザ用モデル:', (dst / 'custom.onnx').resolve()); print('=' * 60)
    print('\n次にやること:')
    print('  1) アプリの「AI自動解析 → カスタム検出モデル」で model/custom.onnx を選ぶ（自分だけで使う）')
    print('  2) みんなで使うなら courttrace リポジトリの model/custom.onnx としてアップロード（自動で使われます）')


if __name__ == '__main__':
    main()
