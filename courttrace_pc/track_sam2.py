#!/usr/bin/env python
"""
コートトレース PC解析: SAM 2 で選手とボールを追跡し、プロジェクトJSONにキーフレームを書き込む。

使い方:
  python track_sam2.py --video 試合.mp4 --project 試合.courttrace.json --out 試合.tracked.json
  オプション: --fps 6 (追跡する1秒あたりのフレーム数) --start 0 --end 60 (秒) --model tiny|small --device cpu|cuda

入力JSONに必要なもの:
  prompts : Webアプリ「手がかりを付ける」で付けた印 [{entity:'A4'|'ball', t, u, v}]
  calibs  : 位置合わせ（コートの目印 → 画像座標）
  scan.Hs : (あれば) ブラウザ解析で得たフレームごとの射影変換。カメラが動く映像ではこれを使う。
出力JSON: 各選手/ボールの kfs にコート座標（m）の位置が入る。Webアプリの「プロジェクト読込」で開くとボードで再生できる。
"""
import argparse, json, math, os, shutil, sys, tempfile, time
import numpy as np
import cv2

# ---------------- court model (FIBA, metres) — Webアプリと同じ定義
L, W = 28.0, 15.0
RIMX, KEYW, KEYL, THREER, THREESIDE, THREESTRAIGHT, CCR = 1.575, 4.9, 5.8, 6.75, 0.9, 2.99, 1.8
def landmarks():
    yc, kw = W/2, KEYW/2
    out = {}
    for side, s in ((0, '左 '), (1, '右 ')):
        mx = (lambda x: L-x) if side else (lambda x: x)
        out[s+'c0'] = (mx(0), 0); out[s+'c1'] = (mx(0), W)
        out[s+'k0'] = (mx(0), yc-kw); out[s+'k1'] = (mx(0), yc+kw)
        out[s+'f0'] = (mx(KEYL), yc-kw); out[s+'f1'] = (mx(KEYL), yc+kw); out[s+'fm'] = (mx(KEYL), yc)
        out[s+'t0'] = (mx(THREESTRAIGHT), THREESIDE); out[s+'t1'] = (mx(THREESTRAIGHT), W-THREESIDE)
        out[s+'tt'] = (mx(RIMX+THREER), yc); out[s+'rim'] = (mx(RIMX), yc)
    out['h0'] = (L/2, 0); out['h1'] = (L/2, W); out['cc'] = (L/2, W/2); out['cb'] = (L/2, W/2-CCR); out['ct'] = (L/2, W/2+CCR)
    return out
LM = landmarks()

def calib_H(c):
    """image(normalized u,v) -> court(x,y)"""
    src, dst = [], []
    for k, uv in c.get('pts', {}).items():
        if k in LM:
            src.append(uv); dst.append(LM[k])
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), 0)
    return H

class HProvider:
    """時刻 t での射影変換 (正規化画像座標 → コート座標)。scan.Hs があればそれを、なければ位置合わせを時刻で選ぶ。"""
    def __init__(self, project):
        self.hs = []
        sc = project.get('scan') or {}
        for h in sc.get('Hs') or []:
            if isinstance(h, dict):
                t, H = h['t'], np.array(h['H'], dtype=np.float64).reshape(3, 3)
            else:
                t, H = h[0], np.array(h[1:10], dtype=np.float64).reshape(3, 3)
            self.hs.append((float(t), H))
        self.hs.sort(key=lambda x: x[0])
        self.calibs = []
        for c in project.get('calibs') or []:
            H = calib_H(c)
            if H is not None:
                self.calibs.append((float(c.get('t', 0)), H))
        self.calibs.sort(key=lambda x: x[0])
        self.ts = np.array([t for t, _ in self.hs]) if self.hs else None
    def at(self, t):
        if self.hs:
            i = int(np.searchsorted(self.ts, t))
            cands = [j for j in (i-1, i) if 0 <= j < len(self.hs)]
            j = min(cands, key=lambda j: abs(self.hs[j][0]-t))
            if abs(self.hs[j][0]-t) <= 1.0:
                return self.hs[j][1]
        H = None
        for ct, cH in self.calibs:
            if ct <= t + 1e-6:
                H = cH
        if H is None and self.calibs:
            H = self.calibs[0][1]
        return H
    def describe(self):
        return f"射影変換: scan.Hs {len(self.hs)}件, 位置合わせ {len(self.calibs)}件"

def img_to_court(H, u, v):
    p = H @ np.array([u, v, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0]/p[2]), float(p[1]/p[2])

# ---------------- video frames
def extract_frames(video, out_dir, t0, t1, fps, max_w=1280):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"動画を開けません: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / src_fps
    t1 = min(t1, dur) if dur else t1
    times = []
    n = int(math.floor((t1 - t0) * fps)) + 1
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        t = t0 + i / fps
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > max_w:
            frame = cv2.resize(frame, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        times.append(t)
    cap.release()
    return times, dur

# ---------------- SAM 2
def load_predictor(model, device):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    here = os.path.dirname(os.path.abspath(__file__))
    ckpts = {
        'tiny': ('configs/sam2.1/sam2.1_hiera_t.yaml', 'sam2.1_hiera_tiny.pt'),
        'small': ('configs/sam2.1/sam2.1_hiera_s.yaml', 'sam2.1_hiera_small.pt'),
        'base': ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
    }
    cfg, fname = ckpts[model]
    for d in (os.path.join(here, 'ckpt'), os.path.join(os.path.dirname(here), 'ckpt'), 'C:/Users/vashi/AppData/Local/Temp/ct_train/ckpt', '.'):
        path = os.path.join(d, fname)
        if os.path.exists(path):
            break
    else:
        sys.exit(f"チェックポイントが見つかりません: {fname}  (https://dl.fbaipublicfiles.com/segment_anything_2/092824/{fname})")
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA が使えないため CPU で実行します'); device = 'cpu'
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    pred = build_sam2_video_predictor(cfg, path, device=device)
    return pred, device

def mask_to_point(mask, kind):
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return None
    if kind == 'ball':
        return float(xs.mean()), float(ys.mean()), int(len(xs))
    # 選手: マスクの下端中央 = 足元
    y_bottom = np.percentile(ys, 98)
    sel = ys >= y_bottom - 3
    return float(xs[sel].mean()), float(y_bottom), int(len(xs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--project', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=6.0); ap.add_argument('--start', type=float, default=0.0); ap.add_argument('--end', type=float, default=1e9)
    ap.add_argument('--model', default='tiny', choices=['tiny', 'small', 'base']); ap.add_argument('--device', default='cpu')
    ap.add_argument('--chunk', type=float, default=20.0, help='一度に追跡する秒数（メモリ節約）')
    ap.add_argument('--keep-frames', action='store_true')
    a = ap.parse_args()

    project = json.load(open(a.project, encoding='utf-8'))
    prompts = project.get('prompts') or []
    if not prompts:
        sys.exit('JSONに手がかり(prompts)がありません。Webアプリの「手がかりを付ける」で選手とボールをクリックしてから保存してください。')
    hp = HProvider(project)
    print(hp.describe())
    if not hp.hs and not hp.calibs:
        sys.exit('位置合わせがありません。Webアプリで位置合わせをしてから保存してください。')
    if not hp.hs:
        print('注意: フレームごとの射影変換(scan.Hs)がないため、位置合わせを時刻で切り替えるだけになります。カメラが動く映像ではWebアプリで一度「動画全体を自動解析」を実行してから保存すると精度が上がります。')

    players = {p['id']: p for p in project['players']}
    entities = sorted({q['entity'] for q in prompts})
    print(f"追跡対象: {len(entities)} 体 ({', '.join(entities)})  手がかり {len(prompts)} 件")

    t_start = max(a.start, min(q['t'] for q in prompts) - 0.5)
    tmp = tempfile.mkdtemp(prefix='ct_frames_')
    print(f"フレーム抽出中 ({a.fps} fps) ...", flush=True)
    times, dur = extract_frames(a.video, tmp, t_start, a.end, a.fps)
    if not times:
        sys.exit('フレームを取り出せませんでした')
    print(f"  {len(times)} フレーム ({times[0]:.1f}s〜{times[-1]:.1f}s, 動画長 {dur:.1f}s)")

    pred, device = load_predictor(a.model, a.device)
    print(f"SAM 2 ({a.model}, {device}) 読み込み完了", flush=True)

    obj_ids = {e: i+1 for i, e in enumerate(entities)}
    id_obj = {v: k for k, v in obj_ids.items()}
    frames_per_chunk = max(2, int(a.chunk * a.fps))
    results = {e: [] for e in entities}   # (t, u, v, area)
    lost_at = {e: [] for e in entities}
    last_mask = {}                        # obj -> mask (bool) at last frame of previous chunk
    total = len(times); done = 0; t0 = time.time()
    import torch
    sample = cv2.imread(os.path.join(tmp, '00000.jpg')); fh, fw = sample.shape[:2]

    for c0 in range(0, total, frames_per_chunk):
        c1 = min(total, c0 + frames_per_chunk)
        cdir = os.path.join(tmp, f'chunk_{c0:05d}'); os.makedirs(cdir, exist_ok=True)
        for k, i in enumerate(range(c0, c1)):
            dst = os.path.join(cdir, f'{k:05d}.jpg')
            if not os.path.exists(dst):
                os.link(os.path.join(tmp, f'{i:05d}.jpg'), dst) if hasattr(os, 'link') else shutil.copy(os.path.join(tmp, f'{i:05d}.jpg'), dst)
        with torch.inference_mode():
            state = pred.init_state(video_path=cdir, offload_video_to_cpu=True, offload_state_to_cpu=(device == 'cpu'))
            added = set()
            # 手がかり（クリック）をこのチャンク内のフレームに追加
            for q in prompts:
                e = q['entity']; fi = int(round((q['t'] - times[0]) * a.fps)); li = fi - c0
                if 0 <= li < (c1 - c0):
                    pts = np.array([[q['u'] * fw, q['v'] * fh]], dtype=np.float32); lbl = np.array([1], dtype=np.int32)
                    pred.add_new_points_or_box(state, frame_idx=li, obj_id=obj_ids[e], points=pts, labels=lbl)
                    added.add(e)
            # 前チャンクの最後のマスクを引き継ぐ
            for oid, m in last_mask.items():
                e = id_obj[oid]
                if e in added or m is None or m.sum() < 6:
                    continue
                pred.add_new_mask(state, frame_idx=0, obj_id=oid, mask=m)
                added.add(e)
            if not added:
                done += (c1 - c0); shutil.rmtree(cdir, ignore_errors=True); continue
            chunk_masks = {}
            for fidx, oids, logits in pred.propagate_in_video(state):
                t = times[c0 + fidx]
                for j, oid in enumerate(oids):
                    m = (logits[j] > 0.0).squeeze().cpu().numpy()
                    e = id_obj[int(oid)]
                    pt = mask_to_point(m, 'ball' if e == 'ball' else 'player')
                    if pt is None:
                        if not lost_at[e] or t - lost_at[e][-1] > 1.0:
                            lost_at[e].append(t)
                        chunk_masks[int(oid)] = None
                        continue
                    results[e].append((t, pt[0] / fw, pt[1] / fh, pt[2]))
                    chunk_masks[int(oid)] = m
            last_mask = chunk_masks
            pred.reset_state(state)
        done += (c1 - c0)
        el = time.time() - t0; eta = el / done * (total - done) if done else 0
        print(f"  {done}/{total} フレーム  経過 {el/60:.1f}分  残り約 {eta/60:.1f}分", flush=True)
        shutil.rmtree(cdir, ignore_errors=True)

    # ---------------- コート座標へ変換して JSON に書き込む
    n_kf = 0
    for e, obs in results.items():
        kfs = []
        for (t, u, v, area) in obs:
            H = hp.at(t)
            if H is None:
                continue
            q = img_to_court(H, u, v)
            if q is None:
                continue
            x, y = q
            if x < -2 or x > L + 2 or y < -2 or y > W + 2:
                continue
            kfs.append({'t': round(t, 2), 'x': round(max(-1, min(L+1, x)), 2), 'y': round(max(-1, min(W+1, y)), 2), 'v': True})
        # 見失った時刻に非表示キーを入れる
        for tl in lost_at[e]:
            kfs.append({'t': round(tl, 2), 'v': False})
        kfs.sort(key=lambda k: k['t'])
        if e == 'ball':
            project['ball']['kfs'] = kfs
        elif e in players:
            players[e]['kfs'] = kfs
        n_kf += len(kfs)
        print(f"  {e}: {len(obs)} 点, 見失い {len(lost_at[e])} 回" + (f" (最初: {lost_at[e][0]:.1f}s)" if lost_at[e] else ''))
    project.setdefault('settings', {})['seg'] = {'a': round(times[0], 2), 'b': round(times[-1], 2)}
    project['pcTrack'] = {'model': a.model, 'fps': a.fps, 'frames': total, 'lost': {e: [round(x, 1) for x in v] for e, v in lost_at.items()}}
    json.dump(project, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)
    if not a.keep_frames:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"完了: {a.out} に {n_kf} キーフレームを書き込みました。Webアプリの「プロジェクト読込」で開いてください。")

if __name__ == '__main__':
    main()
