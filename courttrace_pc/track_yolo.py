#!/usr/bin/env python
"""
コートトレース PC解析（CPUでも実用的）: YOLO + ByteTrack で選手とボールを追跡し、プロジェクトJSONに書き込む。

  python track_yolo.py --video 試合.mp4 --project 試合.courttrace.json --out 試合.tracked.json
  オプション: --model yolov10s.pt (yolov8s.pt / yolo11s.pt でも可) --imgsz 1280 --stride 3 (3フレームに1回=約10fps)
              --start 0 --end 600 (秒)

必要な入力JSON: calibs（位置合わせ）。scan.Hs（ブラウザ解析のフレームごとの射影変換）があればカメラの動きに追従。
prompts（Webアプリの「手がかり」）があれば、その位置にいる追跡IDに背番号を割り当てる。
出力: 手がかりで番号が付いた選手は players[].kfs に、それ以外の人は scan.tracks（点線コマ）に、ボールは ball.kfs に。
"""
import argparse, json, math, sys, time
import numpy as np
import cv2
from track_sam2 import HProvider, img_to_court, L, W

def torso_color(frame, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    sx, sy = int(x1 + 0.3 * w), int(y1 + 0.18 * h); ex, ey = int(x1 + 0.7 * w), int(y1 + 0.5 * h)
    crop = frame[max(0, sy):max(sy + 1, ey), max(0, sx):max(sx + 1, ex)]
    if crop.size == 0:
        return None
    b, g, r = cv2.mean(crop)[:3]
    return (r, g, b)

def feat(c):
    mx, mn = max(c), min(c); V = mx / 255; S = (mx - mn) / mx if mx else 0
    return np.array([V * 2, S * 1.2, (c[0] - c[1]) / 255 * 1.5, (c[1] - c[2]) / 255 * 1.5])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--project', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='yolov10s.pt'); ap.add_argument('--imgsz', type=int, default=1280); ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--start', type=float, default=0.0); ap.add_argument('--end', type=float, default=1e9); ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--margin', type=float, default=0.7, help='コート外の許容 (m)')
    a = ap.parse_args()

    from ultralytics import YOLO
    project = json.load(open(a.project, encoding='utf-8'))
    hp = HProvider(project); print(hp.describe())
    if not hp.hs and not hp.calibs:
        sys.exit('位置合わせがありません。Webアプリで位置合わせをしてから保存してください。')
    prompts = project.get('prompts') or []
    players = {p['id']: p for p in project['players']}

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f'動画を開けません: {a.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = int(a.start * src_fps); f1 = min(nframes, int(a.end * src_fps)) if nframes else int(a.end * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    model = YOLO(a.model)
    print(f'YOLO {a.model} 読み込み完了。{fw}x{fh} {src_fps:.1f}fps, フレーム {f0}〜{f1}, stride {a.stride}（約 {src_fps/a.stride:.1f} fps で追跡）', flush=True)

    tracks = {}   # tid -> {'pts':[(t,x,y,u,v)], 'cols':[]}
    ball_obs = [] # (t,x,y,tid_near)
    t0 = time.time(); fi = f0; processed = 0
    while fi < f1:
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - f0) % a.stride:
            fi += 1; continue
        t = fi / src_fps
        H = hp.at(t)
        res = model.track(frame, persist=True, tracker='bytetrack.yaml', imgsz=a.imgsz, conf=a.conf, classes=[0, 32], verbose=False)[0]
        cur = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy(); cls = res.boxes.cls.cpu().numpy().astype(int); ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.full(len(cls), -1)
            for (x1, y1, x2, y2), c, tid in zip(xyxy, cls, ids):
                u, v = float(((x1 + x2) / 2) / fw), float(y2 / fh); x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                if H is None:
                    continue
                q = img_to_court(H, u, v)
                if q is None:
                    continue
                x, y = q
                if c == 0:
                    if x < -a.margin or x > L + a.margin or y < -a.margin or y > W + a.margin:
                        continue
                    if tid < 0:
                        continue
                    tr = tracks.setdefault(int(tid), {'pts': [], 'cols': [], 'boxes': []})
                    tr['pts'].append((t, float(np.clip(x, -1, L + 1)), float(np.clip(y, -1, W + 1)), u, v))
                    tr['boxes'].append((t, x1 / fw, y1 / fh, x2 / fw, y2 / fh))
                    col = torso_color(frame, (x1, y1, x2, y2))
                    if col: tr['cols'].append(col)
                    cur.append((int(tid), x, y))
                elif c == 32:
                    if x < -2 or x > L + 2 or y < -2 or y > W + 2:
                        continue
                    ball_obs.append([t, float(x), float(y), None, u, v])
        # ボールの保持者 = 同フレームで最も近い選手
        for o in ball_obs:
            if o[0] == t and cur:
                tid, d = min(((tid, math.hypot(cx - o[1], cy - o[2])) for tid, cx, cy in cur), key=lambda z: z[1])
                if d < 2.0: o[3] = tid
        processed += 1; fi += 1
        if processed % 50 == 0:
            el = time.time() - t0; done = (fi - f0) / max(1, f1 - f0)
            print(f'  {t:7.1f}s  {done*100:4.0f}%  経過 {el/60:.1f}分  残り約 {el/done*(1-done)/60:.1f}分  追跡中 {len(cur)}人', flush=True)
    cap.release()

    # ---- 短すぎる追跡を捨て、手がかりで背番号を割り当てる
    good = {tid: tr for tid, tr in tracks.items() if len(tr['pts']) >= 3}
    assigned = {}  # tid -> entity
    for q in prompts:
        e = q['entity']
        if e == 'ball':
            continue
        best, bd = None, 1e9
        for tid, tr in good.items():
            for (bt, x1, y1, x2, y2) in tr['boxes']:
                if abs(bt - q['t']) <= 0.35 and x1 <= q['u'] <= x2 and y1 <= q['v'] <= y2:
                    d = abs(bt - q['t']) + abs((x1 + x2) / 2 - q['u'])
                    if d < bd: bd, best = d, tid
        if best is not None:
            assigned[best] = e
    # 同じ選手に複数の追跡IDが割り当てられた場合は時系列で連結される（kfs をまとめる）
    n_kf = 0
    for pid in players: players[pid]['kfs'] = []
    for tid, e in assigned.items():
        if e in players:
            players[e]['kfs'] += [{'t': round(t, 2), 'x': round(x, 2), 'y': round(y, 2), 'v': True} for (t, x, y, u, v) in good[tid]['pts']]
    for pid, p in players.items():
        p['kfs'].sort(key=lambda k: k['t']); n_kf += len(p['kfs'])

    # ---- 残りの追跡はチーム色で2分割して点線コマ（scan.tracks）に
    rest = {tid: tr for tid, tr in good.items() if tid not in assigned}
    feats = {tid: np.median(np.array([feat(c) for c in tr['cols']]), axis=0) for tid, tr in rest.items() if tr['cols']}
    clusters = [{'i': 0, 'n': 0, 'rgb': [128, 128, 128], 'team': 'A'}, {'i': 1, 'n': 0, 'rgb': [128, 128, 128], 'team': 'B'}, {'i': 2, 'n': 0, 'rgb': [110, 110, 110], 'team': 'X', 'other': True}]
    kmap = {}
    if len(feats) >= 2:
        X = np.array(list(feats.values())); ids = list(feats.keys())
        # 初期値: 長い追跡（選手らしいもの）の中で明るさの下位/上位30%の平均（観客や審判の外れ値に引きずられない）
        lens = np.array([len(rest[t]['pts']) for t in ids]); longi = np.where(lens >= max(10, np.percentile(lens, 60)))[0]
        if len(longi) < 4: longi = np.argsort(-lens)[:max(4, len(ids)//3)]
        order = longi[np.argsort(X[longi, 0])]; k3 = max(1, len(order)//3)
        cen = np.array([X[order[:k3]].mean(0), X[order[-k3:]].mean(0)])
        for _ in range(30):
            lab = np.argmin(((X[:, None, :] - cen[None]) ** 2).sum(2), axis=1)
            for k in range(2):
                if (lab == k).any(): cen[k] = X[lab == k].mean(0)
        d = np.sqrt(np.min(((X[:, None, :] - cen[None]) ** 2).sum(2), axis=1))
        for k in range(2):
            md = np.median(d[lab == k]) if (lab == k).any() else 0
            for i in np.where(lab == k)[0]:
                kmap[ids[i]] = 2 if d[i] > max(0.35, 2.4 * md) else k
        for k in range(2):
            cols = [np.median(np.array(rest[ids[i]]['cols']), axis=0) for i in range(len(ids)) if kmap.get(ids[i]) == k]
            if cols: clusters[k]['rgb'] = [float(v) for v in np.mean(cols, axis=0)]
        if sum(clusters[1]['rgb']) > sum(clusters[0]['rgb']):
            clusters[0]['team'], clusters[1]['team'] = 'B', 'A'
    scan_tracks = []
    for tid, tr in rest.items():
        k = kmap.get(tid, 2); clusters[k]['n'] += 1
        scan_tracks.append({'id': int(tid), 'k': int(k), 'player': None, 'pts': [[round(t, 2), round(x, 2), round(y, 2), round(u, 4), round(v, 4)] for (t, x, y, u, v) in tr['pts']]})
    # ---- ボール: 静止した誤検出を除去、保持者付きで ball.kfs へ
    keep = []
    for o in ball_obs:
        n = sum(1 for p in ball_obs if math.hypot(p[1] - o[1], p[2] - o[2]) < 0.6)
        span = max(p[0] for p in ball_obs if math.hypot(p[1] - o[1], p[2] - o[2]) < 0.6) - min(p[0] for p in ball_obs if math.hypot(p[1] - o[1], p[2] - o[2]) < 0.6)
        if not (n >= 6 and span >= 8): keep.append(o)
    ball_kfs = []
    for o in keep:
        kf = {'t': round(o[0], 2), 'x': round(o[1], 2), 'y': round(o[2], 2), 'v': True}
        if o[3] is not None and o[3] in assigned: kf['holder'] = assigned[o[3]]
        ball_kfs.append(kf)
    project['ball']['kfs'] = ball_kfs
    tt = [tr['pts'][0][0] for tr in good.values()] + [tr['pts'][-1][0] for tr in good.values()]
    project['scan'] = {'a': round(min(tt), 2) if tt else 0, 'b': round(max(tt), 2) if tt else 0, 'dt': a.stride / src_fps, 'clusters': clusters,
                       'tracks': scan_tracks, 'ball': [[round(o[0], 2), round(o[1], 2), round(o[2], 2), 0] for o in keep], 'Hs': (project.get('scan') or {}).get('Hs', [])}
    project.setdefault('settings', {})['seg'] = {'a': project['scan']['a'], 'b': project['scan']['b']}
    json.dump(project, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    print(f'完了: 追跡 {len(good)} 本（背番号割り当て {len(assigned)} 本）、ボール {len(keep)} 回、キーフレーム {n_kf} 件 → {a.out}')
    print('Webアプリの「プロジェクト読込」で開いてください。')

if __name__ == '__main__':
    main()
