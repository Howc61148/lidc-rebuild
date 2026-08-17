"""
前處理結果目視驗證

將標註框畫回影像，確認框的位置與影像亮度是否正確。
座標軸顛倒、框偏移這類錯誤不會產生錯誤訊息，只能以目視發現。

用法:
    python visualize.py
    python visualize.py --multi-only --out check_multi.png
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd

TILE_SIZE = 384
BOX_PADDING = 6      # 結節通常很小，框外擴數像素才看得清楚


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/LIDC-IDRI/processed")
    ap.add_argument("--n", type=int, default=9, help="顯示幾張")
    ap.add_argument("--out", default="check.png")
    ap.add_argument("--multi-only", action="store_true",
                    help="只看含多顆結節的切片")
    args = ap.parse_args()

    mf = pd.read_csv(os.path.join(args.data, "manifest.csv"))
    sel = mf[mf.n_boxes > 1] if args.multi_only else mf[mf.n_boxes > 0]
    if sel.empty:
        print("找不到符合條件的影像")
        return
    sel = sel.head(args.n)

    tiles = []
    for _, r in sel.iterrows():
        img = cv2.imread(os.path.join(args.data, "images", r.filename),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        lbl = os.path.join(args.data, "labels",
                           r.filename.replace(".png", ".txt"))
        with open(lbl) as f:
            for line in f:
                _, cx, cy, bw, bh = map(float, line.split())
                x1 = int((cx - bw / 2) * w) - BOX_PADDING
                y1 = int((cy - bh / 2) * h) - BOX_PADDING
                x2 = int((cx + bw / 2) * w) + BOX_PADDING
                y2 = int((cy + bh / 2) * h) + BOX_PADDING
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(vis, f"{r.patient_id} z={r.slice_idx} n={r.n_boxes}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(cv2.resize(vis, (TILE_SIZE, TILE_SIZE)))

    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    tiles += [np.zeros((TILE_SIZE, TILE_SIZE, 3), np.uint8)] * (cols * rows - len(tiles))
    grid = np.vstack([np.hstack(tiles[i * cols:(i + 1) * cols])
                      for i in range(rows)])

    cv2.imwrite(args.out, grid)
    print(f"已存成 {args.out}（{len(sel)} 張）")

    # 灰階分布可確認肺窗是否生效：套用肺窗後應撐滿 0-255，
    # 平均值約 60-130；若集中於某一端則窗位設定有誤
    px = np.concatenate([
        cv2.imread(os.path.join(args.data, "images", f),
                   cv2.IMREAD_GRAYSCALE).ravel()
        for f in sel.filename
    ])
    print(f"灰階分布: min={px.min()} max={px.max()} mean={px.mean():.1f}")


if __name__ == "__main__":
    main()