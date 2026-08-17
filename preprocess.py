"""
LIDC-IDRI → YOLO 偵測訓練資料

用法:
    python preprocess.py --limit 200
    python preprocess.py --limit 1018 --out D:/LIDC-IDRI/processed
"""

import argparse
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm

# ── 參數 ──────────────────────────────────────────────────
MIN_READERS = 3        # 共識門檻，採 LUNA16 標準（4 位醫師中至少 3 位）
CONSENSUS_LEVEL = 0.5  # 框的範圍：至少半數醫師認定屬於病灶的像素
WINDOW_CENTER = -600   # 肺窗窗位，須與 gui_app/predictor.py 一致
WINDOW_WIDTH = 1500    # 肺窗窗寬，須與 gui_app/predictor.py 一致
NEG_RATIO = 1.5        # 負樣本數 = 正樣本 × 此值
NEG_MARGIN = 2         # 距任何標註切片 N 張以內的不取為負樣本
MIN_BOX_PX = 3         # 框小於此像素數即捨棄
SEED = 42


def apply_lung_window(volume):
    """HU 轉 8-bit 灰階。

    固定窗界而非逐張 min-max：金屬植入物的 HU 可達 8000 以上，
    逐張正規化會使含金屬的切片整張被壓暗，同一結節在相鄰切片
    呈現不同亮度。固定窗同時確保與推論端的灰階處理一致。
    """
    lo = WINDOW_CENTER - WINDOW_WIDTH / 2
    hi = WINDOW_CENTER + WINDOW_WIDTH / 2
    v = np.clip(volume, lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    return v.astype(np.uint8)


def mask_to_bbox(mask2d):
    """從 2D 遮罩取外接矩形，回傳 (x1, y1, x2, y2)；全空則回 None。"""
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def analyse_scan(scan):
    """分析單一病人的標註。

    回傳 (slice_boxes, bad_slices, touched, nodule_rows)：
      slice_boxes  {切片編號: [框, ...]}  合格結節的標註
      bad_slices   含低共識結節、須整張排除的切片
      touched      被任何標註碰到的切片，供負樣本避開
      nodule_rows  每顆結節的紀錄
    """
    slice_boxes = defaultdict(list)
    bad_slices, touched = set(), set()
    nodule_rows = []

    # cluster_annotations 依三維位置將各醫師的標記歸群成物理結節。
    # 共識判斷必須在此層級進行：以切片為單位計數會把
    # 「A 醫師標結節甲、B 醫師標結節乙」誤判為兩人對同一顆有共識。
    for nid, anns in enumerate(scan.cluster_annotations(verbose=False), start=1):
        n_slices = set()
        for a in anns:
            n_slices.update(int(k) for k in a.contour_slice_indices)
        touched.update(n_slices)

        n_readers = len(anns)
        malig = float(np.median([a.malignancy for a in anns]))
        diam = float(np.median([a.diameter for a in anns]))

        row = dict(patient_id=scan.patient_id, nodule_idx=nid,
                   n_readers=n_readers, malignancy_median=malig,
                   diameter_median_mm=round(diam, 2),
                   n_slices=len(n_slices))

        if n_readers < MIN_READERS:
            # 低共識結節不採為正樣本，但其所在切片也不能保留：
            # 該結節實體上存在於影像中，若保留為「乾淨」影像，
            # 等同標註一顆真實結節為背景。
            bad_slices.update(n_slices)
            row["status"] = f"excluded_low_consensus({n_readers}<{MIN_READERS})"
            nodule_rows.append(row)
            continue

        cmask, cbbox, _ = consensus(anns, clevel=CONSENSUS_LEVEL)
        y0, x0, z0 = cbbox[0].start, cbbox[1].start, cbbox[2].start

        kept = 0
        for i in range(cmask.shape[2]):
            bb = mask_to_bbox(cmask[:, :, i])
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            box = (x0 + x1, y0 + y1, x0 + x2, y0 + y2)  # 轉回原圖座標
            if (box[2] - box[0]) < MIN_BOX_PX or (box[3] - box[1]) < MIN_BOX_PX:
                continue
            slice_boxes[z0 + i].append(box)
            kept += 1

        row["status"] = "included"
        row["n_slices_with_box"] = kept
        nodule_rows.append(row)

    return slice_boxes, bad_slices, touched, nodule_rows


def process_scan(scan, img_dir, lbl_dir, rng):
    """處理單一病人，輸出影像與標註，回傳 manifest 資料列。"""
    slice_boxes, bad_slices, touched, nodule_rows = analyse_scan(scan)

    pos = sorted(set(slice_boxes) - bad_slices)
    if not pos:
        return [], nodule_rows

    vol = scan.to_volume(verbose=False)
    img = apply_lung_window(vol)
    h, w, n_sl = img.shape

    # 負樣本取自同一批病人，避免模型學到掃描儀差異而非病灶特徵。
    # NEG_MARGIN 排除標註切片附近的切片，因結節為立體，
    # 邊緣切片可能仍有殘影。
    forbidden = set()
    for k in touched:
        forbidden.update(range(k - NEG_MARGIN, k + NEG_MARGIN + 1))
    neg_pool = [k for k in range(n_sl) if k not in forbidden]
    n_neg = min(len(neg_pool), int(round(len(pos) * NEG_RATIO)))
    neg = sorted(rng.sample(neg_pool, n_neg)) if n_neg else []

    rows = []
    for k in pos + neg:
        name = f"{scan.patient_id}_{k:04d}"
        cv2.imwrite(os.path.join(img_dir, name + ".png"), img[:, :, k])

        boxes = slice_boxes.get(k, []) if k in pos else []
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            for x1, y1, x2, y2 in boxes:
                f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")

        rows.append(dict(
            patient_id=scan.patient_id, slice_idx=k, filename=name + ".png",
            n_boxes=len(boxes), is_positive=int(k in pos),
            slice_thickness=scan.slice_thickness,
            pixel_spacing=scan.pixel_spacing,
            img_h=h, img_w=w,
        ))
    return rows, nodule_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/LIDC-IDRI/processed")
    ap.add_argument("--limit", type=int, default=5, help="處理幾位病人")
    ap.add_argument("--max-thickness", type=float, default=None,
                    help="只收切片厚度 <= 此值的病人（預設不篩）")
    args = ap.parse_args()

    rng = random.Random(SEED)
    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    q = pl.query(pl.Scan)
    if args.max_thickness:
        q = q.filter(pl.Scan.slice_thickness <= args.max_thickness)
    scans = sorted(q.all(), key=lambda s: s.patient_id)[:args.limit]

    print(f"預計處理 {len(scans)} 位病人 → {args.out}\n")

    manifest, nodules = [], []
    for scan in tqdm(scans, desc="處理中"):
        try:
            r, n = process_scan(scan, img_dir, lbl_dir, rng)
            manifest += r
            nodules += n
        except Exception as e:
            print(f"\n  [略過] {scan.patient_id}: {type(e).__name__}: {e}")

    if not manifest:
        print("沒有產出任何資料，檢查上方錯誤訊息")
        return

    mf = pd.DataFrame(manifest)
    nf = pd.DataFrame(nodules)
    mf.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    nf.to_csv(os.path.join(args.out, "nodules.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"病人數        : {mf.patient_id.nunique()}")
    print(f"影像總數      : {len(mf)}")
    print(f"  正樣本      : {int(mf.is_positive.sum())}")
    print(f"  負樣本      : {int((1 - mf.is_positive).sum())}")
    print(f"標註框總數    : {int(mf.n_boxes.sum())}")
    print(f"  含多顆結節的切片: {int((mf.n_boxes > 1).sum())}")
    print(f"\n結節總數      : {len(nf)}")
    for st, c in nf.status.value_counts().items():
        print(f"  {st}: {c}")
    print("=" * 60)
    print(f"\n輸出於 {args.out}")


if __name__ == "__main__":
    main()