"""
匯出完整掃描與結節層級標準答案，供病灶層級評估使用

訓練資料只含正樣本切片與抽樣的負樣本。若在其上評估，未被抽入的
切片可能產生的誤報不會被計入，FP/scan 會被低估。本腳本對指定 split
的每次掃描輸出全部切片，並直接由 pylidc 的標註產生結節層級的標準答案，
評估時不再從 2D 框重新推測結節身分。

低共識結節（醫師人數低於 MIN_READERS）記為忽略區：模型偵測到它們
不算對也不算錯，與 LUNA16 對非參考標準結節的處理方式相同。

用法:
    python export_eval.py --split test
    python export_eval.py --split val
"""

import argparse
import json
import os

import cv2
import numpy as np
import pylidc as pl
from tqdm import tqdm

from preprocess import MIN_READERS, apply_lung_window


def nodule_gt(scan):
    """回傳該次掃描所有結節的中心、直徑與共識狀態。"""
    nodules = []
    for nid, anns in enumerate(scan.cluster_annotations(verbose=False), start=1):
        # ann.centroid 為 [列, 欄, 切片] 的影像座標，多位醫師取平均
        c = np.mean([a.centroid for a in anns], axis=0)
        nodules.append({
            "id": nid,
            "n_readers": len(anns),
            "status": "included" if len(anns) >= MIN_READERS else "ignore",
            "center_px": [float(c[1]), float(c[0]), float(c[2])],   # x, y, z
            "diameter_mm": float(np.median([a.diameter for a in anns])),
        })
    return nodules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/LIDC-IDRI/processed")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    args = ap.parse_args()

    with open(os.path.join(args.data, "split.json")) as f:
        patients = set(json.load(f)[args.split])

    out_dir = os.path.join(args.data, f"eval_{args.split}")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    scans = [s for s in pl.query(pl.Scan).all() if s.patient_id in patients]
    scans.sort(key=lambda s: (s.patient_id, s.id))
    print(f"{args.split} 集：{len(patients)} 位病人，{len(scans)} 次掃描\n")

    gt = {}
    for scan in tqdm(scans, desc="匯出中"):
        key = f"{scan.patient_id}_s{scan.id}"
        vol = scan.to_volume(verbose=False)
        img = apply_lung_window(vol)
        h, w, n = img.shape

        for k in range(n):
            cv2.imwrite(os.path.join(img_dir, f"{key}_{k:04d}.png"), img[:, :, k])

        # slice_spacing 為相鄰切片的實際間距，與 slice_thickness 可能不同
        gt[key] = {
            "patient_id": scan.patient_id,
            "scan_id": scan.id,
            "n_slices": int(n),
            "img_h": int(h), "img_w": int(w),
            "pixel_spacing_mm": float(scan.pixel_spacing),
            "slice_spacing_mm": float(scan.slice_spacing or scan.slice_thickness),
            "nodules": nodule_gt(scan),
        }

    with open(os.path.join(out_dir, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=1, ensure_ascii=False)

    n_inc = sum(1 for s in gt.values() for nd in s["nodules"] if nd["status"] == "included")
    n_ign = sum(1 for s in gt.values() for nd in s["nodules"] if nd["status"] == "ignore")
    n_img = sum(s["n_slices"] for s in gt.values())
    print(f"\n輸出 {n_img} 張切片、{n_inc} 顆評估結節、{n_ign} 顆忽略區結節")
    print(f"  → {out_dir}")


if __name__ == "__main__":
    main()
