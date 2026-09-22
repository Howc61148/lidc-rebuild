"""
病灶層級評估（完整掃描）

在 export_eval.py 匯出的完整掃描上推論，以結節層級的標準答案計分，
產出 FROC 曲線、CPM 與病人層級 bootstrap 信賴區間。

與前一版（evaluate_v1_sampled.py）的差異：
  - 推論範圍為每次掃描的全部切片，FP/scan 以完整掃描計
  - 標準答案直接使用結節身分，不再由 2D 框重新聚合
  - 命中判定改為三維距離（mm）小於結節半徑，參考 LUNA16
  - 低共識結節為忽略區；同一結節的重複偵測不計為誤報

用法:
    python evaluate.py --weights runs/detect/xxx/weights/best.pt --split val
    python evaluate.py --weights ... --split test    # 僅於模型定案後使用
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

CPM_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8]   # LUNA16 定義的七個取樣點
MAX_SLICE_GAP = 2                              # 預測聚合：z 方向容許的最大間隔
SEED = 42


def parse_stem(stem):
    """檔名 patient_sSCANID_ZZZZ → (scan_key, z)。"""
    pid, sid, z = stem.rsplit("_", 2)
    return f"{pid}_{sid}", int(z)


def cluster_predictions(boxes):
    """將各切片的預測框聚合為 3D 候選。

    boxes: [(z, cx, cy, w, h, score), ...]，座標為像素。
    相鄰切片且框重疊者視為同一候選；分數取最大。
    不以固定像素距離判斷，避免不同大小的結節用同一門檻。
    """
    boxes = sorted(boxes, key=lambda b: b[0])
    clusters = []
    for z, cx, cy, w, h, s in boxes:
        r = max(w, h) / 2
        placed = False
        for c in clusters:
            if z - c["z_max"] > MAX_SLICE_GAP:
                continue
            # 與該候選最後一張切片上的框比較：中心落在對方半徑內即視為重疊
            rz, rcx, rcy, rr = c["last"]
            if np.hypot(cx - rcx, cy - rcy) <= max(r, rr):
                c["pts"].append((cx, cy, z))
                c["score"] = max(c["score"], s)
                if z >= c["z_max"]:
                    c["z_max"] = z
                    c["last"] = (z, cx, cy, r)
                placed = True
                break
        if not placed:
            clusters.append({"pts": [(cx, cy, z)], "score": s,
                             "z_max": z, "last": (z, cx, cy, r)})

    for c in clusters:
        p = np.array(c["pts"])
        c["center_px"] = p.mean(axis=0).tolist()
    return clusters


def to_mm(center_px, ps, ss):
    """像素座標 (x, y, z) → mm。"""
    x, y, z = center_px
    return np.array([x * ps, y * ps, z * ss])


def match_scan(preds, scan_gt):
    """Classify candidates using the project's positive-reference-first rule.

    Preserve the existing nearest-included-target assignment, strict radius
    boundary, coordinate conversion, and duplicate suppression. A candidate
    is ignored only if it matches no included target. n_ign counts CANDIDATES,
    not unique ignored nodules. This is not a full LUNA16 implementation.
    """
    ps, ss = scan_gt["pixel_spacing_mm"], scan_gt["slice_spacing_mm"]
    targets = [(to_mm(n["center_px"], ps, ss), n["diameter_mm"] / 2, n["id"])
               for n in scan_gt["nodules"] if n["status"] == "included"]
    ignores = [(to_mm(n["center_px"], ps, ss), n["diameter_mm"] / 2)
               for n in scan_gt["nodules"] if n["status"] == "ignore"]
    hit = set()
    records, n_ign, n_dup = [], 0, 0

    for c in sorted(preds, key=lambda c: -c["score"]):
        p = to_mm(c["center_px"], ps, ss)

        # Included targets take precedence over overlapping ignore regions.
        best = None
        for tc, tr, tid in targets:
            d = np.linalg.norm(p - tc)
            if d < tr and (best is None or d < best[0]):
                best = (d, tid)
        if best is not None:
            if best[1] in hit:
                n_dup += 1
            else:
                hit.add(best[1])
                records.append((c["score"], True))
        elif any(np.linalg.norm(p - ic) < ir for ic, ir in ignores):
            n_ign += 1
        else:
            records.append((c["score"], False))

    return records, len(targets), n_ign, n_dup


def froc(records, n_scans, n_gt):
    """Emit one curve point per DISTINCT score threshold.

    All candidates with exactly equal scores enter together. Retain the
    existing sens_at() stepwise convention; do not add interpolation or round
    scores. As in the previous implementation, no initial origin is emitted.
    """
    tp = fp = 0
    curve = []
    ordered = sorted(records, key=lambda r: -r[0])
    for i, (score, is_tp) in enumerate(ordered):
        if is_tp:
            tp += 1
        else:
            fp += 1
        # A single threshold cannot select only some candidates with this score.
        if i + 1 < len(ordered) and ordered[i + 1][0] == score:
            continue
        curve.append((fp / n_scans, tp / n_gt))
    return curve


def sens_at(curve, target_fp):
    return max((rec for fps, rec in curve if fps <= target_fp), default=0.0)


def bootstrap_ci(per_patient, n_boot=1000, target_fp=2.0, seed=SEED):
    """以病人為單位有放回重抽。同一病人的結節彼此相關，
    以結節為單位重抽會高估有效樣本數，使區間過窄。"""
    rng = np.random.default_rng(seed)
    pids = list(per_patient)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(pids), size=len(pids), replace=True)
        recs, n_gt, n_scans = [], 0, 0
        for i in pick:
            d = per_patient[pids[i]]
            recs += d["records"]
            n_gt += d["n_gt"]
            n_scans += d["n_scans"]
        if n_gt and n_scans:
            vals.append(sens_at(froc(recs, n_scans, n_gt), target_fp))
    if not vals:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="D:/LIDC-IDRI/processed")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.01,
                    help="推論門檻須設低，FROC 才能掃完整條曲線")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="推論尺寸；預設沿用訓練時的設定")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "test":
        print("※ 正在使用 test 集，此集僅應於模型定案後使用。\n")

    eval_dir = os.path.join(args.data, f"eval_{args.split}")
    with open(os.path.join(eval_dir, "gt.json"), encoding="utf-8") as f:
        gt = json.load(f)
    img_dir = os.path.join(eval_dir, "images")
    img_paths = sorted(os.path.join(img_dir, fn) for fn in os.listdir(img_dir)
                       if fn.endswith(".png"))

    n_scans = len(gt)
    n_patients = len({s["patient_id"] for s in gt.values()})
    n_gt = sum(1 for s in gt.values() for n in s["nodules"] if n["status"] == "included")
    n_ign_gt = sum(1 for s in gt.values() for n in s["nodules"] if n["status"] == "ignore")
    print(f"{args.split} 集：{n_patients} 位病人、{n_scans} 次掃描、{len(img_paths)} 張切片")
    print(f"評估結節 {n_gt} 顆，忽略區結節 {n_ign_gt} 顆\n")

    model = YOLO(args.weights)
    imgsz = args.imgsz or model.overrides.get("imgsz", 640)
    print(f"推論中（imgsz={imgsz}）...")

    raw = defaultdict(list)
    for i in range(0, len(img_paths), 64):
        batch = img_paths[i:i + 64]
        for path, res in zip(batch, model.predict(batch, conf=args.conf,
                                                  imgsz=imgsz, verbose=False)):
            key, z = parse_stem(os.path.basename(path)[:-4])
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                raw[key].append((z, (x1 + x2) / 2, (y1 + y2) / 2,
                                 x2 - x1, y2 - y1, float(b.conf[0])))

    print("聚合並匹配...")
    per_patient = defaultdict(lambda: {"records": [], "n_gt": 0, "n_scans": 0})
    all_records, tot_ign, tot_dup = [], 0, 0
    for key, scan_gt in gt.items():
        preds = cluster_predictions(raw.get(key, []))
        recs, k_gt, n_ign, n_dup = match_scan(preds, scan_gt)
        pid = scan_gt["patient_id"]
        per_patient[pid]["records"] += recs
        per_patient[pid]["n_gt"] += k_gt
        per_patient[pid]["n_scans"] += 1
        all_records += recs
        tot_ign += n_ign
        tot_dup += n_dup

    curve = froc(all_records, n_scans, n_gt)
    sens = [sens_at(curve, pt) for pt in CPM_POINTS]
    cpm = float(np.mean(sens))
    r2 = sens_at(curve, 2.0)
    ci = bootstrap_ci(dict(per_patient), target_fp=2.0)
    n_tp = sum(1 for _, t in all_records if t)
    n_fp = len(all_records) - n_tp

    print("\n" + "=" * 60)
    print(f"病灶層級評估（{args.split} 集，完整掃描）")
    print("=" * 60)
    print(f"掃描 {n_scans}   病人 {n_patients}   評估結節 {n_gt}")
    print(f"候選總數 {len(all_records) + tot_ign + tot_dup}："
          f"TP {n_tp}、FP {n_fp}、落入忽略區 {tot_ign}、重複命中 {tot_dup}\n")
    print("FROC — 各假陽性率下的召回率：")
    for pt, s in zip(CPM_POINTS, sens):
        print(f"  {pt:>6.3f} FP/scan : {s:.3f}")
    print(f"\nCPM（七點平均）: {cpm:.3f}")
    print(f"召回率 @ 2 FP/scan: {r2:.3f}")
    if ci:
        print(f"  95% CI（病人層級 bootstrap, 1000 次）: [{ci[0]:.3f}, {ci[1]:.3f}]")
    if r2 >= 1.0:
        print(f"  ※ 召回率為 1.0，bootstrap 退化；rule of three 漏診率上界約 {3 / n_gt:.3f}")
    print("=" * 60)

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(args.weights)),
                                    f"eval_{args.split}_full.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split, "weights": args.weights, "imgsz": imgsz,
            "mode": "full_scan", "matching": "3D distance (mm) < nodule radius",
            "n_scans": n_scans, "n_patients": n_patients,
            "n_slices": len(img_paths), "n_nodules": n_gt, "n_ignore": n_ign_gt,
            "n_tp": n_tp, "n_fp": n_fp, "n_in_ignore": tot_ign, "n_duplicate": tot_dup,
            "cpm": cpm, "froc": dict(zip(map(str, CPM_POINTS), sens)),
            "recall_at_2fp": r2, "ci_95": ci,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n已存至 {out}")


if __name__ == "__main__":
    main()
