"""
病灶層級評估 — 初版（保留供對照，請勿用於新結果）

本版在訓練用的抽樣切片上推論，FP/scan 的分母不是完整掃描；
標準答案由 2D 框重新聚合，相鄰結節可能被誤併；命中判定以像素計。
這些問題已於 evaluate.py 修正，詳見 docs/WORKLOG.md「評估方法的修正」。

保留原因：README 與 WORKLOG 中「抽樣切片 vs 完整掃描」的對照數字
（0.878 / 0.844）由本版產生。

原用法:
    python evaluate_v1_sampled.py --weights ... --split test
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

CPM_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8]   # LUNA16 定義的七個取樣點
MAX_SLICE_GAP = 2        # 3D 聚合：z 方向容許的最大間隔
MAX_CENTER_DIST = 30     # 3D 聚合：xy 平面容許的最大中心距離（像素）
MIN_HIT_RADIUS = 5.0     # 命中判定的最小半徑，避免極小結節難以命中
SEED = 42


def parse_stem(stem):
    """檔名格式 patient_sSCANID_ZZZZ，回傳 (scan_key, z)。

    以掃描而非病人為 key：同一病人的多次掃描其切片編號各自獨立，
    混在一起會使 3D 聚合誤將不同掃描的框合併。
    """
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].startswith("s"):
        return f"{parts[0]}_{parts[1]}", int(parts[2])
    # 相容舊格式 patient_ZZZZ
    pid, z = stem.rsplit("_", 1)
    return pid, int(z)


def load_ground_truth(data_dir, filenames):
    """讀取標註檔並聚合為 3D 結節，回傳 {scan_key: [nodule, ...]}。"""
    per_scan = defaultdict(list)

    for fn in filenames:
        stem = fn.replace(".png", "")
        key, z = parse_stem(stem)
        lbl = os.path.join(data_dir, "labels", stem + ".txt")
        if not os.path.exists(lbl):
            continue
        with open(lbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = map(float, parts)
                per_scan[key].append((z, cx, cy, w, h))

    return {k: cluster_3d(boxes) for k, boxes in per_scan.items()}


def cluster_3d(boxes):
    """將各切片的 2D 框聚合為 3D 結節。

    依切片順序處理，若某框與既有結節在 z 方向與 xy 平面上皆夠近，
    即併入該結節。缺少此步驟時，一顆橫跨八張切片的結節會被
    計為八個獨立目標。
    """
    boxes = sorted(boxes, key=lambda b: b[0])
    nodules = []

    for z, cx, cy, w, h in boxes:
        placed = False
        for nod in nodules:
            z_last = nod["z_range"][1]
            if z - z_last > MAX_SLICE_GAP:
                continue
            lx, ly = nod["slices"][z_last][:2]
            if np.hypot((cx - lx) * 512, (cy - ly) * 512) <= MAX_CENTER_DIST:
                nod["slices"][z] = (cx, cy, w, h)
                nod["z_range"] = (nod["z_range"][0], max(z_last, z))
                placed = True
                break
        if not placed:
            nodules.append({"slices": {z: (cx, cy, w, h)}, "z_range": (z, z)})

    for nod in nodules:
        zs = sorted(nod["slices"])
        z_mid = zs[len(zs) // 2]
        cx, cy, w, h = nod["slices"][z_mid]
        nod["center"] = (cx * 512, cy * 512, z_mid)
        nod["radius_px"] = max(w, h) * 512 / 2
    return nodules


def match(pred_nodules, gt_nodules):
    """LUNA16 命中規則：偵測中心落在標準答案中心「直徑一半」內即命中。

    一顆標準答案最多對應一個偵測，且由信心較高者優先配對；
    同一結節上的其餘偵測計為誤報。
    """
    used = set()
    is_tp = []
    for p in sorted(pred_nodules, key=lambda n: -n["score"]):
        px, py, pz = p["center"]
        hit = None
        for i, g in enumerate(gt_nodules):
            if i in used:
                continue
            gx, gy, gz = g["center"]
            if abs(pz - gz) > MAX_SLICE_GAP:
                continue
            if np.hypot(px - gx, py - gy) <= max(g["radius_px"], MIN_HIT_RADIUS):
                hit = i
                break
        if hit is not None:
            used.add(hit)
            is_tp.append(True)
        else:
            is_tp.append(False)
    return is_tp, len(used)


def froc(records, n_scans, total_gt):
    """由高至低掃過信心門檻，回傳 [(FP/scan, 召回率, 門檻), ...]。"""
    tp = fp = 0
    curve = []
    for r in sorted(records, key=lambda r: -r["score"]):
        if r["is_tp"]:
            tp += 1
        else:
            fp += 1
        curve.append((fp / n_scans, tp / total_gt, r["score"]))
    return curve


def sens_at(curve, target_fp):
    """指定 FP/scan 下的召回率。"""
    return max((rec for fps, rec, _ in curve if fps <= target_fp), default=0.0)


def cpm_from_curve(curve):
    """CPM：七個指定 FP/scan 點上召回率的平均（LUNA16 標準）。"""
    sens = [sens_at(curve, pt) for pt in CPM_POINTS]
    return float(np.mean(sens)), sens


def bootstrap_ci(per_patient, n_boot=1000, target_fp=2.0, seed=SEED):
    """以病人為單位有放回重抽，估計召回率的 95% 信賴區間。

    不以結節或掃描為單位：同一病人的結節共用掃描儀、重建參數與該病人的
    解剖特徵，彼此相關。當成獨立樣本會高估有效樣本數，使區間過窄。
    """
    rng = np.random.default_rng(seed)
    pids = list(per_patient)
    n_scans_total = sum(d["n_scans"] for d in per_patient.values())
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(pids), size=len(pids), replace=True)
        recs, gt, scans = [], 0, 0
        for i in pick:
            d = per_patient[pids[i]]
            recs += d["records"]
            gt += d["n_gt"]
            scans += d["n_scans"]
        if gt == 0 or scans == 0:
            continue
        vals.append(sens_at(froc(recs, scans, gt), target_fp))
    if not vals:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rule_of_three(n):
    """n 個樣本全對時，失敗率的 95% 上界約為 3/n。

    此情況下 bootstrap 會退化為 [1.0, 1.0]——任何重採樣都不會出錯，
    區間為數學上的必然而非證據強度。
    """
    return 3.0 / n if n > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="D:/LIDC-IDRI/processed")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.01,
                    help="推論門檻須設低，FROC 才能掃完整條曲線")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "test":
        print("※ 正在使用 test 集，此集僅應於模型定案後使用一次。\n")

    with open(os.path.join(args.data, f"{args.split}.txt")) as f:
        img_paths = [ln.strip() for ln in f if ln.strip()]
    filenames = [os.path.basename(p) for p in img_paths]

    print(f"載入標準答案（{len(filenames)} 張影像）...")
    gt = load_ground_truth(args.data, filenames)
    total_gt = sum(len(v) for v in gt.values())
    print(f"  {len(gt)} 次掃描（{len({k.split('_s')[0] for k in gt})} 位病人），"
          f"3D 聚合後共 {total_gt} 顆結節\n")

    print("執行推論...")
    model = YOLO(args.weights)
    raw = defaultdict(list)
    for i in range(0, len(img_paths), 64):
        batch = img_paths[i:i + 64]
        for path, res in zip(batch, model.predict(batch, conf=args.conf,
                                                  verbose=False)):
            stem = os.path.basename(path).replace(".png", "")
            key, z = parse_stem(stem)
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxyn[0].tolist()
                raw[key].append((z, (x1 + x2) / 2, (y1 + y2) / 2,
                                 x2 - x1, y2 - y1, float(b.conf[0])))

    print("3D 聚合並匹配...")
    per_patient, all_records = defaultdict(
        lambda: {"records": [], "n_gt": 0, "n_scans": 0}), []
    for key in gt:
        preds = cluster_3d([r[:5] for r in raw.get(key, [])])

        # 聚合後的結節取其各切片中的最高信心分數
        scores = defaultdict(float)
        for z, cx, cy, w, h, s in raw.get(key, []):
            k = (round(cx, 4), round(cy, 4))
            scores[k] = max(scores[k], s)
        for nod in preds:
            nod["score"] = max(
                scores.get((round(nod["slices"][z][0], 4),
                            round(nod["slices"][z][1], 4)), 0.0)
                for z in nod["slices"])

        is_tp, _ = match(preds, gt[key])
        recs = [{"score": p["score"], "is_tp": t}
                for p, t in zip(sorted(preds, key=lambda n: -n["score"]), is_tp)]

        # 匹配以掃描為單位，但統計以病人為單位彙整：
        # 同一病人的多次掃描不是獨立樣本
        pid = key.split("_s")[0]
        per_patient[pid]["records"] += recs
        per_patient[pid]["n_gt"] += len(gt[key])
        per_patient[pid]["n_scans"] += 1
        all_records += recs

    per_patient = dict(per_patient)
    n_scans = len(gt)
    curve = froc(all_records, n_scans, total_gt)
    cpm, sens_list = cpm_from_curve(curve)
    r2 = sens_at(curve, 2.0)

    print("\n" + "=" * 60)
    print(f"病灶層級評估（{args.split} 集）")
    print("=" * 60)
    print(f"掃描數 {n_scans}   病人數 {len(per_patient)}   "
          f"結節數 {total_gt}   偵測數 {len(all_records)}\n")
    print("FROC — 各假陽性率下的召回率：")
    for pt, s in zip(CPM_POINTS, sens_list):
        print(f"  {pt:>6.3f} FP/scan : {s:.3f}")
    print(f"\nCPM（七點平均）: {cpm:.3f}")
    print(f"\n召回率 @ 2 FP/scan: {r2:.3f}")

    ci = bootstrap_ci(per_patient, target_fp=2.0)
    if ci:
        print(f"  95% CI（病人層級 bootstrap, 1000 次）: [{ci[0]:.3f}, {ci[1]:.3f}]")
    if r2 >= 1.0:
        ub = rule_of_three(total_gt)
        print(f"  ※ 召回率為 1.0，bootstrap 將退化為 [1.0, 1.0]。")
        print(f"    改用 rule of three：漏診率 95% 上界約 {ub:.3f}"
              f"（真實召回率可能低至 {1 - ub:.3f}）")

    print("=" * 60)
    print("\n註：病灶層級（一顆結節任一切片被偵測到即算命中），")
    print("    與訓練過程顯示的 slice-level 指標不可直接比較。")

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(args.weights)),
        f"eval_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split, "weights": args.weights,
            "n_scans": n_scans, "n_patients": len(per_patient),
            "n_nodules": total_gt,
            "cpm": cpm,
            "froc": dict(zip(map(str, CPM_POINTS), sens_list)),
            "recall_at_2fp": r2,
            "ci_95": ci,
        }, f, indent=2)
    print(f"\n已存至 {out}")


if __name__ == "__main__":
    main()
