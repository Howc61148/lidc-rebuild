"""
病人層級資料切分

依病人而非影像切分。同一病人的相鄰切片極為相似，若隨機分配，
模型在訓練時等同看過測試集的近乎相同影像（資料洩漏）。

用法:
    python split_data.py
    python split_data.py --ratios 0.7 0.15 0.15 --seed 42
"""

import argparse
import json
import os
import random

import pandas as pd

SEED = 42


def stratified_patient_split(pat_stats, ratios, rng):
    """依標註框數量分層，再切成三組。

    先按框數排序，逐一分配給「目前最缺人」的組別，使各組的
    多結節與少結節病人比例接近，避免測試集全是簡單案例。
    """
    ordered = sorted(pat_stats.items(), key=lambda kv: -kv[1]["n_boxes"])

    names = ["train", "val", "test"]
    splits = {k: [] for k in names}
    targets = dict(zip(names, ratios))
    counts = {k: 0 for k in names}
    total = 0

    for pid, _ in ordered:
        total += 1
        deficits = {k: targets[k] - (counts[k] / total) for k in names}
        best = max(deficits.values())
        # 並列時隨機決定，避免固定偏向某組
        cands = [k for k in names if abs(deficits[k] - best) < 1e-9]
        pick = rng.choice(cands)
        splits[pick].append(pid)
        counts[pick] += 1

    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/LIDC-IDRI/processed")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    help="train val test 比例")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"比例加總必須為 1，目前是 {sum(args.ratios)}")

    rng = random.Random(args.seed)
    mf = pd.read_csv(os.path.join(args.data, "manifest.csv"))

    pat_stats = {}
    for pid, g in mf.groupby("patient_id"):
        pat_stats[pid] = {
            "n_images": len(g),
            "n_pos": int(g.is_positive.sum()),
            "n_boxes": int(g.n_boxes.sum()),
            "slice_thickness": float(g.slice_thickness.iloc[0]),
        }

    splits = stratified_patient_split(pat_stats, args.ratios, rng)

    # 資料洩漏不會產生錯誤訊息，只會讓結果失真。
    # 以 assert 將其轉為會中斷執行的錯誤。
    s_tr, s_va, s_te = (set(splits[k]) for k in ["train", "val", "test"])
    assert not (s_tr & s_va), "train 與 val 有重疊病人"
    assert not (s_tr & s_te), "train 與 test 有重疊病人"
    assert not (s_va & s_te), "val 與 test 有重疊病人"
    assert len(s_tr | s_va | s_te) == len(pat_stats), "有病人未被分配"

    # ultralytics 支援以 txt 清單指定資料集，毋須複製影像檔
    img_dir = os.path.join(args.data, "images").replace("\\", "/")
    for name, pids in splits.items():
        files = sorted(mf[mf.patient_id.isin(set(pids))].filename.tolist())
        with open(os.path.join(args.data, f"{name}.txt"), "w") as f:
            for fn in files:
                f.write(f"{img_dir}/{fn}\n")

    # 評估時須沿用同一份切分，否則數字不可比
    with open(os.path.join(args.data, "split.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in splits.items()}, f, indent=2)

    with open(os.path.join(args.data, "lidc.yaml"), "w") as f:
        f.write(f"path: {args.data}\n")
        f.write("train: train.txt\nval: val.txt\ntest: test.txt\n")
        f.write("\nnames:\n  0: nodule\n")

    print("=" * 66)
    print(f"{'':6} {'病人':>5} {'影像':>7} {'正樣本':>7} {'負樣本':>7} {'框數':>7} {'厚度中位':>8}")
    for name in ["train", "val", "test"]:
        pids = splits[name]
        sub = mf[mf.patient_id.isin(set(pids))]
        th = sorted(pat_stats[p]["slice_thickness"] for p in pids)
        med = th[len(th) // 2] if th else 0
        print(f"{name:6} {len(pids):>5} {len(sub):>7} "
              f"{int(sub.is_positive.sum()):>7} "
              f"{int((1 - sub.is_positive).sum()):>7} "
              f"{int(sub.n_boxes.sum()):>7} {med:>8.2f}")
    print("=" * 66)
    print("✓ 三組病人無重疊")
    print(f"\n輸出:")
    print(f"  {args.data}/split.json   切分結果")
    print(f"  {args.data}/lidc.yaml    YOLO 設定檔")
    print(f"  {args.data}/{{train,val,test}}.txt")


if __name__ == "__main__":
    main()