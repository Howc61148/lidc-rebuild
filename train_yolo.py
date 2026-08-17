"""
YOLO 結節偵測訓練

用法:
    python train_yolo.py --batch 8
    python train_yolo.py --model yolo11s.pt --imgsz 960 --name exp_960
"""

import argparse
import json
import os
from datetime import datetime

import torch
from ultralytics import YOLO

# ── 資料增強 ──────────────────────────────────────────────
# 預設值為一般彩色照片調校，套用於 CT 需調整：
#   hsv_h/hsv_s = 0   灰階影像無色相與飽和度資訊
#   hsv_v = 0.15      預設 0.4。前處理已統一灰階標準，
#                     過大的亮度抖動會重新引入該變異
#   flipud = 0        上下翻轉的胸腔影像解剖上不存在
#   degrees = 10      病人擺位傾斜是真實存在的變異
#   mosaic            拼接四張影像，理論上破壞 CT 解剖連續性，
#                     但實測於本資料量下開關差異在雜訊範圍內
AUG = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
    degrees=10.0, translate=0.1, scale=0.3, shear=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=10, mixup=0.0,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/LIDC-IDRI/processed/lidc.yaml")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="預訓練權重，n/s/m/l/x 由小到大")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16,
                    help="VRAM 不足時調小（3060 6GB 建議 8）")
    ap.add_argument("--patience", type=int, default=30,
                    help="連續 N 輪無進步即提早停止")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("找不到 GPU。PyPI 預設安裝的是 CPU 版 PyTorch，"
                         "須從 pytorch.org 指定的來源安裝 CUDA 版。")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    name = args.name or f"{args.model.replace('.pt', '')}_{datetime.now():%m%d_%H%M}"

    model = YOLO(args.model)   # 載入 COCO 預訓練權重

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        name=name,
        # 舊版將以下三項設為 False，導致訓練全程無監控、
        # 且只留下最後一輪的權重而非表現最佳者
        val=True,
        save=True,
        plots=True,
        **AUG,
    )

    out_dir = str(model.trainer.save_dir)
    with open(os.path.join(out_dir, "experiment.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "data": args.data,
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "seed": args.seed,
            "augmentation": AUG,
            "gpu": torch.cuda.get_device_name(0),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n完成。輸出於 {out_dir}")
    print(f"權重: {model.trainer.best}")
    print("\n訓練過程顯示的是 validation 的 slice-level 指標。")
    print("病灶層級評估請用 evaluate.py。")


if __name__ == "__main__":
    main()