"""
環境與資料就緒檢查

執行 preprocess.py 之前先跑這支，確認：
  1. 套件版本正確（pylidc 相依舊版 numpy 與 SQLAlchemy）
  2. pylidc.conf 設定正確，能找到 DICOM 影像
  3. 影像的 HU 值合理（確認 RescaleSlope/Intercept 有被套用）
  4. consensus() 可正常運作

用法:
    python check_setup.py
"""

import os
import sys

MIN_READERS = 3


def check_packages():
    """確認套件可載入並印出版本。"""
    print("── 套件 ──")
    ok = True
    try:
        import numpy as np
        print(f"  numpy      {np.__version__}")
        if tuple(map(int, np.__version__.split(".")[:2])) >= (1, 24):
            print("    ⚠ pylidc 需要 numpy<1.24（1.24 移除了 np.int）")
            ok = False
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        ok = False

    for mod, name in [("cv2", "opencv"), ("pandas", "pandas"),
                      ("pydicom", "pydicom"), ("skimage", "scikit-image")]:
        try:
            m = __import__(mod)
            print(f"  {name:10} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  ✗ {name} 未安裝")
            ok = False

    try:
        import pylidc  # noqa: F401
        print("  pylidc     已安裝")
    except ImportError as e:
        print(f"  ✗ pylidc: {e}")
        ok = False

    return ok


def check_config():
    """確認 pylidc.conf 存在且內容正確。"""
    print("\n── 設定檔 ──")
    conf = os.path.join(os.path.expanduser("~"), "pylidc.conf")
    print(f"  位置: {conf}")

    if not os.path.exists(conf):
        print("  ✗ 不存在")
        print("    Windows 提示：檔案總管預設隱藏副檔名，")
        print("    用記事本存檔容易變成 pylidc.conf.txt")
        return False

    with open(conf, encoding="utf-8", errors="replace") as f:
        content = f.read().strip()
    print("  內容:")
    for line in content.splitlines():
        print(f"    {line}")
    return True


def check_data():
    """確認影像可載入，且 HU 值範圍合理。"""
    import numpy as np
    import pylidc as pl
    from pylidc.utils import consensus

    print("\n── 資料 ──")

    # pylidc 內建全部 1018 位病人的標註資料庫，此數字與實際下載量無關
    scans = pl.query(pl.Scan).all()
    print(f"  標註資料庫收錄 {len(scans)} 位病人")

    available = []
    for scan in scans:
        try:
            path = scan.get_path_to_dicom_files()
        except Exception:
            continue
        if os.path.isdir(path) and any(f.endswith(".dcm") for f in os.listdir(path)):
            available.append(scan)

    print(f"  影像可用     {len(available)} 位")
    if not available:
        print("  ✗ 找不到任何影像。常見原因：")
        print("    - pylidc.conf 的 path 未指向含 LIDC-IDRI-0001 的那一層")
        print("    - 下載尚未完成")
        return False

    th = np.array([s.slice_thickness for s in available], dtype=float)
    print(f"  切片厚度     {th.min():.2f} / {np.median(th):.2f} / {th.max():.2f} mm"
          f"（最小 / 中位 / 最大）")
    print(f"    ≤2.5mm: {(th <= 2.5).sum()} 位")

    scan = available[0]
    print(f"\n  載入 {scan.patient_id} 測試...")
    vol = scan.to_volume(verbose=False)
    print(f"    尺寸 {vol.shape}  數值範圍 {vol.min()} ~ {vol.max()}")
    # 正常 HU：空氣約 -1000、骨頭 +1000 以上。若落在 0-255 或 0-4095，
    # 表示 RescaleSlope/Intercept 未被套用
    if vol.min() < -900 and vol.max() > 500:
        print("    ✓ HU 值範圍正常")
    else:
        print("    ⚠ 不像 HU 值，確認 RescaleSlope/Intercept 是否套用")

    print("\n  測試 consensus()...")
    for s in available[:10]:
        for anns in s.cluster_annotations(verbose=False):
            if len(anns) < MIN_READERS:
                continue
            cmask, cbbox, _ = consensus(anns, clevel=0.5)
            print(f"    {s.patient_id}: {len(anns)} 位醫師標記，"
                  f"共識遮罩 {cmask.shape}")
            print("    ✓ consensus() 正常")
            return True
    print("    前 10 位病人中無達到共識門檻的結節，非錯誤")
    return True


def main():
    ok = check_packages()
    if not ok:
        print("\n套件有問題，先修正後再繼續。")
        sys.exit(1)

    if not check_config():
        print("\n設定檔有問題，修正後再繼續。")
        sys.exit(1)

    if check_data():
        print("\n環境就緒，可執行 preprocess.py")


if __name__ == "__main__":
    main()