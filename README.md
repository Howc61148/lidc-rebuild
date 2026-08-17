# LIDC-IDRI 肺結節偵測 — 前處理與偵測管線重建

以 LIDC-IDRI 資料集建立肺結節偵測模型，涵蓋前處理、資料切分、YOLO 訓練與病灶層級評估。

本專案為大學畢業專題「肺結節 AI 輔助診斷系統」的**偵測部分重建**。原專題為四人團隊完成的兩階段系統（YOLO 偵測 → 雙輸入 CNN 良惡性分類 → Lung-RADS 分級 → PyQt5 介面），本人負責前處理與部分偵測工作。畢業後重新檢視時，於前處理與評估環節發現數項方法論問題，遂將該部分重做並補齊文件。分類模型與介面維持原狀，不在本專案範圍內。

---

## 結果

於 105 位病人、207 顆結節的驗證集（病灶層級）：

| 指標 | 數值 |
|---|---|
| 召回率 @ 2 FP/scan | **0.884**（95% CI: 0.837–0.935） |
| CPM（LUNA16 七點平均） | **0.827** |

FROC 曲線：

| FP/scan | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| 召回率 | 0.647 | 0.749 | 0.821 | 0.850 | 0.884 | 0.913 | 0.928 |

**比較上的說明**：LUNA16 文獻中的頂尖系統 CPM 約 0.92–0.95，但多為 3D 模型並搭配第二階段假陽性抑制。本專案為 2D 單階段架構，兩者的任務設定不同，數值不宜直接比較。

測試集（104 位病人）保留至模型定案後評估，尚未使用。

---

## 原版本的已知限制

重建的動機來自以下發現。列出以說明各項設計決定的理由。

### 前處理

**共識判斷的層級** — 原流程以「切片」為單位累計標記過該切片的醫師人數。LIDC 由 4 位放射科醫師獨立判讀，同一切片可能存在多顆結節；若 A 醫師標記結節甲、B 醫師標記結節乙，會被誤判為兩人對同一顆有共識，後續的特徵中位數亦會混合不同結節的評分。

**單一結節保留** — 原流程於每張切片僅保留面積最大的輪廓。對偵測任務而言，被捨棄的結節在標準答案中成為背景，等同標註真實存在的病灶為「無」。實測資料中有 309 張切片含多顆結節。

**灰階正規化** — 原流程採逐張切片 min-max 正規化，未套用 RescaleSlope/Intercept 轉換為 HU、亦未做窗寬窗位處理。實測 HU 範圍達 −2048 至 8155（極端值來自掃描視野外區域與金屬植入物），含金屬的切片整張被壓暗，同一結節在相鄰切片呈現不同亮度。此外，推論端使用標準肺窗處理，與訓練端不一致。

### 訓練資料

**標註為固定值** — 供 YOLO 使用的資料集，每張影像的標註皆為 `0 0.5 0.5 0.125 0.125`。該資料集由「以結節為中心裁切」的 256×256 影像構成，故標準答案被裁切幾何決定。此設定下，一個固定輸出中央框的規則即可獲得接近滿分，且訓練集不含任何無結節的影像。

### 評估

**評估層級** — 原評估以切片為單位。結節橫跨多張切片，此方式無法回答臨床關心的「整顆結節是否被發現」。同一模型改以病灶層級評估後，召回率由 0.652 變為 0.884。

**信賴區間** — 原文件記載「Bootstrap 95% CI [1.000, 1.000]」。當 33 個樣本全數正確時，任何重採樣皆不會出現錯誤，該區間為數學上的必然結果，無法反映不確定性。此情況應改用 rule of three（33 樣本全對時，漏診率 95% 上界約 3/33 ≈ 9%）。

---

## 主要設計決定

| 項目 | 選擇 | 理由 |
|---|---|---|
| 結節歸群 | `pylidc.cluster_annotations()` | 依三維位置將各醫師標記歸群為物理結節，共識判斷須在此層級進行 |
| 共識門檻 | ≥3 位醫師 | 與 LUNA16 標準一致，結果可與文獻對照 |
| 低共識結節 | 連同該切片排除 | 僅移除標註而保留影像，將使圖中存在未標註的真實結節 |
| 標註框幾何 | `consensus(clevel=0.5)` | 取至少半數判讀者認定屬於病灶的區域。實測同一結節的直徑在不同醫師間可差 43% |
| 灰階處理 | 固定肺窗 WC −600 / WW 1500 | 與推論端一致。關鍵在於訓練與推論使用同一組參數 |
| 負樣本 | 正樣本 × 1.5，取自同批病人 | 若取自其他病人，模型可能學到掃描儀差異而非病灶特徵 |
| 資料切分 | 病人層級分層切分，assert 驗證 | 同一病人的相鄰切片高度相似，隨機切分會造成資料洩漏 |
| 資料增強 | 依 CT 特性調整 | 灰階影像無色相飽和度資訊；亮度抖動過大會抵銷灰階標準化 |
| 評估 | 3D 聚合 + LUNA16 協定 | 以病灶為單位計分，並產出 FROC 與 CPM |
| 信賴區間 | 病人層級 bootstrap | 同一病人的結節彼此相關，以結節為單位重抽會使區間過窄 |

各決定的完整脈絡與實驗紀錄見 [`docs/WORKLOG.md`](docs/WORKLOG.md)，評估指標的定義與選用理由見 [`docs/METRICS.md`](docs/METRICS.md)。

---

## 環境

前處理與訓練分為兩個環境。pylidc 相依舊版套件，與 ultralytics 的需求衝突。

**前處理**

```bash
conda create -n lidc-prep python=3.10
conda activate lidc-prep
pip install -r requirements-prep.txt
```

**訓練**

```bash
conda create -n lidc-yolo python=3.10
conda activate lidc-yolo
# PyPI 預設安裝 CPU 版 PyTorch，須指定官方來源
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics
```

CUDA 版本依驅動而定，可用 `nvidia-smi` 確認。較舊的驅動改用 `cu126`。

**pylidc 設定**

於使用者家目錄建立 `pylidc.conf`：

```ini
[dicom]
path = D:\LIDC-IDRI\DICOM
```

路徑須指向包含 `LIDC-IDRI-0001` 等資料夾的那一層。Windows 使用者注意：檔案總管預設隱藏副檔名，以記事本存檔容易變成 `pylidc.conf.txt`。

---

## 使用方式

資料需自 [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) 下載，下載時勿取消勾選 annotation。

```bash
# 1. 確認環境與資料就緒
python check_setup.py

# 2. 前處理
python preprocess.py --limit 1018 --out D:/LIDC-IDRI/processed

# 3. 目視驗證
python visualize.py --multi-only

# 4. 病人層級切分
python split_data.py

# 5. 訓練
python train_yolo.py --batch 8 --epochs 60

# 6. 病灶層級評估
python evaluate.py --weights runs/detect/xxx/weights/best.pt --split val
```

---

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `check_setup.py` | 確認套件版本、pylidc 設定與資料可讀性 |
| `preprocess.py` | LIDC 標註 → YOLO 訓練資料（歸群、共識篩選、肺窗、負樣本） |
| `visualize.py` | 將標註框畫回影像，目視確認前處理正確性 |
| `split_data.py` | 病人層級分層切分，產出資料清單與 `split.json` |
| `train_yolo.py` | YOLO 訓練，含 CT 專屬的增強設定與實驗紀錄 |
| `evaluate.py` | 3D 聚合、LUNA16 病灶層級匹配、FROC、CPM、bootstrap CI |
| `docs/WORKLOG.md` | 問題、設計決定與實驗紀錄 |
| `docs/METRICS.md` | 評估指標的定義與選用理由 |

---

## 資料處理規模

| 項目 | 數量 |
|---|---|
| 下載病人數 | 1,018 |
| 產出資料的病人數 | 696 |
| 影像 | 19,714（正樣本 7,888 / 負樣本 11,826） |
| 標註框 | 8,208 |
| 合格結節 | 1,392（共 2,651 顆，1,259 顆未達共識門檻） |
| 含多顆結節的切片 | 309 |
| 切分 | 487 / 105 / 104 位病人 |

---

## 已知限制

- 2D 逐切片偵測，未利用 z 軸的立體資訊
- 未實作第二階段假陽性抑制，此為 LUNA16 文獻的常見架構
- 影像尺寸 640、模型為 YOLO11n，受限於訓練硬體（RTX 3060 Laptop, 6GB）
- 未篩選切片厚度。厚切片的結節出現在較少切片上，且受部分容積效應影響；厚度已記錄於 manifest 供後續分析
- 共識門檻 ≥3 排除了約 48% 的結節，門檻設為 2 的對照實驗尚未進行

---

## 資料來源

Armato SG III, McLennan G, Bidaut L, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2):915–931, 2011.

評估協定參考 LUNA16 challenge（Setio AAA, et al. *Medical Image Analysis*, 42:1–13, 2017）。

## 授權

MIT