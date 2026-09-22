# LIDC-IDRI 肺結節偵測 — 前處理與偵測管線重建

以 LIDC-IDRI 資料集建立肺結節偵測模型，涵蓋前處理、資料切分、YOLO 訓練與病灶層級評估。

本專案為大學畢業專題「肺結節 AI 輔助診斷系統」的**前處理與偵測部分重建**。原專題由四人團隊完成，整合 YOLO 偵測、雙輸入 CNN 結節分類、圖形化介面與報告輸出；本人主要負責資料前處理與偵測模型的資料建置。畢業後重新檢視時，發現前處理與評估流程有需要修正之處，遂重建相關流程並整理實驗紀錄。分類模型與介面未於此專案重做，也不屬於下列成果的評估範圍。

本專案用於研究與學習展示；下列結果是結節偵測評估，不是良惡性診斷準確率，也不是臨床效能驗證。

---

## 結果

本次定案模型為 YOLO11s、影像尺寸 960，權重為 `runs/detect/yolo11s_960/weights/best.pt`。於本專案 test 集的 104 位病人、105 次**完整 CT 掃描**（26,707 張切片、191 顆評估結節）上進行病灶層級評估。以下為 2026-09-22 完成評分修補後的本機重算結果：

| 指標 | 數值 |
|---|---|
| 召回率 @ 2 FP/scan | **0.822**（95% CI: 0.768–0.882） |
| CPM（七點平均） | **0.717** |

FROC 曲線：

| FP/scan | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| 召回率 | 0.435 | 0.592 | 0.670 | 0.764 | 0.822 | 0.853 | 0.880 |

結果來源：[完整評估 JSON](reports/eval_test_full_scoring_fixed.json)；輸入檢查：[audit 報告](reports/audit_eval_test.json)。原始 JSON 保留未改，紀錄來源與檔案雜湊見 [reports/README.md](reports/README.md)。

**評估修正分為兩個階段。**

1. **評估範圍與參考標準重建**：初版在 test 集中依訓練資料建置方式抽出的切片上推論，未納入該次掃描的其餘切片；不是在 train 集上計算 test 成績。初版另以二維框重新聚合標準答案，人工案例顯示有合併不同結節的風險，且配對使用自訂像素條件。後續改為完整掃描輸入、直接保留標註歸群後的結節身分、毫米距離配對與低共識忽略區。結節數從 189 變為 191 的確切原因，不能只靠總數差推定。
2. **評分邊界修補與重算（2026-09-22）**：相同信心分數的候選一起進入 FROC；正式評估結節的命中及重複命中優先於忽略區。執行兩項邊界檢查、14 項人工測試與輸入一致性檢查後，沿用原權重重新評估。與前次完整掃描紀錄相比，主要指標在原先顯示精度下相同，不代表所有逐候選配對完全相同。

同一權重在不同評估版本下的歷史紀錄：

| | 抽樣切片（初版） | 完整掃描（修正後） |
|---|---|---|
| 切片數 | 2,967 | 26,707 |
| 評估結節 | 189 | 191 |
| 召回率 @ 2 FP/scan | 0.878 | 0.822 |
| CPM | 0.844 | 0.717 |

上表兩次評估使用相同的模型權重，未重新訓練。數值差異反映評估輸入與計分流程的改變，不是更換模型造成的。由於多項評估設定同時修正，未拆分各項修正的影響。

初版程式保留為 `evaluate_v1_sampled.py`，不再用於產生目前成果；修正過程見 [`docs/WORKLOG.md`](docs/WORKLOG.md)。

**文獻參照**：Setio 等人（2017）的 LUNA16 論文中，完整結節偵測賽道的最佳系統 CPM 為 0.811；使用官方提供候選的假陽性抑制賽道，最佳單一系統為 0.908，多系統組合最高達 0.952。這些是該論文報告的歷史結果，不代表目前最佳成績。

本專案 CPM 為 0.717，但資料切分、參考標準與評分實作不同，因此上述數值僅提供文獻背景，不作同條件的效能比較。方法定義見 [`docs/METRICS.md`](docs/METRICS.md)，文獻來源見 [Setio et al., 2017](https://arxiv.org/abs/1612.08012)。

**檢查範圍**：本機回報評分邊界檢查與 14 項人工測試通過；audit 報告為 `PASS`，105 次掃描的 26,707 張 PNG 均成功解碼，異常與未完成檢查各為 0。這只支持列出的函式測試與檔案／資料庫中繼資料一致性，不涵蓋 PNG 像素內容是否正確、HU 轉換、人工標註正確性、完整 DICOM 幾何或臨床效能。

---

## 原版本的已知限制

重建的動機來自以下發現。列出以說明各項設計決定的理由。

### 前處理

**共識判斷的層級** — 原流程以「切片」為單位累計標記過該切片的醫師人數。資料提供多位放射科醫師的標註，同一切片可能存在多顆結節；若 A 醫師標記結節甲、B 醫師標記結節乙，會被誤判為兩人對同一顆有共識，後續的特徵中位數亦會混合不同結節的評分。

**單一結節保留** — 原流程於每張切片僅保留面積最大的輪廓。對偵測任務而言，被捨棄的結節在標準答案中成為背景，等同標註真實存在的病灶為「無」。實測資料中有 309 張切片含多顆結節。

**灰階正規化** — 原流程採逐張切片 min-max 正規化；既有工作紀錄記載影像灰階處理與推論端固定肺窗不一致。重建版統一 HU 轉換與肺窗設定。原紀錄中的 HU 範圍與視覺觀察保留於 WORKLOG，不將極端值的成因當作已逐例驗證的結論。

### 訓練資料

**另一條資料建置路線** — 原 repo 中的 `build_lidc_yolo.py` 以「結節置中裁切」的 256×256 影像為輸入，標註為固定值 `0 0.5 0.5 0.125 0.125`，且不含無結節的影像。此設定下一個固定輸出中央框的規則即可獲得接近滿分。惟依原專題報告，部署的偵測模型係以 Roboflow 平台的真實標註框訓練，該腳本與部署模型的關係無法自公開資料確認。此處列出以說明重建版何以採用完整切片與真實共識框。

### 評估

**評估層級** — 原評估以切片為單位。結節橫跨多張切片，此方式無法回答臨床關心的「整顆結節是否被發現」。重建版改以病灶為單位計分。

**切分方式** — 依既有工作紀錄所引述的原專題報告，切片隨機分配存在同一病人跨組的風險；公開原 repo 又有多條資料流程，不能據此認定所有版本均採相同切分。重建版明確使用病人層級切分，並加入組間重疊檢查。

**信賴區間** — 既有工作紀錄引述「Bootstrap 95% CI [1.000, 1.000]」。對固定且全數正確的二元結果直接重抽，可能得到退化區間，不能據此宣稱沒有不確定性。本次結果使用病人層級 bootstrap；不能將 rule of three 當成所有滿分結果的自動替代方式，適用條件與程式保留提示見 [`docs/METRICS.md`](docs/METRICS.md)。

---

## 主要設計決定

| 項目 | 選擇 | 理由 |
|---|---|---|
| 結節歸群 | `pylidc.cluster_annotations()` | 依三維位置將各醫師標記歸群為物理結節，共識判斷須在此層級進行 |
| 共識門檻 | ≥3 位醫師 | 本專案的標註納入條件；門檻相同不代表參考標準或完整評估協定相同 |
| 低共識結節 | 訓練資料建置時連同涉及切片排除；完整掃描評估時保留影像、設忽略區 | 這是本專案的處理選擇；因此多框保留的範圍是最終保留的訓練切片，不是所有原始切片 |
| 標註框幾何 | `consensus(clevel=0.5)` | 取至少半數判讀者認定屬於病灶的區域。實測同一結節的直徑在不同醫師間可差 43% |
| 灰階處理 | 固定肺窗 WC −600 / WW 1500 | 與推論端一致。關鍵在於訓練與推論使用同一組參數 |
| 負樣本 | 訓練資料按正樣本切片數 × 1.5 抽取，取自同批病人 | 維持資料來源接近並控制背景比例；未以對照實驗確認此比例最優。完整掃描評估不沿用此抽樣 |
| 資料切分 | 病人層級分層切分，assert 驗證 | 同一病人的相鄰切片高度相似，隨機切分會造成資料洩漏 |
| 資料增強 | 依 CT 特性調整 | 灰階影像無色相飽和度資訊；亮度抖動過大會抵銷灰階標準化 |
| 評估 | 完整掃描、結節身分為標準答案、三維 mm 距離匹配 | 參考 LUNA16 的命中規則與 FROC/CPM 形式；低共識結節為忽略區 |
| 信賴區間 | 病人層級 bootstrap，1,000 次，seed 42 | 同一病人的掃描、候選與結節整批重抽，不將每顆結節當作獨立抽樣單位 |

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
```

先依 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/)選擇適合系統與驅動的安裝指令；需要 GPU 時確認所選套件支援 CUDA。不要把「PyPI 一律只提供 CPU 版」當成通則，也不要只憑顯示的 CUDA 數字固定套用另一台電腦的 wheel。

安裝 PyTorch 後，再執行：

```bash
pip install -r requirements-yolo.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

已能完成本次實驗的環境不需為這次文件更新重新安裝。重建環境時須保存實際套件版本；目前結果 JSON 並未包含完整環境鎖定資訊。

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

前處理與完整掃描匯出使用 `lidc-prep`；訓練、評分函式測試及模型評估使用 `lidc-yolo`。以下是流程說明，不是要求已完成評估者全部重跑。

```bat
conda activate lidc-prep
python check_setup.py
python preprocess.py --limit 1018 --out D:/LIDC-IDRI/processed
python visualize.py --multi-only
python split_data.py

conda activate lidc-yolo
python train_yolo.py --model yolo11s.pt --imgsz 960 --batch 16 --epochs 60
```

模型選擇應使用 val 集。`train_yolo.py` 的輸出位置以實際訓練紀錄為準，不能假設每次命名均相同。模型定案後的 test 流程如下：

```bat
conda activate lidc-prep
python export_eval.py --split test
python audit_eval_inputs.py --data D:/LIDC-IDRI/processed --split test

conda activate lidc-yolo
python check_evaluation_edges.py --evaluate evaluate.py
python test_evaluation_scoring.py --evaluate evaluate.py
```

本次固定權重的重新評估指令如下（明列原預設 `conf=0.01` 與原推論尺寸 960；不是新的調參設定）：

```bat
python evaluate.py --weights runs/detect/yolo11s_960/weights/best.pt --data D:/LIDC-IDRI/processed --split test --conf 0.01 --imgsz 960 --out eval_test_full_scoring_fixed.json
```

已有同名結果時先保存紀錄；`evaluate.py --out` 會寫入指定檔案。此次公開報告的副本放在 `reports/`。`best.pt` 指訓練後權重，不是根目錄下載的 `yolo11s.pt`。原始影像、訓練權重與 `runs/` 不納入本程式碼庫，下載程式本身不會取得本次實驗的權重。


---

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `check_setup.py` | 確認套件版本、pylidc 設定與資料可讀性 |
| `preprocess.py` | LIDC 標註 → YOLO 訓練資料（歸群、共識篩選、肺窗、負樣本） |
| `visualize.py` | 將標註框畫回影像，目視確認前處理正確性 |
| `split_data.py` | 病人層級分層切分，產出資料清單與 `split.json` |
| `train_yolo.py` | YOLO 訓練，含 CT 專屬的增強設定與實驗紀錄 |
| `export_eval.py` | 匯出指定 split 的完整掃描與結節層級標準答案 |
| `evaluate.py` | 完整掃描推論、三維距離匹配、FROC、CPM、bootstrap CI |
| `evaluate_v1_sampled.py` | 歷史抽樣切片評估，不作目前成果依據 |
| `check_evaluation_edges.py` | 同分候選、忽略區優先順序與候選計數的人工案例檢查 |
| `test_evaluation_scoring.py` | 評分函式的 14 項人工回歸測試 |
| `audit_eval_inputs.py` | PNG、split 與 pylidc 資料庫中繼資料一致性檢查 |
| `reports/eval_test_full_scoring_fixed.json` | 修補後完整掃描評估的原始彙總結果 |
| `reports/audit_eval_test.json` | 本次輸入檢查原始報告與範圍限制 |
| `reports/README.md` | 紀錄來源、檔案雜湊及可重現性界線 |
| `docs/WORKLOG.md` | 問題、設計決定與實驗紀錄 |
| `docs/METRICS.md` | 評估指標的定義與選用理由 |

---

## 資料處理規模

| 項目 | 數量 |
|---|---|
| 前處理上限（原命令 `--limit` 設定） | 1,018 |
| 產出資料的病人數 | 696 |
| 影像 | 19,714（正樣本 7,888 / 負樣本 11,826） |
| 標註框 | 8,208 |
| 合格結節 | 1,392（共 2,651 顆，1,259 顆未達共識門檻） |
| 含多顆結節的切片 | 309 |
| train / val / test 切分 | 487 / 105 / 104 位病人 |

---

## 已知限制

- 偵測模型採 2D 逐切片輸入；後處理會聚合跨切片候選，但不等於模型已使用 3D 影像特徵。2.5D 或 3D 輸入可列為後續研究方向，效果尚未驗證。
- 本次 `conf=0.01` 及固定後處理下，共有 6,385 個聚合候選：180 個 TP、6,098 個 FP、90 個忽略候選、17 個重複命中。相當於約 60.8 個候選／掃描及 58.1 個 FP／掃描，**不是** 2 FP/scan 操作點的候選統計。
- 90 是忽略候選數，不是 90 顆不同結節；目前未計算 103 顆忽略區結節中被命中的唯一結節數。
- 在這次推論與後處理設定下有 11 顆評估結節未命中；未逐顆分析前，不能斷定模型完全沒有輸出任何相關框，也不能宣稱其他設定均無法偵測。
- 未實作第二階段假陽性抑制；尚未做對照實驗，不能把缺少此階段確定為目前 CPM 的主要原因。
- 定案模型同時調整模型大小（yolo11n → yolo11s）與尺寸（640 → 960），未拆解兩者各自的影響；歷史 val 比較使用舊評估流程，未以目前方法重算。
- 未依切片厚度篩選；厚度記錄於 manifest，薄／厚切片分組表現尚未分析。
- 訓練資料採共識門檻與切片排除條件，只對產出資料的病人切分；完整掃描指既定 test 病人的全部匯出切片，不代表完整 LIDC-IDRI 或外部臨床母群。門檻 ≥2 的對照尚未進行。
- 輸入 audit 不是完整 DICOM 幾何或像素內容驗證，人工評分案例也不涵蓋所有可能的資料情況。
- test 集曾在修正評估程式後重測；依工作紀錄未因此選新模型。後續方法比較應回到 val 集，不能把此批 test 結果當作完全未被檢視的新資料。
- 彙總 JSON 未保存逐候選分數、配對結果、權重 SHA-256 或完整執行環境，因此無法只靠該 JSON 獨立重算全部 FROC 與信賴區間。
---

## 資料來源

Armato SG III, McLennan G, Bidaut L, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2):915–931, 2011.

指標與配對概念參考 [LUNA16 論文](https://arxiv.org/abs/1612.08012)（Setio AAA, et al. *Medical Image Analysis*, 42:1–13, 2017）；本專案實作與官方完整流程不同。

重建、文件整理與程式檢查過程使用 Claude 與 ChatGPT 協助；具體協助、實際執行與方法更正記錄於 WORKLOG，不將工具提出的建議一律描述為獨立發現。

## 授權

本程式碼庫的授權標示為 MIT，以 `LICENSE` 為準；此標示不取代資料集、第三方套件或模型權重各自的授權條件。
