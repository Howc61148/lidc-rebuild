# 本次評估與輸入檢查紀錄

本目錄收錄 2026-09-22 本機執行後提供的兩份 JSON 原檔，文件更新時沒有重算、修飾或改寫其數值。

| 檔案 | 來源與用途 |
|---|---|
| [eval_test_full_scoring_fixed.json](eval_test_full_scoring_fixed.json) | 修補後 `evaluate.py` 的完整掃描 test 彙總結果 |
| [audit_eval_test.json](audit_eval_test.json) | `audit_eval_inputs.py` 的輸入檔案／中繼資料一致性報告 |

## 結果摘要

104 位病人、105 次掃描、26,707 張切片、191 顆正式評估結節；另有 103 顆忽略區結節。

- Recall @ 2 FP/scan：`0.8219895287958116`。
- 95% CI：`[0.7680366347569957, 0.882361156753204]`，為上述 Recall 的病人 bootstrap 區間。
- CPM：`0.7165295437546746`。
- audit：`PASS`，異常及未完成檢查各為 0。

這些是本機評估結果，不是第三方重跑 CT 後的獨立結果；audit 不計算模型效能。

## 執行設定的來源

本機回報的命令為：

```bat
python evaluate.py --weights runs/detect/yolo11s_960/weights/best.pt --split test --out eval_test_full_scoring_fixed.json
```

JSON 記錄權重路徑、`imgsz=960`、資料規模、計數與指標，但未記錄 `conf`。該命令未另外指定 `conf`，對應此次程式預設為 `0.01`；這是依命令與程式對照得到，不是 JSON 原有欄位。Bootstrap 的 1,000 次與 seed 42 亦以目前程式為依據。

兩項邊界檢查及 14 項人工測試通過的結果來自本機終端回報，見 [WORKLOG 第五節](../docs/WORKLOG.md)；兩份 JSON 不是這些人工測試的原始輸出。

## 檔案識別

文件整理時比對，所提供的 `evaluate.py` SHA-256 與 audit 的 `local_evaluate_sha256` 相同：

```text
cb099595c661c8e55672b456e9cc106fee7cc22d231db4091ad2d4e41b216fb3
```

兩份 JSON 的 SHA-256：

```text
eval_test_full_scoring_fixed.json
1ff46fa291dddeb9ce9b6c3b24278891a412c23ca75c36f1a88cce92ac9ef27d

audit_eval_test.json
6b0679372501f2e06a418cbfc79ead7ce4de37b85aef96f4eb59e4443eadb7e1
```

Audit 另保存 `gt_sha256` 與 `split_sha256`，但 `gt.json`、`split.json` 與 CT 影像不隨本次文件包公開。上述程式 hash 並非權重 hash；評估 JSON 本身也未保存程式 hash，與此次執行的對應仍依本機操作紀錄交代。

## 可解讀範圍

目前未收錄原始逐候選分數、配對明細、完整 FROC 門檻列表、權重 SHA-256 或完整環境鎖定資訊。因此可以核對彙總計數及七點平均，不能只用這份 JSON 獨立重算全部推論與 bootstrap，或認定不同機器上會逐位元重現。

Audit 的 `PASS` 只適用於它列出的 PNG、split 與 pylidc 資料庫中繼資料檢查。報告未驗證像素內容、HU 轉換、人工標註正確性、DICOM 方向矩陣或完整三維座標轉換，也不是臨床驗證。

歷史 0.878／CPM 0.844 屬抽樣切片評估，留於 [WORKLOG](../docs/WORKLOG.md)，不覆寫為本次數字。當前階梯式 FROC、病灶配對與指標解讀見 [METRICS](../docs/METRICS.md)。
