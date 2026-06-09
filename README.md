# AIoT Lab 2 — Smart Classroom Occupancy Detection & Model Deployment

> **Môn học:** Triển khai, phát triển ứng dụng AI và IoT  
> **Dataset:** UCI Occupancy Detection · 20,560 records · Feb 2015  
> **Stack:** Python 3.11 · Jupyter Lab · scikit-learn · FastAPI · uvicorn

---

## Kiến trúc hệ thống

![Kiến trúc AIoT Pipeline](https://github.com/khanhly-dn/AIoT_Lab2/blob/main/MH.png?raw=true)

![Luồng dữ liệu và Deploy](https://github.com/khanhly-dn/AIoT_Lab2/blob/main/MH%20(2).png?raw=true)

---

## Bài toán

Phòng học thông minh gắn cảm biến **nhiệt độ, độ ẩm, ánh sáng, CO₂**. Hệ thống dự đoán trạng thái phòng (*có người / trống*) từ dữ liệu sensor theo thời gian thực, từ đó đưa ra lệnh điều khiển thiết bị tự động — tiết kiệm năng lượng và cải thiện chất lượng không khí.

---

## Pipeline

```
UCI Occupancy Dataset (20,560 records)
    → [1] Kiểm tra schema IoT
    → [2] Làm sạch: timestamp, duplicate, outlier, missing
    → [3] Tạo feature dataset (+ hour, dayofweek, co2_rolling, co2_delta)
    → [4] Chia train/test theo thời gian — tránh data leakage
    → [5] Train Logistic Regression baseline
    → [6] Tính anomaly_score bằng Z-score
    → [7] Sinh decision_log.csv (200 rows)
    → [8] Lưu model .joblib
    → [9] Deploy FastAPI /predict
    → [10] Test API end-to-end ✅
```

---

## Kết quả model — Notebook Output

### Schema & Cleaning Report

```json
{
  "required_columns": ["date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"],
  "missing_columns": [],
  "duplicated_rows": 0,
  "n_rows": 20560,
  "n_columns": 8
}
```

```json
{
  "before_rows": 20560,
  "after_rows": 20560,
  "removed_rows": 0,
  "bad_timestamp_rows": 0,
  "duplicate_rows": 0,
  "outlier_counts": {
    "Temperature": 0, "Humidity": 0, "Light": 0, "CO2": 0, "HumidityRatio": 0
  }
}
```

### Train / Test Split (theo thời gian)

| | Từ | Đến | Số dòng |
|---|---|---|---|
| **Train** | 2015-02-02 14:19 | 2015-02-14 19:38 | 15,420 |
| **Test** | 2015-02-14 19:40 | 2015-02-18 09:19 | 5,140 |

Train Occupancy distribution: `0 → 76.31%` · `1 → 23.69%`

### Feature Dataset (5 dòng đầu)

| timestamp | Temperature | Humidity | Light | CO2 | hour | dayofweek | Occupancy |
|---|---|---|---|---|---|---|---|
| 2015-02-02 14:19 | 23.700 | 26.272 | 585.20 | 749.20 | 14 | 0 | 1 |
| 2015-02-02 14:19 | 23.718 | 26.290 | 578.40 | 760.40 | 14 | 0 | 1 |
| 2015-02-02 14:21 | 23.730 | 26.230 | 572.67 | 769.67 | 14 | 0 | 1 |
| 2015-02-02 14:22 | 23.722 | 26.125 | 493.75 | 774.75 | 14 | 0 | 1 |
| 2015-02-02 14:23 | 23.754 | 26.200 | 488.60 | 779.00 | 14 | 0 | 1 |

### Metrics — `outputs/metrics.json`

| Metric | Score |
|---|---|
| **Accuracy** | **99.44%** |
| Precision | 97.51% |
| **Recall** | **99.91%** |
| F1 Score | 98.69% |
| **ROC-AUC** | **99.88%** |

```json
{
  "accuracy": 0.9943579766536965,
  "precision": 0.9750889679715302,
  "recall": 0.9990884229717412,
  "f1": 0.9869428185502026,
  "roc_auc": 0.9988282300727526,
  "confusion_matrix": [[4015, 28], [1, 1096]]
}
```

**Confusion Matrix:**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 4015 ✅ | 28 ❌ |
| **Actual 1** | 1 ❌ | 1096 ✅ |

### Notebook Checklist Output

```
OK  data\telemetry_clean.csv
OK  data\feature_dataset.csv
OK  models\occupancy_baseline.joblib
OK  outputs\metrics.json
OK  outputs\decision_log.csv
OK  outputs\figures\01_co2_time_series.png
OK  outputs\figures\02_confusion_matrix.png
OK  outputs\figures\03_occupancy_probability.png

NOTEBOOK PASSED: Có thể chuyển sang bước deploy FastAPI.
```

---

## API Test Output — `python src/test_api.py`

```
Checking /health ...
200 {'status': 'ok', 'model_loaded': True, 'model_version': 'lab2-v1'}

Checking /model-info ...
200
{
  "model_name": "occupancy_logistic_regression_baseline",
  "model_version": "lab2-v1",
  "feature_cols": ["Temperature","Humidity","Light","CO2","HumidityRatio","hour","dayofweek"],
  "decision_outputs": [
    "CHECK_SENSOR_OR_DATA",
    "ALERT_VENTILATION_AND_TURN_FAN_ON",
    "ROOM_OCCUPIED_LIGHTING_NEEDED",
    "ROOM_OCCUPIED_KEEP_COMFORT_MODE",
    "ROOM_EMPTY_SAVE_ENERGY"
  ]
}

Checking /predict ...
200
{
  "input": {
    "room_id": "room_101",
    "device_id": "env_node_01",
    "timestamp": "2015-02-06 06:32:00",
    "Temperature": 21.93, "Humidity": 28.786,
    "Light": 469.53, "CO2": 834.85, "HumidityRatio": 0.001824
  },
  "model_output": {
    "occupancy_probability": 0.0945,
    "predicted_occupancy": 0,
    "anomaly_score": 2.8451,
    "is_anomaly": false
  },
  "decision": {
    "decision": "ROOM_EMPTY_SAVE_ENERGY",
    "command_hint": "ac_state=ECO; fan_state=OFF",
    "safety_note": "Phòng có khả năng trống; chỉ chuyển chế độ tiết kiệm nếu không có lịch học."
  }
}

API TEST PASSED: FastAPI model deployment is working.
```

---

## Decision Logic

| Decision | Điều kiện | Lệnh điều khiển |
|---|---|---|
| `ROOM_OCCUPIED_KEEP_COMFORT_MODE` | Có người, sensor bình thường | Giữ chế độ comfort |
| `ROOM_OCCUPIED_LIGHTING_NEEDED` | Có người, ánh sáng thấp | Bật đèn |
| `ALERT_VENTILATION_AND_TURN_FAN_ON` | CO₂ cao bất thường | Bật quạt thông gió |
| `ROOM_EMPTY_SAVE_ENERGY` | Phòng trống | `ac_state=ECO; fan_state=OFF` |
| `CHECK_SENSOR_OR_DATA` | Anomaly score cao | Cảnh báo kiểm tra sensor |

---

## Cấu trúc thư mục

```
AIoT_Lab2/
├── data/
│   ├── DATA_SOURCES.md
│   ├── occupancy_fallback_same_schema.csv
│   ├── telemetry_clean.csv          ← sinh ra sau khi chạy notebook
│   └── feature_dataset.csv          ← sinh ra sau khi chạy notebook
├── notebooks/
│   └── 01_data_prep_baseline_deploy_ready.ipynb
├── src/
│   ├── data_utils.py                ← hàm lõi: clean, train, decision
│   ├── download_data.py             ← tải UCI dataset hoặc dùng fallback
│   ├── run_training_pipeline.py     ← chạy pipeline không cần notebook
│   ├── app.py                       ← FastAPI: /health, /model-info, /predict
│   ├── test_api.py                  ← kiểm tra API end-to-end
│   └── check_outputs.py             ← checklist tự động
├── models/
│   └── occupancy_baseline.joblib    ← model version: lab2-v1
├── outputs/
│   ├── metrics.json
│   ├── decision_log.csv             ← 200 rows
│   └── figures/
│       ├── 01_co2_time_series.png
│       ├── 02_confusion_matrix.png
│       └── 03_occupancy_probability.png
├── docs/
│   ├── sample_payload_predict.json
│   └── teacher_checklist.md
├── requirements.txt
└── README.md
```

---

## Cài môi trường & Chạy

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Chạy notebook

```bash
jupyter lab
# Mở: notebooks/01_data_prep_baseline_deploy_ready.ipynb → Run All Cells
```

### Chạy nhanh qua terminal

```bash
python src/download_data.py
python src/run_training_pipeline.py
python src/check_outputs.py
```

### Deploy & Test API

```bash
# Terminal 1 — khởi động server
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — test end-to-end
python src/test_api.py
```

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Checklist hoàn thành

- [x] Notebook chạy hết không lỗi — `NOTEBOOK PASSED`
- [x] `models/occupancy_baseline.joblib` — version `lab2-v1`
- [x] `outputs/metrics.json` — accuracy 99.44%, ROC-AUC 99.88%
- [x] `outputs/decision_log.csv` — 200 rows đủ cột decision
- [x] FastAPI chạy được, `/docs` truy cập thành công
- [x] `python src/test_api.py` — `API TEST PASSED`

---

## Công nghệ

`Python 3.11` · `pandas` · `scikit-learn` · `joblib` · `FastAPI` · `uvicorn` · `Jupyter Lab` · `matplotlib` · `scipy`

---

## Lộ trình môn học

| Lab | Nội dung |
|---|---|
| Lab 1 | Thiết kế kiến trúc AIoT, use-case, data fields |
| Lab 1.1 | Mô phỏng luồng dữ liệu telemetry |
| **Lab 2** | **← bạn đang ở đây** · Data prep + baseline + FastAPI deploy |
| Lab 3 | Anomaly Detection nâng cao |
| Lab 4 | Forecasting & dự báo xu hướng |
| Lab 5 | Inference Service: versioning, logging, monitoring |
