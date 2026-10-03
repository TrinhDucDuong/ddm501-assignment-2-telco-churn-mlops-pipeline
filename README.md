# ddm501-assignment-2-telco-churn-mlops-pipeline

**Trịnh Đức Dương - 25ms13290 | DDM501**

Bài 2 kế thừa bài 1 về thiết kế và baseline nhưng **là project chạy độc lập**, có dữ liệu, mã nguồn, dependency và môi trường riêng. Không cần đặt cạnh bài 1.

## Nội dung nộp

- `DDM501_Assignment2_25ms13290_TrinhDucDuong.pdf`: báo cáo theo đủ 5 nhóm tiêu chí.
- `report/`: nội dung chỉnh sửa được, sơ đồ, biểu đồ và script dựng lại PDF.
- `churn/mlops.py`: pipeline 7 stage, 10 thí nghiệm, MLflow và quality gate.
- `dags/churn_training.py`: DAG Airflow thực thi cùng các hàm stage.
- `evidence/`: test, HTTP, kết quả thí nghiệm, registry và log Airflow thực tế.

## Chạy độc lập trên Windows

Yêu cầu Python 3.10 hoặc 3.11. Tại **ddm501-assignment-2-telco-churn-mlops-pipeline**:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
./.venv/Scripts/python.exe -m churn
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe scripts/verify.py
```

Các lệnh sử dụng `.venv` **của chính bài 2**. Pipeline tạo `artifacts/experiments.csv`, `summary.json`, `mlflow.db`, `model.joblib`, `champion.json`, `scored_customers.csv`. Chạy lại tạo run thí nghiệm mới. API phục vụ artifact mới sau khi khởi động lại.

Xem MLflow trên máy:

```powershell
./.venv/Scripts/python.exe -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5052
```

Mở http://127.0.0.1:5052 để xem các run và model `ddm501-telco-churn`, alias `candidate`. Dừng bằng Ctrl+C. Store tạo trên máy này có artifact URI tuyệt đối. Khi clone, runtime artifacts đã được bỏ qua bởi Git nên chạy pipeline sẽ tạo store mới. Khi **di chuyển cả folder có artifacts cũ**, đặt `$env:CHURN_OUTPUT_DIR='artifacts-fresh'` trước khi chạy pipeline; API khi đó cần `$env:CHURN_MODEL_PATH='artifacts-fresh/model.joblib'` và MLflow UI dùng `sqlite:///artifacts-fresh/mlflow.db`. Không tái sử dụng database runtime từ đường dẫn cũ. Pipeline phát hiện tình huống này và báo lỗi kèm hướng dẫn, không ghi vào thư mục cũ.

API:

```powershell
./.venv/Scripts/python.exe -m uvicorn churn.api:create_app --factory --host 127.0.0.1 --port 8012
```

Mở http://127.0.0.1:8012/docs. Dùng payload `examples/request.json`. Chạy train trước khi khởi động API.

## Chạy Airflow thật bằng Docker

Airflow chạy trên Linux container; không cài Airflow vào Python Windows. Yêu cầu Docker Desktop đang chạy Linux containers:

```powershell
docker compose build airflow-test
docker compose run --rm airflow-test
```

Lệnh này migrate metadata DB rồi thực thi `airflow dags test telco_churn_training 2026-10-03`. Đây là thực thi DAG thật với đủ ingest → validate → prepare → train → evaluate → deploy → score. Kết quả được bind mount ra `artifacts/docker/airflow_runs/`. Mỗi DAG run có thư mục/registry riêng. `dags test` không phải hệ thống scheduler/HA chạy liên tục. Log thực thi bàn giao nằm tại `evidence/airflow.txt`.

Để chạy pipeline thuần container (không Airflow):

```bash
docker build -t ddm501-assignment2 .
docker run --rm -v "${PWD}/artifacts/container:/app/artifacts" ddm501-assignment2
```

Linux/macOS có thể chạy trực tiếp bằng `python3 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, `.venv/bin/python -m churn`.

## Cấu hình và tái lập

`config.yaml` khai báo seed, capacity, 10 cấu hình và ngưỡng gate. Các biến môi trường: `CHURN_DATA_PATH`, `CHURN_OUTPUT_DIR`, `CHURN_MODEL_PATH`, `MLFLOW_TRACKING_URI`; Airflow thêm `CHURN_CONFIG`. Không lưu secret trong YAML.

Các stage chạy riêng khi cần khôi phục: `python -m churn --stage validate` (chỉ chạy khi đầu vào stage đã tồn tại). Lựa chọn mô hình dùng AP **validation**, không chọn theo test. Chi phí liên hệ giới hạn top 20%; ngưỡng 0.5 chỉ phục vụ metric chẩn đoán. Không suy luận hiệu quả chiến dịch từ nhãn churn.

Mỗi lần chạy có hash dữ liệu/mã nguồn, split IDs, version registry và môi trường. Dữ liệu là mẫu IBM giả lập, không có temporal holdout thực tế. Các chỉ tiêu SLA và lợi ích kinh doanh trong báo cáo là mục tiêu thiết kế, không phải kết quả vận hành đã đo.

## Git repo riêng

Dựng lại báo cáo sau khi đã chạy pipeline:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-report.txt
./.venv/Scripts/python.exe report/build_report.py
```

Font Unicode được đóng gói trong project. Khi chạy lại Airflow, có thể cập nhật bằng chứng cho báo cáo bằng cách chép `artifacts/docker/airflow-verification.json` thành `evidence/airflow-summary.json` trước khi dựng PDF.

Project không phụ thuộc bài 1 hoặc file ở thư mục cha. `.gitignore` loại môi trường/cache/database/artifacts runtime; source, PDF, report source, config, public data và `evidence/` được giữ. `requirements-lock-windows.txt` ghi môi trường đã kiểm tra; không sử dụng lock Windows để cài container Linux. Sau khi di chuyển/clone, tạo `.venv` mới bằng `setup.ps1` và chạy lại pipeline.

Repository: https://github.com/TrinhDucDuong/ddm501-assignment-2-telco-churn-mlops-pipeline

Project dùng Git repository riêng, nhánh `main`, remote `origin` trỏ tới địa chỉ trên.

`setup.ps1` uses `requirements-lock-windows.txt` when present to restore the tested full environment; otherwise it uses `requirements.txt`.
