# Hướng dẫn tái lập – Dự đoán khách hàng đăng ký tiền gửi kỳ hạn (Bank Term Deposit Subscription Prediction)

> Tên tiếng Anh: *Bank Term Deposit Subscription Prediction: A Comparative Study of Five Machine Learning Models on Imbalanced Telemarketing Data*

Toàn bộ mã được chạy trên **Google Colab** (CPU). Tài liệu này giúp người chưa tham gia dự án chạy lại pipeline và thu được kết quả khớp với báo cáo trong sai số nêu ở Mục 9.

## 0. Thông tin chung

- **Đề tài:** Dự đoán khách hàng đăng ký tiền gửi kỳ hạn (ML truyền thống, dữ liệu mất cân bằng)
- **Học phần:** ADY201m | **Học kỳ:** Third | **GVHD:** Le Nhat Tung | **Giảng viên phản biện:** Thu Le
- **Thành viên:**

| Họ tên | MSSV | Vai trò trong pipeline |
| --- | --- | --- |
| Dương Minh Khang (trưởng nhóm) | SE210801 | Thực hiện phần lớn pipeline (dữ liệu, mô hình, đánh giá, báo cáo) |
| Nguyễn Huy Hoàng | SE210917 | Câu lệnh SQL (EDA) và code Python vẽ hình (chia đều cho 3 thành viên) |
| Nguyễn Minh Khôi | SE210717 | Câu lệnh SQL (EDA) và code Python vẽ hình (chia đều cho 3 thành viên) |

- **Repo:** https://github.com/HuyHoang129/ADY201m.git
- **Phiên bản chấm:** tag `v1.0-final` (commit `<điền mã commit sau khi tạo tag>`)
- **Dữ liệu:** `bank-additional-full.csv` nằm sẵn ở thư mục gốc repo (5,8 MB)
- **Checkpoint:** không cung cấp; người chấm huấn luyện lại từ đầu (khoảng 28 phút, xem Mục 3 và 7)
- **Bài báo:** file PDF ở thư mục gốc repo
- **Cập nhật:** 12/11/2026 | **Phiên bản tài liệu:** v1.0

## 1. Quick Start (chạy trên Google Colab)

1. Mở https://colab.research.google.com → **Tệp → Mở sổ tay → GitHub** → dán `HuyHoang129/ADY201m` → chọn `ADY201m.ipynb`.
2. Chọn **Thời gian chạy → Ngắt kết nối và xóa thời gian chạy** (để bắt đầu từ môi trường sạch).
3. Chọn **Thời gian chạy → Chạy tất cả**. Ô đầu tiên của notebook tự clone repo, cài thư viện từ `requirements.txt` và đọc dữ liệu; không cần Google Drive.
4. Chờ khoảng **28 phút** (CPU 2 nhân). Cho phép trình duyệt tải file khi Colab hỏi (ô vẽ Hình 6 gọi `files.download`).
5. Chạy ô kiểm tra ở Mục 8 để đối chiếu kết quả.

Kết quả mong đợi (Bảng 5, dòng LightGBM tuned): ngưỡng ≈ 0,617; PR-AUC ≈ 0,4658; F1 ≈ 0,5071; Recall ≈ 0,5399; Precision ≈ 0,4781.

## 2. Cấu trúc thư mục

```
ADY201m/
├── ADY201m.ipynb                  # toàn bộ pipeline (EDA bằng SQL → Block 1–11)
├── bank-additional-full.csv       # dữ liệu gốc UCI, phân tách bằng dấu ";"
├── requirements.txt               # phiên bản thư viện đã khóa
├── README.md
├── REPRODUCE.md                   # tài liệu này
├── LICENSE                        # MIT
├── Bank_Term_Deposit_..._Data.pdf # bài báo
└── models/                        # TẠO RA khi chạy notebook (không có sẵn trong repo)
```

Pipeline nằm trong một notebook và các ô **phụ thuộc thứ tự chạy** (Block 5 kiểm tra biến của Block 4). Vì vậy luôn dùng "Chạy tất cả" từ môi trường sạch.

| Phần của notebook | Nội dung |
| --- | --- |
| EDA | Đọc dữ liệu, truy vấn SQL (SQLite trong bộ nhớ), biểu đồ khám phá |
| Block 1–2 | Cấu hình, chia 64/16/20, tiền xử lý, pipeline SMOTE-Tomek trong fold |
| Block 3 | Hàm đánh giá, khai báo 5 mô hình |
| Block 4 | Cross-validation 5-fold, Kịch bản A và B |
| Block 5 | `RandomizedSearchCV` cho mô hình được chọn (LightGBM) |
| Block 6 | Chọn ngưỡng trên tập validation, đánh giá một lần trên test |
| Block 7–10 | Top 20 đặc trưng, biểu đồ, phân tích top-k, Hình 6 |
| Block 11 | Lưu kết quả vào `models/` |

## 3. Yêu cầu phần cứng và phần mềm

| Hạng mục | Môi trường nhóm đã chạy | Tối thiểu khuyến nghị |
| --- | --- | --- |
| Nền tảng | Google Colab (miễn phí) | Google Colab |
| CPU | 2 nhân (`os.cpu_count() = 2`) | 2 nhân |
| RAM | 12 GB | 8 GB |
| GPU | Không dùng (runtime CPU) | Không cần |
| Python | 3.13.18 | 3.13.x |
| Dung lượng đĩa | dưới 1 GB cho kết quả | 1 GB |
| Thời gian chạy toàn bộ | 27,9 phút | — |
| Trong đó Block 4 (CV) | khoảng 11 phút | — |
| Trong đó Block 5 (tuning) | khoảng 10 phút | — |

Notebook dùng `google.colab.files`, nên **chỉ chạy trọn vẹn trên Colab**.

## 4. Cài đặt môi trường

Ô đầu của notebook thực hiện việc này; nếu cần làm tay:

```python
!git clone https://github.com/HuyHoang129/ADY201m.git /content/ADY201m
%cd /content/ADY201m
!git checkout v1.0-final
!pip install -q -r requirements.txt
```

Các phiên bản đã khóa trong `requirements.txt` đúng bằng môi trường nhóm đã chạy. Kiểm tra cài đặt:

```python
import sys, sklearn, imblearn, xgboost, lightgbm, pandas, numpy, scipy, joblib, matplotlib, seaborn
print(sys.version)
for m in (sklearn, imblearn, xgboost, lightgbm, pandas, numpy, scipy, joblib, matplotlib, seaborn):
    print(m.__name__, m.__version__)
# Mong đợi: sklearn 1.6.1, imblearn 0.14.2, lightgbm 4.6.0, xgboost 3.4.1,
#           pandas 2.2.3, numpy 2.4.6, scipy 1.16.3, joblib 1.6.0, matplotlib 3.10.0, seaborn 0.13.2
```

Dự án không dùng API key hay biến môi trường.

## 5. Dữ liệu

### 5.1 Nguồn và giấy phép
- UCI Bank Marketing Dataset (bản `bank-additional-full`), Moro, Rita, Cortez (2012), DOI 10.24432/C5K306.
- Bài gốc cần trích dẫn: Moro, Cortez, Rita (2014), *A data-driven approach to predict the success of bank telemarketing*, Decision Support Systems 62, 22–31.
- Giấy phép dữ liệu: CC BY 4.0 (theo trang UCI; vui lòng đối chiếu lại tại nguồn).

### 5.2 Mô tả
41.188 bản ghi, 21 cột (20 đặc trưng + nhãn `y`), phân tách bằng dấu `;`. Tỷ lệ nhãn dương 11,27%. Không có ô trống; giá trị thiếu được mã hóa là `unknown` và giữ như một mức riêng.

### 5.3 Cách lấy dữ liệu và checksum
File có sẵn trong repo. Kiểm tra:

```bash
sha256sum bank-additional-full.csv
# Mong đợi: 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8
```

Nếu muốn dùng file ở vị trí khác, chỉ cần sửa biến `DATA_PATH` ở ô đầu notebook.

### 5.4 Tiền xử lý
Thực hiện trong notebook (Block 2), theo thứ tự:
1. Loại các cột kinh tế vĩ mô dư thừa: giữ `euribor3m`, bỏ `emp.var.rate`, `nr.employed`, `cons.price.idx`, `cons.conf.idx`.
2. Nhóm tuổi thành 4 khoảng; gộp tháng thành 4 quý.
3. Tạo `has_prev_success`, `has_prev_failure`, `previously_contacted`; thay `pdays = 999` bằng `-1`.
4. Mã hóa nhãn `y` thành 0/1.
5. Sau khi chia tập: one-hot các biến phân loại, căn cột tập validation/test theo tập train.
6. Kịch bản B loại hoàn toàn `duration`; Kịch bản A giữ lại.
7. Trong mỗi fold CV: chọn 20 đặc trưng theo mutual information → `StandardScaler` → SMOTETomek → mô hình (tất cả chỉ fit trên phần train của fold).

### 5.5 Chia tập
`train_test_split` phân tầng, `random_state=42`: 20% làm test (8.238 mẫu, 928 mẫu dương), 80% còn lại chia 80:20 thành train (26.360) và validation (6.590). Không lưu file chỉ số; việc chia lại giống hệt nhờ cố định seed và cùng phiên bản scikit-learn.

## 6. Cấu hình và siêu tham số

Siêu tham số được khai báo **trực tiếp trong notebook** (không có file YAML). Cấu hình năm mô hình so sánh có trong Bảng 1 của bài báo.

| Siêu tham số LightGBM (đã tinh chỉnh) | Giá trị | Không gian tìm |
| --- | --- | --- |
| `n_estimators` | 400 | {400, 600, 800, 1000} |
| `max_depth` | 5 | {4, 5, 6, 8} |
| `learning_rate` | 0,01 | {0,01; 0,02; 0,03; 0,05} |
| `num_leaves` | 15 | {15, 31, 50} |
| `subsample` | 0,7 | {0,7; 0,8; 0,9} |
| `colsample_bytree` | 0,8 | {0,7; 0,8; 0,9} |
| `reg_lambda` | 1,0 | {1,0; 2,0; 3,0} |
| `min_child_samples` | 30 | {20, 30, 50} |
| SMOTE `sampling_strategy` | 0,8 | {0,6; 0,7; 0,8} |

- Phương pháp tìm: `RandomizedSearchCV`, 12 cấu hình, 4-fold CV, metric `average_precision`, chỉ dùng tập train (không dùng test).
- Chọn mô hình: trung bình PR-AUC của 5-fold CV ở Kịch bản B.
- Kết quả tìm kiếm lưu ở `models/lightgbm_tuning_results.csv`, tham số tốt nhất ở `models/lightgbm_best_params.json`.

## 7. Huấn luyện

Không có script riêng: huấn luyện nằm trong notebook.

- **Lệnh:** chạy tất cả các ô của `ADY201m.ipynb` (Mục 1).
- **Đầu ra:** thư mục `models/` (xem Mục 10).
- **Thời gian thực tế:** 27,9 phút trên Colab CPU 2 nhân (CV khoảng 11 phút, tuning khoảng 10 phút).
- **Không có checkpoint:** mô hình cuối được huấn luyện lại mỗi lần chạy và lưu thành `models/best_model.joblib` (file này không được commit; `*.joblib` bị loại trừ).
- **Không có cấu hình rút gọn:** toàn bộ pipeline đã dưới 30 phút nên không cần bản rút gọn.

## 8. Đánh giá và suy luận

Đánh giá cuối nằm trong Block 6: chọn ngưỡng trên validation (F1 lớn nhất với Recall ≥ 0,50), rồi đánh giá một lần trên test. Metric dùng `sklearn.metrics`: `average_precision_score` (PR-AUC), `f1_score`, `recall_score`, `precision_score`, `accuracy_score`, `balanced_accuracy_score`.

Ô kiểm tra sau khi chạy xong:

```python
import pandas as pd, json
t = pd.read_csv("models/final_test_B.csv").iloc[0]
exp = {"Threshold":0.617, "PR-AUC (AP)":0.4658, "F1":0.5071, "Recall":0.5399, "Precision":0.4781}
for k,v in exp.items():
    print(f"{k:12s} chạy lại={t[k]:.4f}  báo cáo={v:.4f}  lệch={abs(t[k]-v):.4f}")
print(json.load(open("models/deployment_info.json"))["random_seed"])   # mong đợi: 42
```

Suy luận trên dữ liệu mới: nạp pipeline đã lưu rồi áp ngưỡng trong `models/deployment_info.json`:

```python
import joblib, json
pipe = joblib.load("models/best_model.joblib")
info = json.load(open("models/deployment_info.json"))
# X_new phải có đúng các cột trong info["input_columns"]
proba = pipe.predict_proba(X_new[info["input_columns"]])[:, 1]
pred = (proba >= info["threshold"]).astype(int)
```

## 9. Seed và tính tất định

- `SEED = 42` dùng cho: hai lần `train_test_split`, `StratifiedKFold(shuffle=True)`, tham số `random_state` của các mô hình và SMOTETomek, và `RandomizedSearchCV`.
- Không dùng hàm `set_seed` toàn cục (không dùng deep learning).
- Chỉ chạy **một seed** (42); không có kết quả mean ± std theo nhiều seed.
- Một số bước dùng `n_jobs=-1` (song song).

**Kết quả chạy lại trên Colab sạch (ngày 12/11/2026) so với bài báo:**

| Chỉ số | Chạy lại | Bài báo | Lệch |
| --- | --- | --- | --- |
| Ngưỡng | 0,6177 | 0,617 | 0,0007 |
| PR-AUC | 0,4632 | 0,4658 | 0,0026 |
| F1 | 0,5068 | 0,5071 | 0,0003 |
| Recall | 0,5399 | 0,5399 | 0,0000 |
| Precision | 0,4776 | 0,4781 | 0,0005 |

**Ngưỡng sai số chấp nhận:** kết quả tái lập được coi là khớp nếu mỗi chỉ số trên lệch không quá **0,01** so với bài báo.

## 10. Ánh xạ kết quả ↔ báo cáo

Tất cả đều sinh ra khi chạy "Chạy tất cả" notebook; file nằm trong `models/`.

| Bảng/Hình trong báo cáo | Block | File output |
| --- | --- | --- |
| Bảng 1 – Cấu hình, LightGBM tuned | 5 | `lightgbm_best_params.json`, `lightgbm_tuning_results.csv` |
| Bảng 2 – CV Kịch bản A | 4, 11 | `cv_A.csv` |
| Bảng 4 – CV Kịch bản B | 4, 11 | `cv_B.csv` |
| Hình 2, Hình 3 (so sánh CV) | 4, 8 | `comparison_cv_A_B.csv` |
| Bảng 3, Bảng 5, Bảng 7 (test 5 mô hình, so sánh A/B) | 6, 8 | hiển thị trong notebook (chưa có file CSV riêng) |
| Bảng 5 – dòng LightGBM (tuned) | 6, 11 | `final_test_B.csv`, `model_meta.json` |
| Hình 4 – Đường PR | 8 | `pr_curve.png` |
| Mục 4.3 – 20 đặc trưng được chọn | 7 | `final_20_features_B.csv` |
| Hình 5 – Gain importance | 8 | `figure5b_lightgbm_tuned.png`, `lightgbm_tuned_feature_importance.csv` |
| Bảng 6 – Ranked calling (top-k) | 9, 11 | `topk_results.csv`, `topk_analysis.png` |
| Hình 6 – Cumulative gain | 10 | `cumulative_gain_B.csv`, `cumulative_gain_B.png`, `fig6_ranked_calling.pdf` (tải về qua trình duyệt) |

## 11. Hạn chế và xử lý sự cố

**Phần không tái lập hoàn toàn:**
- Chạy lại cho kết quả lệch nhỏ so với bài báo (Mục 9, tối đa 0,0026 ở PR-AUC). Nguyên nhân chưa được xác minh; có thể do khác phiên bản thư viện so với lần chạy gốc hoặc do chạy song song.
- Colab cập nhật phiên bản Python/thư viện theo thời gian. `requirements.txt` khóa phiên bản của lần chạy ngày 12/11/2026; nếu Colab dùng Python khác 3.13 thì một số phiên bản có thể không cài được.
- Chỉ có một seed, chưa có khoảng tin cậy cho chênh lệch giữa các mô hình; chưa có kiểm định thống kê (bài báo cũng nêu điều này).
- Dữ liệu chia ngẫu nhiên, không chia theo thời gian.

**Lỗi thường gặp:**

| Thông báo lỗi | Nguyên nhân | Cách khắc phục |
| --- | --- | --- |
| `ModuleNotFoundError: google.colab` | Chạy ngoài Colab | Chạy trên Google Colab |
| `NameError: Chưa chạy Block 4 đầy đủ` | Chạy ô không theo thứ tự | Ngắt kết nối và xóa thời gian chạy, rồi "Chạy tất cả" |
| `FileNotFoundError: bank-additional-full.csv` | Không đứng trong thư mục repo | Chạy lại ô đầu (clone repo + `%cd`) hoặc sửa `DATA_PATH` |
| `ValueError: numpy.dtype size changed` | Cài thư viện xong chưa khởi động lại | **Thời gian chạy → Khởi động lại phiên**, rồi chạy lại |
| `files.download` không tải được | Trình duyệt chặn tải xuống | Cho phép tải nhiều file cho trang Colab |
| Kết quả lệch quá 0,01 | Khác phiên bản thư viện | Chạy ô kiểm tra phiên bản ở Mục 4 và đối chiếu |

## 12. Giấy phép, trích dẫn, khai báo AI

- **Giấy phép mã nguồn:** MIT (file `LICENSE`).
- **Dữ liệu:** UCI Bank Marketing, CC BY 4.0 (trích dẫn ở Mục 5.1).
- **Thư viện:** scikit-learn, imbalanced-learn, XGBoost, LightGBM, pandas, NumPy, SciPy, matplotlib, seaborn, joblib (phiên bản ở Mục 4). Tài liệu trích dẫn đầy đủ nằm trong phần tham khảo của bài báo (PDF trong repo).
- **Công cụ AI đã dùng:** Dola, ChatGPT và Gemini hỗ trợ viết code. Nhóm kiểm chứng bằng cách chạy lại toàn bộ notebook từ môi trường Colab sạch và đối chiếu số liệu với bài báo (Mục 9). Tài liệu hướng dẫn tái lập này được soạn với sự hỗ trợ của Claude (Anthropic) và được nhóm kiểm tra theo số liệu thực tế.
