# Các thí nghiệm trong báo cáo checkpoint — luồng dữ liệu từ input đến output

Tài liệu mô tả chính xác từng thí nghiệm **có trong bảng mục 4 của báo cáo
checkpoint**: cái gì đi vào, model khởi tạo thế nào, cái gì đi ra.

Phạm vi: 8 thí nghiệm — E1a, E2, E3, E4, Leak-1, Leak-2, E2-time, E2-domain.
Bảy cái chạy 5 seed (42–46); riêng E3 không có seed vì là phép tính xác định.

> `results/runs.csv` còn chứa một số lần chạy khác ngoài phạm vi tài liệu này.

---

## 1. Luồng chung — mọi thí nghiệm đều đi qua đây

```
 4 file .txt thô
   data/raw/output_{meta,review}_yelp{Hotel,Res}Data_NRYRcleaned.txt
         │
         │  src/load_data.py   ── ghép meta + text theo SỐ DÒNG
         ▼
 reviews.parquet                        67,395 dòng
   row_id · date · review_id · user_id · product_id · label
   star · text · domain · (useful, funny, cool ← không bao giờ dùng)
         │
         ├─► src/features/behavior.py  ──► behavior.parquet     19 cột
         ├─► src/features/text.py      ──► text_stats.parquet   11 cột
         ├─► src/features/text.py      ──► TF-IDF  (fit lúc chạy, không lưu)
         └─► src/graph/build_edges.py  ──► edges.npz   8,743,304 cạnh
         │
         ▼
 splits/*.json      danh sách row_id cho train / val / test
         │
         ▼
 scripts/run_baselines.py     vòng lặp (seed × thí nghiệm)
         │
         ▼
 src/evaluate.py     ngưỡng chọn trên val → chấm trên test
         │
         ▼
 results/runs.csv  ──► scripts/summarize.py ──► results/summary.md
```

**Bốn quy tắc áp dụng cho mọi thí nghiệm**

1. `row_id` (0 … 67,394) là khóa duy nhất, thứ tự dòng không bao giờ đổi.
2. TF-IDF, StandardScaler và ngưỡng **chỉ fit trên `train`**.
3. Ngưỡng chọn trên `val` (quét 91 giá trị 0.05→0.95, lấy macro-F1 cao nhất).
4. Mọi chỉ số báo cáo đo trên `test`, 5 seed (42–46).

**Đặc trưng behavior tính trên toàn bộ dataset**, không chỉ train — vì chúng
chỉ dùng `date/user_id/product_id/star`, không đụng nhãn. Đây là thiết lập
*transductive* đã khai báo. Ngoại lệ: với split `time` thì đây **là** rò rỉ
thời gian (xem §5).

---

## 2. Bốn cách chia dữ liệu

| Split | Quy tắc | train / val / test | Dùng cho |
|---|---|---|---|
| `user_disjoint` | chia theo **người viết**, 60/20/20 | 40,258 / 13,667 / 13,470 | kết quả chính |
| `random` | xáo đều review, 40/20/40 | 26,958 / 13,479 / 26,958 | đo mức rò rỉ |
| `time` | train ≤2010, test ≥2011 | 32,663 / 5,765 / 28,967 | độ bền thời gian |
| `domain` | train nhà hàng → test khách sạn | 52,309 / 9,232 / 4,506 | chuyển lĩnh vực |

---

## 3. Thí nghiệm theo tín hiệu (cùng split `user_disjoint`)

### E1a — Text

```
INPUT    cột `text` thô của các dòng train (40,258 chuỗi)

KHỞI TẠO TfidfVectorizer(word)  ngram=(1,2)  min_df=3  max_features=200k
                                sublinear_tf  strip_accents=unicode
         TfidfVectorizer(char)  analyzer=char_wb  ngram=(3,5)  min_df=5
                                max_features=200k  sublinear_tf
         → hstack hai ma trận thưa
         LogisticRegression(C=4, class_weight="balanced", max_iter=2000)

QUY TRÌNH fit vectorizer + LR trên train
         transform val/test bằng vectorizer ĐÃ fit (không fit lại)

OUTPUT   P(fake) cho từng dòng val và test
KẾT QUẢ  AUC 0.753 ± .004   AP 0.344 ± .005   macro-F1 0.636
```

### E2 — Behavior

```
INPUT    19 cột behavior.parquet
         tầng review  : star, extreme, dev, abs_dev, rank_pct, prod_same_day
         tầng người   : u_n, u_singleton, u_max_per_day, u_pos_ratio,
                        u_neg_ratio, u_mean_abs_dev, u_extreme_ratio,
                        u_active_days, u_mean_star, u_std_star
         tầng quán    : p_mean_star, p_n, p_star_entropy

KHỞI TẠO XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       scale_pos_weight = n_genuine/n_fake ≈ 6.5,
                       eval_metric="aucpr")

OUTPUT   P(fake)
KẾT QUẢ  AUC 0.793 ± .006   AP 0.359 ± .013   macro-F1 0.629
```

> `scale_pos_weight` tính lại theo từng split, vì tỷ lệ fake mỗi phần khác nhau.
>
> Hơn 70% độ quan trọng của model rơi vào nhóm đặc trưng **tầng người viết**;
> riêng `u_active_days` và `u_n` chiếm 31.5%. Nói cách khác, E2 chủ yếu đang
> chấm *người*, không phải *bài review*.

### E3 — Graph *(không có mô hình nào)*

Chạy bằng `scripts/analyze_graph.py`. Không có seed vì phép tính hoàn toàn
xác định.

```
INPUT    edges.npz + nhãn của riêng các dòng train
QUY TRÌNH với mỗi review test: tìm hàng xóm qua 3 quan hệ,
         giữ lại hàng xóm NẰM TRONG TRAIN (tức là biết nhãn),
         điểm = tỷ lệ fake trong số đó
         (không có hàng xóm nào → gán tỷ lệ nền 0.132)
OUTPUT   điểm số trong [0,1]
KẾT QUẢ  AUC 0.629   AP 0.236
         tách riêng: R-U-R 0.500 · R-T-R 0.551 · R-S-R 0.627
```

> R-U-R đạt đúng 0.500 vì split user-disjoint khiến **không một cạnh R-U-R nào**
> bắc qua ranh giới train/test — review test chỉ có hàng xóm cùng người viết
> cũng nằm trong test, mà nhãn của chúng bị giấu.

### E4 — Text + Behavior *(kết quả tốt nhất, hai tầng)*

Thí nghiệm phức tạp nhất — **hai mô hình xếp tầng**.

```
TẦNG TRONG  TF-IDF + LogisticRegression (hệt E1a)  →  sinh 1 số: text_score

   dòng TRAIN → out-of-fold, StratifiedKFold(5, shuffle, seed)
                mỗi fold: fit trên 4 phần kia, chấm phần còn lại
                ↳ LÝ DO: nếu fit trên cả train rồi chấm chính train,
                  text_score đã "nhìn thấy nhãn của chính nó"; cây sẽ
                  tin tuyệt đối vào cột đó và sập khi ra test.

   dòng VAL/TEST → dùng LR fit trên TOÀN BỘ train

TẦNG NGOÀI  XGBClassifier (cùng cấu hình E2)

INPUT    19 behavior  +  11 text stats  +  1 text_score  =  31 cột
OUTPUT   P(fake)
KẾT QUẢ  AUC 0.838 ± .002   AP 0.442 ± .012   macro-F1 0.679
```

**Cache:** điểm TF-IDF lưu ở `data/processed/text_cache/`, khóa theo
(split, seed) + vân tay của file split. Chạy lại mà chỉ đổi cây thì mất ~1
phút thay vì ~30 phút.

---

## 4. Hai thí nghiệm chẩn đoán rò rỉ

Hai cái này **không phải ứng viên mô hình**. Chúng đo xem rò rỉ đáng giá bao nhiêu.

### Leak-1 — thêm 3 cột vote

```
INPUT    19 behavior + useful + funny + cool  =  22 cột
KHỞI TẠO XGBoost, y hệt E2
KẾT QUẢ  AUC 0.891 ± .003   (E2: 0.793   →  +0.098)
Ý NGHĨA  0/8,919 fake có vote, vì review bị lọc bị ẩn nên không ai bấm nút.
         "Có vote" ≈ "không bị lọc" = chính cái nhãn.
```

### Leak-2 — đổi sang split `random`

```
INPUT    19 cột behavior (y hệt E2)
KHÁC     chỉ đổi split: random thay vì user_disjoint
KẾT QUẢ  AUC 0.806 ± .002   (E2: 0.793   →  +0.013)
Ý NGHĨA  43.6% review test có cùng-reviewer nằm trong train; review cùng
         người trùng nhãn 99.7% → model học thuộc người viết.
         Với model behavior ảnh hưởng nhỏ, nhưng với model đồ thị thì
         R-U-R nhảy 0.500 → 0.767.
```

---

## 5. Hai thí nghiệm về độ bền (cùng model E2, đổi split)

### E2-time

```
SPLIT    train ≤ 2010 (32,663)  →  test ≥ 2011 (28,967)
KẾT QUẢ  AUC 0.730 ± .003   AP 0.264
```

> **CẢNH BÁO — rò rỉ thời gian.** Đặc trưng behavior tính trên toàn bộ dataset,
> nên `u_n` và `u_active_days` của một review năm 2010 trong train **đã đếm cả
> hoạt động năm 2012** của người đó. Con số 0.730 là **lạc quan**. Muốn sạch
> phải tính lại đặc trưng chỉ trong cửa sổ train.

### E2-domain

```
SPLIT    train nhà hàng (52,309)  →  test khách sạn (4,506)
         đã loại mọi reviewer khách sạn từng viết review nhà hàng
KẾT QUẢ  AUC 0.593 ± .007   AP 0.209
```

> **CẢNH BÁO — phụ thuộc model.** HistGradientBoosting cho 0.51, XGBoost cho
> 0.59. Luôn ghi kèm tên model khi báo cáo số này.

---

## 6. Chỉ số đầu ra — mỗi dòng `runs.csv` chứa gì

```python
evaluate(y_val, p_val, y_test, p_test, singleton_test)
```

| Cột | Cách tính |
|---|---|
| `auc` | ROC-AUC trên test |
| `ap` | average precision — **chỉ số chính**, vì chỉ 13.2% là dương |
| `macro_f1` | tại ngưỡng chọn trên val |
| `p_at_1pct` | độ chính xác ở 1% điểm cao nhất |
| `auc_singleton` | AUC riêng cho review từ tài khoản 1-review |
| `auc_repeat` | AUC riêng cho review từ reviewer nhiều bài |
| `threshold` | ngưỡng lấy từ val |
| `n_train`, `n_test`, `seed`, `model`, `features`, `split`, `hypothesis` | siêu dữ liệu |

**Không báo cáo accuracy** — đoán tất cả là thật đã được 86.8%.

Hai cột `auc_singleton` / `auc_repeat` là phát hiện quan trọng nhất:

| | 1-review (76% số fake) | Nhiều review |
|---|---|---|
| E1a text | **0.710** | 0.764 |
| E2 behavior | 0.642 | **0.805** |
| E4 gộp | **0.725** | **0.855** |

Text và behavior mạnh ở **hai nhóm ngược nhau**; gộp lại thắng cả hai.

---

## 7. Chạy lại toàn bộ

```bash
python src/download.py && python src/load_data.py && python src/splits.py
python src/features/behavior.py && python src/features/text.py
python src/graph/build_edges.py && python src/eda.py
python scripts/run_baselines.py --seeds 42 43 44 45 46 \
       --only E1a E2-TREE E4 Leak-1 Leak-2 E2-time E2-domain
python scripts/summarize.py && python scripts/analyze_graph.py
python -m pytest tests/ -q          # 13 test
```

Lần chạy nguội đầy đủ khoảng 45 phút trên CPU laptop; phần lớn thời gian là
TF-IDF của E4 (5 fold × 5 seed).

---

## 8. Chưa làm — ngoài phạm vi checkpoint

Chưa có mô hình deep learning nào trong repo: `torch`, `dgl`,
`torch_geometric`, `transformers` đều **chưa cài**. `src/models/` chỉ có
`baselines.py`.

Đồ thị đã **dựng xong** (`edges.npz`) và đã **phân tích**, nhưng chưa có mô
hình nào học trên đó — E3 chỉ là phép chia, không có tham số. GNN đề xuất,
RoBERTa và các phương pháp link analysis thuộc tuần 11–16.
