# Fuzzy C-Means trên 3 bộ dữ liệu UCI (Iris, Wine, Dry Bean)

Project cài đặt thuật toán **Fuzzy C-Means (FCM)** theo hướng OOP (Python),
chạy được trên PyCharm, có kiến trúc mở để kế thừa/mở rộng sau này (ví dụ
Possibilistic C-Means, Gustafson-Kessel, hay thêm bộ dữ liệu mới).

## 1. Cấu trúc project

```
fcm_uci_project/
├── requirements.txt
├── main.py                     # Điểm chạy chính
├── outputs/                    # Biểu đồ + bảng kết quả sẽ được lưu ở đây
└── src/
    ├── clustering/
    │   ├── base.py             # BaseClusteringAlgorithm (abstract class)
    │   └── fcm.py              # FuzzyCMeans, FuzzyCMeansPlusPlus (ví dụ kế thừa)
    ├── data/
    │   ├── base_loader.py      # BaseDatasetLoader (abstract class)
    │   ├── iris_loader.py      # IrisLoader
    │   ├── wine_loader.py      # WineLoader
    │   └── drybean_loader.py   # DryBeanLoader
    ├── evaluation/
    │   └── metrics.py          # ClusteringEvaluator (DB, PC, XB, PE, AC...)
    └── visualization/
        └── plots.py            # ClusterVisualizer (PCA scatter, đồ thị hội tụ)
```

## 2. Cài đặt trong PyCharm

1. Mở PyCharm > **File > Open...** > chọn thư mục `fcm_uci_project`.
2. Tạo virtual environment nếu PyCharm hỏi (khuyến nghị Python 3.10+).
3. Mở **Terminal** trong PyCharm (góc dưới), chạy:
   ```bash
   pip install -r requirements.txt
   ```
4. Chuột phải file `main.py` > **Run 'main'** (hoặc Shift+F10).

## 3. Về bộ dữ liệu Dry Bean

Iris và Wine đã có sẵn trong `scikit-learn` nên chạy được ngay, không cần
internet. Riêng **Dry Bean Dataset** (13.611 mẫu, 16 thuộc tính hình học hạt
đậu, 7 lớp) không có sẵn trong sklearn, code sẽ tự động tải qua thư viện
`ucimlrepo` (cần máy có internet, không bị chặn domain
`archive.ics.uci.edu`).

Nếu máy bạn không tải được tự động, tải thủ công:

1. Vào <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
2. Tải `DryBeanDataset.zip`, giải nén lấy file `Dry_Bean_Dataset.xlsx`
3. Trong `main.py`, sửa dòng khởi tạo loader thành:
   ```python
   DryBeanLoader(scale=True, local_path="duong/dan/toi/Dry_Bean_Dataset.xlsx")
   ```

## 4. Cách mở rộng / kế thừa

### a) Thêm biến thể thuật toán mới

Kế thừa `FuzzyCMeans` và chỉ override phần bạn muốn thay đổi — không cần
viết lại toàn bộ vòng lặp fit(). Ví dụ đã có sẵn trong `src/clustering/fcm.py`:

```python
class FuzzyCMeansPlusPlus(FuzzyCMeans):
    # chỉ đổi cách khởi tạo tâm cụm ban đầu (kiểu k-means++)
    ...
```

Thuật toán FCM được cài đặt trực tiếp bằng NumPy theo các công thức cập nhật
tâm cụm và membership; project không gọi một thư viện FCM có sẵn.

Các điểm mở rộng (method) bạn có thể override trong `FuzzyCMeans`:

| Method                | Vai trò                                   |
|------------------------|--------------------------------------------|
| `_init_membership`     | Cách khởi tạo ma trận membership ban đầu   |
| `_initialize_membership` | Chiến lược khởi tạo dựa trên dữ liệu X   |
| `_compute_distances`   | Công thức khoảng cách (đổi sang Mahalanobis, Minkowski...) |
| `_update_centers`      | Công thức cập nhật tâm cụm                 |
| `_update_membership`   | Công thức cập nhật độ thuộc (membership)   |

### b) Thêm bộ dữ liệu mới

Kế thừa `BaseDatasetLoader`, chỉ cần cài đặt `_load_raw()` trả về
`(X_dataframe, y_series)`, việc chuẩn hoá (scale) được xử lý sẵn ở lớp cha:

```python
class MyDatasetLoader(BaseDatasetLoader):
    def _load_raw(self):
        df = pd.read_csv("my_data.csv")
        y = df.pop("label")
        return df, y
```

## 5. Các chỉ số đánh giá (trong `ClusteringEvaluator`)

- **DB (Davies-Bouldin)**: độ tương đồng giữa các cụm; càng nhỏ càng tốt.
- **PC (Partition Coefficient)**: độ rõ của phân hoạch mờ; càng lớn càng tốt.
- **XB (Xie-Beni)**: độ nén trong cụm so với độ tách giữa các tâm; càng nhỏ càng tốt.
- **PE (Partition Entropy)**: độ mờ của phân hoạch; càng nhỏ càng tốt.
- **NMI, ARI, PUR, F1, AC**: so sánh kết quả với nhãn thật, chỉ dùng để báo cáo.

`F1` và `AC` được tính sau khi ghép tối ưu nhãn cụm với nhãn thật bằng thuật
toán Hungarian. Không nên so sánh trực tiếp vì số hiệu cụm của FCM là tùy ý.

## 6. Kết quả

Sau khi chạy `main.py`, bạn sẽ có trong thư mục `outputs/`:
- `{dataset}_clusters.png`: scatter plot 2D (PCA) so sánh nhãn dự đoán vs nhãn thật
- `{dataset}_objective.png`: đồ thị hội tụ của hàm mục tiêu J_m
- `summary_results.csv`: bảng tổng hợp các chỉ số của cả 3 bộ dữ liệu
