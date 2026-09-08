"""
main.py
-------
Chạy thuật toán Fuzzy C-Means (FCM) trên 3 bộ dữ liệu UCI: Iris, Wine, Dry Bean.

Cách chạy trong PyCharm:
    1. Mở project này bằng PyCharm (File > Open... chọn thư mục fcm_uci_project)
    2. Cài thư viện: mở Terminal trong PyCharm rồi chạy
           pip install -r requirements.txt
    3. Chuột phải vào main.py > Run 'main'

Nếu bộ Dry Bean không tải được qua internet, xem hướng dẫn trong
src/data/drybean_loader.py để nạp bằng file tải thủ công.
"""

import os
import numpy as np
import pandas as pd

from src.clustering import FuzzyCMeans
from src.data import IrisLoader, WineLoader, DryBeanLoader
from src.evaluation import ClusteringEvaluator
from src.visualization import ClusterVisualizer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_fcm_on_dataset(name: str, loader, n_clusters: int, m: float = 2.0,
                        random_state: int = 42):
    print(f"\n{'=' * 60}\nBỘ DỮ LIỆU: {name}\n{'=' * 60}")

    X, y_true = loader.load()
    print(f"Kích thước dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} thuộc tính, "
          f"{len(np.unique(y_true))} lớp thật")

    # --- Huấn luyện FCM ---
    fcm = FuzzyCMeans(n_clusters=n_clusters, m=m, max_iter=200,
                       tol=1e-5, random_state=random_state)
    fcm.fit(X)
    if fcm.converged_:
        print(f"Hội tụ sau {fcm.n_iter_} vòng lặp.")
    else:
        print(f"Dừng ở giới hạn {fcm.n_iter_} vòng lặp, chưa đạt ngưỡng hội tụ.")

    # --- Đánh giá ---
    evaluator = ClusteringEvaluator(X, labels_true=y_true)
    results = evaluator.evaluate(
        fcm.labels_, u=fcm.u_, centers=fcm.cluster_centers_, m=fcm.m
    )

    print("Kết quả đánh giá:")
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")

    # --- Vẽ biểu đồ ---
    viz = ClusterVisualizer(X)
    scatter_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_clusters.png")
    obj_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_objective.png")
    viz.plot_clusters_2d(fcm.labels_, y_true, title=name, save_path=scatter_path)
    viz.plot_objective(fcm.history_, title=f"{name} - J_m qua các vòng lặp",
                        save_path=obj_path)
    print(f"Đã lưu biểu đồ: {scatter_path}\n              {obj_path}")

    return {"dataset": name, **results, "n_iter": fcm.n_iter_}


def main():
    all_results = []

    # 1) Iris: 3 loài hoa
    all_results.append(
        run_fcm_on_dataset("Iris", IrisLoader(scale=True), n_clusters=3)
    )

    # 2) Wine: 3 giống rượu
    all_results.append(
        run_fcm_on_dataset("Wine", WineLoader(scale=True), n_clusters=3)
    )

    # 3) Dry Bean: 7 loại hạt đậu
    #    Nếu không có internet, thay bằng:
    #    DryBeanLoader(scale=True, local_path="duong/dan/Dry_Bean_Dataset.xlsx")
    try:
        all_results.append(
            run_fcm_on_dataset("DryBean", DryBeanLoader(scale=True), n_clusters=7)
        )
    except Exception as exc:
        print(f"\n[!] Bỏ qua Dry Bean vì lỗi khi nạp dữ liệu: {exc}")

    # --- Tổng hợp kết quả ---
    df_results = pd.DataFrame(all_results)
    summary_path = os.path.join(OUTPUT_DIR, "summary_results.csv")
    df_results.to_csv(summary_path, index=False)
    print(f"\n{'=' * 60}\nTỔNG HỢP KẾT QUẢ\n{'=' * 60}")
    print(df_results.to_string(index=False))
    print(f"\nĐã lưu bảng tổng hợp: {summary_path}")


if __name__ == "__main__":
    main()
