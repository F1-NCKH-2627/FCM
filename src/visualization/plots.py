"""
plots.py
--------
ClusterVisualizer: vẽ 2 loại biểu đồ hay dùng khi báo cáo kết quả FCM:

    1) plot_clusters_2d : giảm chiều dữ liệu về 2D bằng PCA rồi vẽ scatter
       (so sánh nhãn dự đoán vs nhãn thật, nếu có).
    2) plot_objective    : vẽ đường cong hội tụ của hàm mục tiêu J_m theo
       từng vòng lặp (history_ lấy từ FuzzyCMeans).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # để chạy được cả khi không có màn hình (headless)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class ClusterVisualizer:
    def __init__(self, X: np.ndarray):
        self.X = X

    def plot_clusters_2d(self, labels_pred: np.ndarray, labels_true: np.ndarray | None = None,
                          title: str = "FCM clustering", save_path: str | None = None):
        X2 = PCA(n_components=2, random_state=0).fit_transform(self.X) \
            if self.X.shape[1] > 2 else self.X

        n_plots = 2 if labels_true is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        axes[0].scatter(X2[:, 0], X2[:, 1], c=labels_pred, cmap="tab10", s=15)
        axes[0].set_title(f"{title} - Nhãn dự đoán (FCM)")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")

        if labels_true is not None:
            axes[1].scatter(X2[:, 0], X2[:, 1], c=labels_true, cmap="tab10", s=15)
            axes[1].set_title(f"{title} - Nhãn thật")
            axes[1].set_xlabel("PC1")
            axes[1].set_ylabel("PC2")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    def plot_objective(self, history: list, title: str = "Hội tụ hàm mục tiêu J_m",
                        save_path: str | None = None):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(history) + 1), history, marker="o", ms=3)
        ax.set_xlabel("Vòng lặp")
        ax.set_ylabel("J_m")
        ax.set_title(title)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
