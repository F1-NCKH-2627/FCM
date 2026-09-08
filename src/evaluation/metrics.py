"""
metrics.py
----------
ClusteringEvaluator: tính các chỉ số dùng trong báo cáo thực nghiệm:

1) Chỉ số nội tại: Davies-Bouldin (DB), Xie-Beni (XB), PC và PE.
2) Chỉ số ngoài: NMI, ARI, Purity (PUR), F1 và Accuracy (AC).

Các chỉ số ngoài chỉ dùng để đánh giá sau khi phân cụm, không tham gia huấn luyện.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
)


class ClusteringEvaluator:
    def __init__(self, X: np.ndarray, labels_true: np.ndarray | None = None):
        self.X = X
        self.labels_true = labels_true

    # ---------------- internal ---------------- #
    def internal_metrics(self, labels_pred: np.ndarray) -> dict:
        if len(np.unique(labels_pred)) < 2:
            return {"DB": float("nan")}
        return {"DB": davies_bouldin_score(self.X, labels_pred)}

    # ---------------- external ---------------- #
    @staticmethod
    def _contingency_matrix(labels_true: np.ndarray,
                            labels_pred: np.ndarray) -> tuple:
        """Tạo bảng đếm giữa lớp thật (hàng) và cụm dự đoán (cột)."""
        labels_true = np.asarray(labels_true)
        labels_pred = np.asarray(labels_pred)
        if labels_true.shape != labels_pred.shape:
            raise ValueError("Nhãn thật và nhãn dự đoán phải cùng kích thước")
        if labels_true.size == 0:
            raise ValueError("Không thể tính AC trên tập dữ liệu rỗng")

        true_values, true_ids = np.unique(labels_true, return_inverse=True)
        pred_values, pred_ids = np.unique(labels_pred, return_inverse=True)
        confusion = np.zeros(
            (true_ids.max() + 1, pred_ids.max() + 1), dtype=np.int64
        )
        np.add.at(confusion, (true_ids, pred_ids), 1)
        return confusion, true_values, pred_values

    @classmethod
    def _mapped_labels(cls, labels_true: np.ndarray,
                       labels_pred: np.ndarray) -> np.ndarray:
        """Ghép mỗi nhãn cụm với nhãn thật bằng thuật toán Hungarian."""
        confusion, true_values, pred_values = cls._contingency_matrix(
            labels_true, labels_pred
        )
        true_index, pred_index = linear_sum_assignment(-confusion)
        mapping = {
            pred_values[pred_id]: true_values[true_id]
            for true_id, pred_id in zip(true_index, pred_index)
        }

        # Nếu số cụm lớn hơn số lớp, ghép cụm dư với lớp chiếm đa số của cụm.
        for pred_id, pred_value in enumerate(pred_values):
            if pred_value not in mapping:
                mapping[pred_value] = true_values[np.argmax(confusion[:, pred_id])]

        return np.asarray([mapping[label] for label in labels_pred])

    @classmethod
    def clustering_accuracy(cls, labels_true: np.ndarray,
                            labels_pred: np.ndarray) -> float:
        """AC sau khi ghép tối ưu nhãn cụm với nhãn thật."""
        mapped = cls._mapped_labels(labels_true, labels_pred)
        return float(np.mean(np.asarray(labels_true) == mapped))

    @classmethod
    def purity(cls, labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        """PUR: tỷ lệ phần tử thuộc lớp đa số trong từng cụm."""
        confusion, _, _ = cls._contingency_matrix(labels_true, labels_pred)
        return float(np.sum(np.max(confusion, axis=0)) / len(labels_true))

    def external_metrics(self, labels_pred: np.ndarray) -> dict:
        if self.labels_true is None:
            return {}
        mapped = self._mapped_labels(self.labels_true, labels_pred)
        return {
            "NMI": normalized_mutual_info_score(self.labels_true, labels_pred),
            "ARI": adjusted_rand_score(self.labels_true, labels_pred),
            "PUR": self.purity(self.labels_true, labels_pred),
            "F1": f1_score(self.labels_true, mapped, average="macro"),
            "AC": float(np.mean(np.asarray(self.labels_true) == mapped)),
        }

    # ---------------- fuzzy-specific ---------------- #
    @staticmethod
    def partition_coefficient(u: np.ndarray) -> float:
        """PC = (1/n) * sum_i sum_j u_ij^2, theo công thức (12a)."""
        n = u.shape[0]
        return float(np.sum(u ** 2) / n)

    @staticmethod
    def partition_entropy(u: np.ndarray) -> float:
        """PE = -(1/n) * sum_i sum_j u_ij * ln(u_ij)."""
        n = u.shape[0]
        safe_u = np.where(u > 0.0, u, 1.0)
        return float(-np.sum(u * np.log(safe_u)) / n)

    def xie_beni_index(self, u: np.ndarray, centers: np.ndarray,
                       m: float) -> float:
        """XB: độ nén của cụm chia cho khoảng cách tâm nhỏ nhất; càng nhỏ càng tốt."""
        distances = np.linalg.norm(
            self.X[:, None, :] - centers[None, :, :], axis=2
        )
        numerator = np.sum((u ** m) * (distances ** 2))

        center_distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        ) ** 2
        np.fill_diagonal(center_distances, np.inf)
        min_center_distance = np.min(center_distances)
        denominator = self.X.shape[0] * min_center_distance
        if np.isclose(denominator, 0.0):
            return float("inf")
        return float(numerator / denominator)

    def fuzzy_metrics(self, u: np.ndarray, centers: np.ndarray,
                      m: float) -> dict:
        return {
            "PC": self.partition_coefficient(u),
            "XB": self.xie_beni_index(u, centers, m),
            "PE": self.partition_entropy(u),
        }

    # ---------------- tổng hợp ---------------- #
    def evaluate(self, labels_pred: np.ndarray, u: np.ndarray | None = None,
                 centers: np.ndarray | None = None, m: float = 2.0) -> dict:
        results = {}
        results.update(self.internal_metrics(labels_pred))
        if u is not None:
            if centers is None:
                raise ValueError("Cần truyền centers để tính chỉ số XB")
            results.update(self.fuzzy_metrics(u, centers, m))
        results.update(self.external_metrics(labels_pred))
        return results
