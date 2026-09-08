"""
fcm.py
------
Cài đặt thuật toán Fuzzy C-Means (FCM) - phân cụm mờ.

Thiết kế theo OOP với các "extension point" (method có thể override) để
sau này kế thừa dễ dàng, ví dụ:
    - Đổi công thức khoảng cách (Mahalanobis, Minkowski...) -> override _compute_distances
    - Đổi chiến lược khởi tạo (k-means++, ...)               -> override _initialize_membership
    - Đổi công thức update tâm cụm (Gustafson-Kessel, ...)   -> override _update_centers
    - Đổi công thức update membership (Possibilistic C-Means)-> override _update_membership

Công thức chuẩn của FCM (Bezdek, 1981):
    Hàm mục tiêu:   J_m = sum_i sum_j (u_ij)^m * ||x_i - v_j||^2
    Update tâm cụm: v_j = ( sum_i u_ij^m * x_i ) / ( sum_i u_ij^m )
    Update membership:
        u_ij = 1 / sum_k ( d_ij / d_ik ) ^ (2 / (m-1))
    với d_ij = ||x_i - v_j|| (khoảng cách Euclid), m > 1 là hệ số mờ (fuzziness).
"""

from __future__ import annotations
import numpy as np

from .base import BaseClusteringAlgorithm


class FuzzyCMeans(BaseClusteringAlgorithm):
    """
    Fuzzy C-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Số cụm c.
    m : float, default=2.0
        Hệ số mờ (fuzziness exponent), m > 1. m càng lớn thì các cụm càng "mờ"
        (membership càng dàn trải), m -> 1 thì FCM tiệm cận K-Means.
    max_iter : int, default=150
        Số vòng lặp tối đa.
    tol : float, default=1e-5
        Ngưỡng hội tụ, dừng khi phần tử thay đổi lớn nhất của U nhỏ hơn tol.
    random_state : int, optional
        Seed để tái lập kết quả khi khởi tạo ngẫu nhiên membership.

    Attributes
    ----------
    cluster_centers_ : ndarray (n_clusters, n_features)
    labels_          : ndarray (n_samples,)      -> nhãn cứng = argmax membership
    u_               : ndarray (n_samples, n_clusters) -> ma trận membership (độ thuộc mờ)
    history_         : list[float]                -> giá trị hàm mục tiêu J_m qua từng vòng lặp
    n_iter_          : int
    converged_       : bool
    """

    def __init__(self, n_clusters: int = 3, m: float = 2.0,
                 max_iter: int = 150, tol: float = 1e-5,
                 random_state: int | None = None):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter,
                          tol=tol, random_state=random_state)
        if m <= 1:
            raise ValueError("Hệ số mờ m phải > 1")
        if max_iter < 1:
            raise ValueError("max_iter phải >= 1")
        if tol <= 0:
            raise ValueError("tol phải > 0")
        self.m = m
        self.u_: np.ndarray | None = None
        self.history_: list[float] = []
        self.converged_: bool = False

    # ------------------------------------------------------------------ #
    # Các hàm "lõi" - tách riêng để lớp con override khi cần              #
    # ------------------------------------------------------------------ #
    def _init_membership(self, n_samples: int) -> np.ndarray:
        """Khởi tạo ma trận membership ngẫu nhiên, mỗi hàng có tổng = 1."""
        rng = np.random.default_rng(self.random_state)
        u = rng.random((n_samples, self.n_clusters))
        return u / u.sum(axis=1, keepdims=True)

    def _initialize_membership(self, X: np.ndarray) -> np.ndarray:
        """Điểm mở rộng cho các chiến lược khởi tạo khác nhau."""
        return self._init_membership(X.shape[0])

    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Khoảng cách Euclid từ mỗi điểm dữ liệu đến từng tâm cụm.
        Trả về ma trận (n_samples, n_clusters).
        Lớp con có thể override để đổi sang khoảng cách khác."""
        diff = X[:, None, :] - centers[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _update_centers(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """v_j = sum_i(u_ij^m * x_i) / sum_i(u_ij^m)"""
        um = u ** self.m
        weights = um.sum(axis=0)[:, None]
        return (um.T @ X) / weights

    def _update_membership(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """u_ij = 1 / sum_k (d_ij / d_ik)^(2/(m-1))"""
        dist = self._compute_distances(X, centers)
        u_new = np.zeros_like(dist)

        # Nếu một điểm trùng tâm cụm, chia đều membership cho các tâm trùng đó.
        # Xử lý riêng trường hợp này giúp tránh phép chia cho 0 trong công thức.
        zero_distance = np.isclose(dist, 0.0)
        rows_with_zero = zero_distance.any(axis=1)
        if np.any(rows_with_zero):
            zero_rows = zero_distance[rows_with_zero]
            u_new[rows_with_zero] = zero_rows / zero_rows.sum(axis=1, keepdims=True)

        normal_rows = ~rows_with_zero
        if np.any(normal_rows):
            power = 2.0 / (self.m - 1.0)
            inverse_distance = dist[normal_rows] ** (-power)
            u_new[normal_rows] = (
                inverse_distance / inverse_distance.sum(axis=1, keepdims=True)
            )

        return u_new

    def _objective(self, X: np.ndarray, u: np.ndarray, centers: np.ndarray) -> float:
        dist = self._compute_distances(X, centers)
        return float(np.sum((u ** self.m) * (dist ** 2)))

    def _validate_data(self, X: np.ndarray,
                       require_enough_samples: bool = True) -> np.ndarray:
        """Kiểm tra dữ liệu đầu vào và chuyển về mảng số thực hai chiều."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều (n_samples, n_features)")
        if require_enough_samples and X.shape[0] < self.n_clusters:
            raise ValueError("Số mẫu phải lớn hơn hoặc bằng số cụm")
        if X.shape[1] == 0:
            raise ValueError("X phải có ít nhất một thuộc tính")
        if not np.all(np.isfinite(X)):
            raise ValueError("X không được chứa NaN hoặc giá trị vô hạn")
        return X

    # ------------------------------------------------------------------ #
    # Interface chính (theo BaseClusteringAlgorithm)                     #
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        X = self._validate_data(X)
        u = self._initialize_membership(X)
        self.history_ = []
        self.converged_ = False

        for it in range(self.max_iter):
            centers = self._update_centers(X, u)
            u_new = self._update_membership(X, centers)
            self.history_.append(self._objective(X, u_new, centers))

            max_change = float(np.max(np.abs(u_new - u)))
            u = u_new
            self.n_iter_ = it + 1
            if max_change < self.tol:
                self.converged_ = True
                break

        # Đồng bộ tâm cụm với ma trận membership cuối cùng.
        centers = self._update_centers(X, u)
        self.history_[-1] = self._objective(X, u, centers)
        self.u_ = u
        self.cluster_centers_ = centers
        self.labels_ = np.argmax(u, axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Trả về ma trận membership (độ thuộc mờ) cho dữ liệu mới."""
        self._check_is_fitted()
        X = self._validate_data(X, require_enough_samples=False)
        return self._update_membership(X, self.cluster_centers_)


class FuzzyCMeansPlusPlus(FuzzyCMeans):
    """
    Ví dụ minh hoạ CÁCH KẾ THỪA FuzzyCMeans:
    chỉ override lại _initialize_membership để dùng chiến lược khởi tạo kiểu
    "k-means++" (chọn tâm cụm ban đầu cách xa nhau) thay vì random thuần tuý,
    thường giúp thuật toán hội tụ nhanh và ổn định hơn.

    Toàn bộ phần còn lại (update centers, update membership, fit, predict...)
    được kế thừa nguyên vẹn từ lớp cha -> đây chính là lợi ích của thiết kế OOP.
    """

    def _initialize_membership(self, X: np.ndarray) -> np.ndarray:
        """Chọn tâm phân tán kiểu k-means++, sau đó suy ra membership."""
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        centers = [X[rng.integers(n_samples)]]

        for _ in range(1, self.n_clusters):
            squared_distances = np.min(
                [np.sum((X - center) ** 2, axis=1) for center in centers],
                axis=0,
            )
            total = squared_distances.sum()
            if np.isclose(total, 0.0):
                next_idx = rng.integers(n_samples)
            else:
                probabilities = squared_distances / total
                next_idx = rng.choice(n_samples, p=probabilities)
            centers.append(X[next_idx])

        return self._update_membership(X, np.asarray(centers))
