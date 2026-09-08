"""
base_loader.py
--------------
Lớp trừu tượng cho việc nạp dữ liệu. Mỗi bộ dữ liệu UCI (Iris, Wine, Dry Bean...)
sẽ có 1 class riêng kế thừa từ BaseDatasetLoader, chỉ cần override _load_raw().

Việc chuẩn hoá dữ liệu (StandardScaler) và trả kết quả (X, y) dưới dạng
numpy array được xử lý chung ở lớp cha -> tránh lặp code.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BaseDatasetLoader(ABC):
    """
    Interface chung: gọi .load() -> trả về (X, y)
        X : ndarray (n_samples, n_features), đã chuẩn hoá (nếu scale=True)
        y : ndarray (n_samples,) nhãn thật (chỉ dùng để đánh giá, KHÔNG dùng
            khi fit FCM vì FCM là thuật toán unsupervised).

    Parameters
    ----------
    scale : bool, default=True
        Có chuẩn hoá dữ liệu (mean=0, std=1) hay không. Với FCM (dựa trên
        khoảng cách Euclid) việc chuẩn hoá gần như luôn cần thiết vì các
        thuộc tính có đơn vị/khoảng giá trị khác nhau.
    """

    def __init__(self, scale: bool = True):
        self.scale = scale
        self.feature_names_: list[str] | None = None
        self.target_names_: list[str] | None = None

    @abstractmethod
    def _load_raw(self) -> tuple[pd.DataFrame, pd.Series]:
        """Trả về (X_df, y_series) chưa chuẩn hoá. Lớp con phải cài đặt."""
        raise NotImplementedError

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        X_df, y = self._load_raw()
        self.feature_names_ = list(X_df.columns)

        X = X_df.values.astype(np.float64)
        if self.scale:
            X = StandardScaler().fit_transform(X)

        return X, np.asarray(y)
