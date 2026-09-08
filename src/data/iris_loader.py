"""
iris_loader.py
--------------
Bộ Iris (150 mẫu, 4 thuộc tính, 3 lớp) có sẵn trong scikit-learn nên không
cần tải file, không cần internet.
"""

from sklearn.datasets import load_iris
from .base_loader import BaseDatasetLoader


class IrisLoader(BaseDatasetLoader):
    """Nạp bộ dữ liệu Iris (UCI) thông qua sklearn.datasets."""

    def _load_raw(self):
        data = load_iris(as_frame=True)
        X_df = data.data
        y = data.target
        self.target_names_ = list(data.target_names)
        return X_df, y
