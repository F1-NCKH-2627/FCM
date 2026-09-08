"""
wine_loader.py
--------------
Bộ Wine (178 mẫu, 13 thuộc tính, 3 lớp) có sẵn trong scikit-learn.
"""

from sklearn.datasets import load_wine
from .base_loader import BaseDatasetLoader


class WineLoader(BaseDatasetLoader):
    """Nạp bộ dữ liệu Wine (UCI) thông qua sklearn.datasets."""

    def _load_raw(self):
        data = load_wine(as_frame=True)
        X_df = data.data
        y = data.target
        self.target_names_ = list(data.target_names)
        return X_df, y
