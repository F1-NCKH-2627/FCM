"""
drybean_loader.py
------------------
Bộ Dry Bean Dataset (13.611 mẫu, 16 thuộc tính hình học, 7 lớp) KHÔNG có sẵn
trong sklearn, phải lấy từ UCI Machine Learning Repository (id = 602):
    https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

Class này hỗ trợ 2 cách nạp dữ liệu, ưu tiên theo thứ tự:
    1) Qua thư viện `ucimlrepo`  (pip install ucimlrepo) - cần internet.
    2) Qua file bạn tải thủ công (.xlsx gốc từ UCI, hoặc .csv) - truyền
       đường dẫn vào tham số `local_path`.

Nếu máy chạy code không có internet, hãy:
    - Vào https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
    - Tải file "DryBeanDataset.zip", giải nén lấy file Excel bên trong
    - Truyền đường dẫn file đó vào DryBeanLoader(local_path="duong/dan/Dry_Bean_Dataset.xlsx")
"""

import pandas as pd
from .base_loader import BaseDatasetLoader


class DryBeanLoader(BaseDatasetLoader):
    """Nạp bộ dữ liệu Dry Bean (UCI, id=602)."""

    TARGET_COLUMN = "Class"

    def __init__(self, scale: bool = True, local_path: str | None = None):
        super().__init__(scale=scale)
        self.local_path = local_path

    def _load_raw(self):
        if self.local_path is not None:
            df = self._load_from_file(self.local_path)
        else:
            df = self._load_from_ucimlrepo()

        y_raw = df[self.TARGET_COLUMN]
        X_df = df.drop(columns=[self.TARGET_COLUMN])

        classes = sorted(y_raw.unique())
        self.target_names_ = classes
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(class_to_idx)

        return X_df, y

    # ------------------------------------------------------------------ #
    def _load_from_ucimlrepo(self) -> pd.DataFrame:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "Chưa cài 'ucimlrepo'. Chạy: pip install ucimlrepo\n"
                "Hoặc tải thủ công bộ Dry Bean và truyền local_path=... "
                "vào DryBeanLoader."
            ) from exc

        dry_bean = fetch_ucirepo(id=602)
        X = dry_bean.data.features
        y = dry_bean.data.targets
        df = pd.concat([X, y], axis=1)
        df.columns = list(X.columns) + [self.TARGET_COLUMN]
        return df

    def _load_from_file(self, path: str) -> pd.DataFrame:
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)
