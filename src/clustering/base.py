"""
base.py
-------
Định nghĩa lớp trừu tượng (abstract base class) cho mọi thuật toán phân cụm
trong project. Mọi thuật toán mới (FCM, Possibilistic C-Means, Gustafson-Kessel,
KMeans wrapper, ...) đều nên kế thừa từ BaseClusteringAlgorithm để đảm bảo
interface thống nhất (fit / predict / fit_predict).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseClusteringAlgorithm(ABC):
    """
    Lớp trừu tượng chung cho mọi thuật toán phân cụm.

    Thuộc tính chung (được set sau khi fit):
        cluster_centers_ : ndarray, shape (n_clusters, n_features)
        labels_          : ndarray, shape (n_samples,)  -> nhãn cứng (hard label)
        n_iter_          : int -> số vòng lặp thực tế đã chạy
    """

    def __init__(self, n_clusters: int, max_iter: int = 150,
                 tol: float = 1e-5, random_state: int | None = None):
        if n_clusters < 2:
            raise ValueError("n_clusters phải >= 2")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # các thuộc tính sẽ được gán sau khi fit
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_iter_: int = 0

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusteringAlgorithm":
        """Huấn luyện thuật toán trên dữ liệu X. Phải trả về self."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gán nhãn cứng cho dữ liệu mới dựa trên tâm cụm đã học."""
        raise NotImplementedError

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Tiện ích: fit rồi trả luôn nhãn cứng của chính X."""
        self.fit(X)
        return self.labels_

    def _check_is_fitted(self):
        if self.cluster_centers_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} chưa được fit. Gọi .fit(X) trước."
            )
