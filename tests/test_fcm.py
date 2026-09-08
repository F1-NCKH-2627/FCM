import unittest

import numpy as np

from src.clustering import FuzzyCMeans, FuzzyCMeansPlusPlus
from src.data import IrisLoader
from src.evaluation import ClusteringEvaluator


class FuzzyCMeansTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = IrisLoader(scale=True).load()
        cls.model = FuzzyCMeans(
            n_clusters=3, m=2.0, max_iter=200, tol=1e-5, random_state=42
        ).fit(cls.X)

    def test_memberships_are_valid(self):
        self.assertTrue(np.all(self.model.u_ >= 0.0))
        self.assertTrue(np.all(self.model.u_ <= 1.0))
        self.assertTrue(np.allclose(self.model.u_.sum(axis=1), 1.0))

    def test_objective_does_not_increase(self):
        changes = np.diff(self.model.history_)
        self.assertTrue(np.all(changes <= 1e-9))

    def test_expected_evaluation_metrics(self):
        metrics = ClusteringEvaluator(self.X, self.y).evaluate(
            self.model.labels_,
            u=self.model.u_,
            centers=self.model.cluster_centers_,
            m=self.model.m,
        )
        self.assertEqual(
            list(metrics), ["DB", "PC", "XB", "PE", "NMI", "ARI", "PUR", "F1", "AC"]
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_accuracy_ignores_cluster_number_permutation(self):
        labels_true = np.array([0, 0, 1, 1])
        labels_pred = np.array([1, 1, 0, 0])
        accuracy = ClusteringEvaluator.clustering_accuracy(
            labels_true, labels_pred
        )
        self.assertEqual(accuracy, 1.0)

    def test_plus_plus_reuses_fit_and_predicts_one_sample(self):
        model = FuzzyCMeansPlusPlus(
            n_clusters=3, random_state=42
        ).fit(self.X)
        self.assertEqual(model.predict_proba(self.X[:1]).shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
