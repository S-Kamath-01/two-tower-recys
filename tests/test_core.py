import unittest

import pandas as pd
import torch

from src.data.dataset import TwoTowerDataset
from src.evaluate import hit_rate_at_k, mrr_at_k, recall_at_k
from src.model import TwoTowerModel
from src.train import bpr_loss


class ModelTests(unittest.TestCase):
    def test_model_shapes_and_loss(self):
        model = TwoTowerModel(num_users=3, num_movies=5, embed_dim=8)
        users = torch.tensor([0, 1])
        positives = torch.tensor([1, 2])
        negatives = torch.tensor([3, 4])
        pos_scores, neg_scores = model(users, positives, negatives)
        self.assertEqual(tuple(pos_scores.shape), (2,))
        self.assertTrue(torch.isfinite(bpr_loss(pos_scores, neg_scores)))

    def test_dataset_negative_is_not_positive(self):
        frame = pd.DataFrame({"user_id": [0, 0], "movie_id": [0, 1]})
        dataset = TwoTowerDataset(frame, num_movies=3)
        for _ in range(20):
            self.assertEqual(dataset[0][2], 2)

    def test_saturated_user_fails_instead_of_looping(self):
        frame = pd.DataFrame({"user_id": [0, 0], "movie_id": [0, 1]})
        dataset = TwoTowerDataset(frame, num_movies=2)
        with self.assertRaises(ValueError):
            dataset[0]


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        predictions = {0: [2, 1], 1: [3, 4]}
        positives = {0: {1}, 1: {3}}
        self.assertEqual(recall_at_k(predictions, positives, 1), 0.5)
        self.assertEqual(hit_rate_at_k(predictions, positives, 1), 0.5)
        self.assertEqual(mrr_at_k(predictions, positives, 2), 0.75)


if __name__ == "__main__":
    unittest.main()
