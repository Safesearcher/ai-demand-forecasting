"""Unit tests for model instantiation and forward-pass inference."""

import unittest
import numpy as np
import torch
from src.models import QuantileMLP, LightGBMQuantileModel, QUANTILES


class TestModelInference(unittest.TestCase):
    def test_quantile_mlp_monotonicity(self):
        # Verify that QuantileMLP forward pass strictly guarantees P10 <= P50 <= P90
        categorical_sizes = {"store": 10, "item": 50}
        numeric_dim = 15
        model = QuantileMLP(
            categorical_sizes=categorical_sizes,
            numeric_dim=numeric_dim,
            hidden_sizes=[64, 32],
            dropout=0.1,
        )
        model.eval()

        batch_size = 16
        cat_inputs = torch.randint(0, 10, (batch_size, 2))
        num_inputs = torch.randn(batch_size, numeric_dim)

        with torch.no_grad():
            preds = model(cat_inputs, num_inputs)

        # Check shapes
        self.assertEqual(preds.shape, (batch_size, 3))

        # Check strict monotonicity: P10 <= P50 <= P90
        p10 = preds[:, 0].numpy()
        p50 = preds[:, 1].numpy()
        p90 = preds[:, 2].numpy()

        self.assertTrue(np.all(p10 <= p50 + 1e-6))
        self.assertTrue(np.all(p50 <= p90 + 1e-6))

    def test_lightgbm_wrapper_sorting(self):
        # Verify that LightGBM wrapper sorts output quantiles monotonically
        model = LightGBMQuantileModel(
            params={"num_leaves": 7, "learning_rate": 0.1, "verbosity": -1},
            num_boost_round=10,
            early_stopping_rounds=5,
        )
        x_dummy = np.random.randn(50, 4)
        y_dummy = np.random.uniform(1.0, 5.0, 50)

        model.fit(x_dummy, y_dummy, x_dummy, y_dummy)
        preds = model.predict(x_dummy[:5])

        self.assertEqual(preds.shape, (5, 3))
        # Verify column ordering
        self.assertTrue(np.all(preds[:, 0] <= preds[:, 1]))
        self.assertTrue(np.all(preds[:, 1] <= preds[:, 2]))


if __name__ == "__main__":
    unittest.main()
