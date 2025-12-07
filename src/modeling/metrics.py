# src/modeling/metrics.py

import math
from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer


def _rmse(y_true, y_pred) -> float:
    """Simple RMSE helper."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _get_scorers() -> Dict[str, object]:
    """
    Build scoring dictionary for cross_validate.

    neg_rmse:
        Negative RMSE so that higher is better. We will flip the sign later.
    r2:
        Standard R2 score.
    """
    return {
        "neg_rmse": make_scorer(
            lambda yt, yp: -math.sqrt(mean_squared_error(yt, yp)),
            greater_is_better=True,
        ),
        "r2": "r2",
    }


__all__ = ["_rmse", "_get_scorers"]
