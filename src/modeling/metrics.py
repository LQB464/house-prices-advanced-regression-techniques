"""
modeling/metrics.py

Utility metrics and scorer builders used throughout the training pipeline,
including RMSE helpers and predefined scorer dictionaries for sklearn
cross-validation workflows.

Extended Description
--------------------
This module provides lightweight metric functions and scorer factories that
standardize model evaluation across the pipeline. It includes:
- an RMSE helper function returning float values
- a scorer dictionary compatible with `cross_validate`
- negative RMSE scoring for consistency with sklearn (higher is better)
- R2 score integration via built-in sklearn metrics

These utilities are used by DefaultModelsMixin and other training components
to ensure consistent evaluation and reproducibility.

Main Components
---------------
- _rmse: compute root-mean-square error as a float
- _get_scorers: build sklearn-compatible scoring dictionary for CV

Usage Example
-------------
>>> from modeling.metrics import _rmse, _get_scorers
>>> rmse = _rmse(y_true, y_pred)
>>> scorers = _get_scorers()

Notes
-----
RMSE is returned as a positive float, while cross-validation uses its negative
form (neg_rmse) internally to satisfy sklearn's convention of "higher is better."
"""

import math
from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer


def _rmse(y_true, y_pred) -> float:
    """
    Compute root-mean-square error (RMSE) between predictions and true values.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predictions corresponding to `y_true`.

    Returns
    -------
    float
        The RMSE value computed as sqrt(mean_squared_error).

    Examples
    --------
    >>> _rmse([1, 2, 3], [1.2, 1.8, 3.5])
    0.304...
    """

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _get_scorers() -> Dict[str, object]:
    """
    Build a dictionary of scoring functions for sklearn cross-validation.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        A dictionary with:
        - "neg_rmse": Negative RMSE scorer (higher is better)
        - "r2": Standard R2 scorer

    Examples
    --------
    >>> scorers = _get_scorers()
    >>> sorted(scorers.keys())
    ['neg_rmse', 'r2']
    """
    
    return {
        "neg_rmse": make_scorer(
            lambda yt, yp: -math.sqrt(mean_squared_error(yt, yp)),
            greater_is_better=True,
        ),
        "r2": "r2",
    }


__all__ = ["_rmse", "_get_scorers"]
