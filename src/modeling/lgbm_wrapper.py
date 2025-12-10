"""
modeling/lgbm_wrapper.py

Lightweight wrapper around LightGBM providing internal validation splitting
for early stopping, enabling compatibility with sklearn meta-estimators
such as StackingRegressor.

Extended Description
--------------------
This module defines `LGBMRegressorWithEarlyStopping`, an sklearn-compatible
regressor that injects its own train validation split during `.fit()`. This
enables early stopping even when LightGBM is embedded inside models that do
not expose validation sets, such as StackingRegressor or GridSearchCV.

Features include:
- automatic internal validation split using `val_size`
- early stopping via LightGBM callback API
- safe adjustment of `num_leaves` based on `max_depth`
- seamless sklearn drop-in replacement (`fit`, `predict`, `get_params`, `set_params`)
- optional dependency handling (works only when LightGBM is installed)

Main Components
---------------
- LGBMRegressorWithEarlyStopping:
  - fit: train LightGBM with early stopping and internal validation split
  - predict: infer using best_iteration_ when available
  - get_params / set_params: sklearn API compatibility

Usage Example
-------------
>>> from modeling.lgbm_wrapper import LGBMRegressorWithEarlyStopping
>>> model = LGBMRegressorWithEarlyStopping(
...     max_n_estimators=2000,
...     early_stopping_rounds=100,
...     val_size=0.2,
... )
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)

Notes
-----
Early stopping requires that LightGBM is installed. If not available, fitting
will raise an ImportError. Internally, this wrapper is used by the default
model registry inside DefaultModelsMixin.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split as tts

# Optional LightGBM dependency
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


class LGBMRegressorWithEarlyStopping(BaseEstimator, RegressorMixin):
    """
    LightGBM regressor with automatic internal validation split for early stopping.

    Extended Description
    --------------------
    This class wraps `lightgbm.LGBMRegressor` to introduce an internal train
    validation split during `.fit()`. This design allows early stopping to
    work even inside sklearn meta-estimators where validation sets are not
    directly exposed.

    Key responsibilities:
    - splitting data internally using `val_size`
    - fitting LightGBM with an early stopping callback
    - tracking `best_iteration_` for inference
    - exposing sklearn-compatible APIs (`fit`, `predict`, `get_params`, `set_params`)

    Parameters
    ----------
    max_n_estimators : int, default 3000
        Maximum boosting rounds for LightGBM.
    learning_rate : float, default 0.03
        Learning rate for boosting.
    num_leaves : int, default 31
        Maximum leaves for each tree.
    max_depth : int, default 12
        Maximum tree depth.
    min_child_samples : int, default 20
        Minimum samples per leaf.
    min_child_weight : float, default 1e-3
        Minimum sum Hessian in a leaf.
    subsample : float, default 0.8
        Row sampling rate for boosting.
    colsample_bytree : float, default 0.8
        Feature sampling per tree.
    reg_alpha : float, default 0.1
        L1 regularization.
    reg_lambda : float, default 1.0
        L2 regularization.
    min_split_gain : float, default 0.0
        Minimum gain required to split.
    early_stopping_rounds : int, default 200
        Number of rounds without improvement before early stopping.
    val_size : float, default 0.15
        Fraction of data used as validation split.
    random_state : int, default 42
        Random seed for splitting and model reproducibility.

    Attributes
    ----------
    model_ : LGBMRegressor or None
        Underlying LightGBM model after fitting.
    best_iteration_ : int or None
        Best iteration discovered via early stopping.

    Examples
    --------
    >>> model = LGBMRegressorWithEarlyStopping(val_size=0.2)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """

    def __init__(
        self,
        max_n_estimators: int = 3000,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: int = 12,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_split_gain: float = 0.0,
        early_stopping_rounds: int = 200,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        self.max_n_estimators = max_n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.random_state = random_state

        self.model_: Optional[LGBMRegressor] = None
        self.best_iteration_: Optional[int] = None

    def fit(self, X, y):
        if not HAS_LGBM:
            raise ImportError(
                "LightGBM is not installed. Please install lightgbm to use LGBMRegressorWithEarlyStopping."
            )

        X = np.asarray(X)
        y = np.asarray(y)

        # Guard num_leaves with respect to max_depth
        num_leaves = self.num_leaves
        if self.max_depth is not None and self.max_depth > 0:
            num_leaves = min(num_leaves, 2 ** self.max_depth - 1)

        # Internal validation split for early stopping
        X_tr, X_val, y_tr, y_val = tts(
            X, y, test_size=self.val_size, random_state=self.random_state
        )

        self.model_ = LGBMRegressor(
            n_estimators=self.max_n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=int(num_leaves),
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_split_gain=self.min_split_gain,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

        self.model_.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            ],
        )


        # Save best iteration for inference
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model_.predict(
            X,
            num_iteration=getattr(self, "best_iteration_", None),
        )

    def get_params(self, deep: bool = True) -> Dict:
        return {
            "max_n_estimators": self.max_n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_split_gain": self.min_split_gain,
            "early_stopping_rounds": self.early_stopping_rounds,
            "val_size": self.val_size,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


__all__ = ["HAS_LGBM", "LGBMRegressorWithEarlyStopping"]
