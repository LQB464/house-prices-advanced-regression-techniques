# src/modeling/lgbm_wrapper.py

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
    LightGBM wrapper that uses an internal train and validation split
    for early stopping. This allows early stopping even when used
    inside StackingRegressor.
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
