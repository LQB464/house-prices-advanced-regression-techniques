"""
modeling/default_models_mixin.py

Default model registry and unified training utilities for benchmarking multiple
regression estimators using a shared preprocessing pipeline.

Extended Description
--------------------
This module defines the `DefaultModelsMixin`, which provides:
- a collection of baseline regression models (linear, ensemble, boosting)
- convenience methods for training a single model or a full baseline suite
- cross-validation with unified scorers (RMSE, R2)
- test-set evaluation with a shared sklearn Pipeline
- model selection utilities based on evaluation metrics

The mixin expects to be combined with TrainerConfig, which provides:
- feature pipeline (`self.feature_pipe_`)
- train test split data (`self.X_train_`, `self.y_train_`)
- storage for results and fitted models (`self.results_`, `self.models_`)
- unified logger and random_state

Main Components
---------------
- get_default_models: return a dictionary of predefined estimators
- train_single_model: train + CV + test evaluation for one model
- train_default_models: train all baseline models
- select_top_models: rank models based on metrics

Usage Example
-------------
>>> class Trainer(TrainerConfig, DefaultModelsMixin):
...     pass
>>> trainer = Trainer()
>>> trainer.train_default_models()
>>> best = trainer.select_top_models(k=3)

Notes
-----
Some models such as XGBoost, CatBoost, and LightGBM are optional and included
only if available in the environment.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR

from .metrics import _rmse, _get_scorers
from .lgbm_wrapper import LGBMRegressorWithEarlyStopping, HAS_LGBM

# Optional dependencies
try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


class DefaultModelsMixin:
    """
    Unified interface for training, evaluating, and ranking baseline models.

    Extended Description
    --------------------
    The `DefaultModelsMixin` manages the full workflow for:
    - retrieving predefined estimators (linear, tree-based, boosting)
    - cross-validating each estimator using the shared preprocessing pipeline
    - fitting the final model on training data
    - evaluating performance on the test set using RMSE and R2
    - recording metrics into `self.results_`
    - registering fitted models into `self.models_`
    - ranking models based on user-selected metrics

    This mixin relies on TrainerConfig to provide:
    `self.feature_pipe_`, dataset splits (`self.X_train_`, `self.y_train_`),
    evaluation storage (`self.results_`, `self.models_`),
    random seed (`self.random_state`), and logger.

    Parameters
    ----------
    None
        The mixin does not introduce its own constructor; it depends on
        attributes supplied by TrainerConfig.

    Attributes
    ----------
    models_ : dict
        Registry mapping model names to fitted sklearn Pipelines.
    results_ : dict
        Evaluation metrics keyed by model name.
    feature_pipe_ : sklearn.Pipeline or None
        Shared preprocessing pipeline applied before all estimators.

    Examples
    --------
    >>> class Trainer(TrainerConfig, DefaultModelsMixin):
    ...     pass
    >>> trainer = Trainer()
    >>> trainer.train_default_models()
    >>> trainer.select_top_models(k=5)
    """

    @staticmethod
    def get_default_models(random_state: int) -> Dict[str, BaseEstimator]:
        models: Dict[str, BaseEstimator] = {
            "random_forest": RandomForestRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            ),
            "elasticnet": ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=50_000,
                random_state=random_state,
            ),
            "ridge": Ridge(alpha=10.0, random_state=random_state),
            "lasso": Lasso(alpha=0.001, max_iter=50_000, random_state=random_state),
            "svr": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        }

        if HAS_XGB:
            models["xgb"] = xgb.XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=random_state,
            )

        if HAS_CATBOOST:
            models["catboost"] = CatBoostRegressor(
                loss_function="RMSE",
                n_estimators=1000,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                subsample=0.8,
                verbose=False,
                random_seed=random_state,
            )

        if HAS_LGBM:
            models["lgbm"] = LGBMRegressorWithEarlyStopping(
                random_state=random_state
            )

        return models

    def train_single_model(
        self,
        name: str,
        estimator: BaseEstimator,
        cv_splits: int = 5,
    ) -> Dict[str, float]:
        if self.feature_pipe_ is None:
            raise RuntimeError("Feature pipeline not built. Call build_preprocessing first.")
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data not available.")

        self.logger.info(f"Training and cross validating model '{name}'")

        pipe = Pipeline(
            [
                ("features", self.feature_pipe_),
                ("model", estimator),
            ]
        )

        scores = cross_validate(
            pipe,
            self.X_train_,
            self.y_train_,
            scoring=_get_scorers(),
            cv=cv_splits,
            n_jobs=-1,
            return_train_score=False,
        )

        cv_rmse_mean = -scores["test_neg_rmse"].mean()
        cv_rmse_std = scores["test_neg_rmse"].std()
        cv_r2_mean = scores["test_r2"].mean()
        cv_r2_std = scores["test_r2"].std()

        self.logger.info(
            f"[{name}] CV RMSE={cv_rmse_mean:.4f}±{cv_rmse_std:.4f} "
            f"CV R2={cv_r2_mean:.4f}±{cv_r2_std:.4f}"
        )

        # Fit on full train
        pipe.fit(self.X_train_, self.y_train_)

        # Evaluate on test
        y_pred = pipe.predict(self.X_test_)
        test_rmse = _rmse(self.y_test_, y_pred)
        test_r2 = float(
            pd.Series(self.y_test_).corr(pd.Series(y_pred)) ** 2
        )  # or use r2_score 

        self.logger.info(
            f"[{name}] Test RMSE={test_rmse:.4f} Test R2={test_r2:.4f}"
        )

        metrics = {
            "cv_rmse_mean": float(cv_rmse_mean),
            "cv_rmse_std": float(cv_rmse_std),
            "cv_r2_mean": float(cv_r2_mean),
            "cv_r2_std": float(cv_r2_std),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
        }

        self.results_[name] = metrics
        self.models_[name] = pipe
        return metrics

    def train_default_models(self, cv_splits: int = 5) -> None:
        """
        Train toàn bộ các default models.
        """
        self.logger.info("Training all default baseline models.")
        base_models = self.get_default_models(random_state=self.random_state)

        for name, est in base_models.items():
            try:
                self.train_single_model(
                    name=name,
                    estimator=est,
                    cv_splits=cv_splits,
                )
            except Exception as e:
                self.logger.exception(
                    f"Failed to train model '{name}' due to error: {e}"
                )

        self.logger.info("Finished training all default models.")

    def select_top_models(self, k: int = 5, by: str = "cv_rmse_mean") -> List[str]:
        if not self.results_:
            raise RuntimeError("No results to select from. Train models first.")

        df = (
            pd.DataFrame(self.results_)
            .T
            .sort_values(by=by, ascending=True)
        )

        top_names = df.head(k).index.tolist()
        self.logger.info(f"Top {k} models by '{by}': {top_names}")
        return top_names
