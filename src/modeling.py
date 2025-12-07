"""
modeling.py

High level model training, model selection and stacking for Ames House Prices.

Responsibilities:
- Load data and split into train and test
- Build feature preprocessing pipeline
- Train default baseline models (including XGB, CatBoost, LightGBM if available)
- Evaluate with cross validation and hold out test
- Select top K models
- Hyperparameter tuning for top models with Optuna (including boosted models)
- Tune 3 model stacking ensemble with Optuna (choosing best 3 tuned models)
- Save numeric results and comparison plots
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import itertools
import logging
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import (
    train_test_split,
    train_test_split as tts,
    cross_validate,
    GridSearchCV,
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    make_scorer,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR

# Optional dependencies
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

from preprocessing import Preprocessor, build_feature_pipeline


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


# LightGBM wrapper with internal early stopping
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

        self.model_ = None
        self.best_iteration_ = None

    def fit(self, X, y):
        if not HAS_LGBM:
            raise RuntimeError("lightgbm is not installed.")

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
            min_data_in_bin=3,
            max_bin=255,
            feature_pre_filter=False,
            force_col_wise=True,
            boosting_type="gbdt",
            objective="regression",
            n_jobs=-1,
            random_state=self.random_state,
        )

        try:
            self.model_.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        except Exception as e:
            print(f"[LGBM wrapper] Early stopping failed: {e}. Training without validation.")
            # Fallback: train on full data with fewer trees
            self.model_.set_params(
                n_estimators=max(500, self.max_n_estimators // 4)
            )
            self.model_.fit(X, y)
            self.best_iteration_ = None

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


class ModelTrainer:
    """
    High level manager for the full regression model workflow.

    Typical usage:
        trainer = ModelTrainer(...)
        trainer.load_data("train.csv")
        trainer.split_data()
        trainer.build_preprocessing()
        trainer.run_full_model_selection_and_stacking()

    The full pipeline does:
        1. Train all default models (including XGB, CatBoost, LGBM if installed)
        2. Select top K models by CV RMSE
        3. Hyperparameter tuning for these top models with Optuna
        4. Tune 3 model stacking ensemble with Optuna over tuned models
        5. Evaluate best stack on the test set
    """

    def __init__(
        self,
        target_col: str = "SalePrice",
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "model_outputs",
        log_level: int = logging.INFO,
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger(log_level=log_level)

        # Data containers
        self.df_: Optional[pd.DataFrame] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None

        # Preprocessing pipeline and model registry
        self.feature_pipe_: Optional[Pipeline] = None
        self.models_: Dict[str, Pipeline] = {}
        self.results_: Dict[str, Dict] = {}

        # Delegate low level preprocessing to Preprocessor
        self.dp = Preprocessor(target_col=self.target_col)

    # -------------------------------------------------------------------------
    # Logger
    # -------------------------------------------------------------------------
    def _build_logger(self, log_level: int) -> logging.Logger:
        """
        Build a logger that writes to output_dir/training.log.

        Avoids adding duplicate handlers if multiple ModelTrainer
        instances are created in the same process.
        """
        logger = logging.getLogger("ModelTrainer")

        if logger.handlers:
            logger.setLevel(log_level)
            return logger

        logger.setLevel(log_level)

        log_path = self.output_dir / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    # -------------------------------------------------------------------------
    # Default models
    # -------------------------------------------------------------------------
    @staticmethod
    def get_default_models(random_state: int) -> Dict[str, object]:
        """
        Return a dictionary of baseline models to be evaluated.

        Names are kept short and consistent. These names will be used as keys
        in self.models_ and self.results_.
        """
        models: Dict[str, object] = {
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
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=20000,
                random_state=random_state,
            ),
            "ridge": Ridge(alpha=10.0, random_state=random_state),
            "lasso": Lasso(
                alpha=0.0005,
                max_iter=20000,
                random_state=random_state,
            ),
            "svr": SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.1,
                gamma="scale",
            ),
        }

        if HAS_XGB:
            models["xgb"] = xgb.XGBRegressor(
                n_estimators=4000,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2.0,
                reg_lambda=3.0,
                reg_alpha=0.2,
                gamma=0.05,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                max_bin=256,
                missing=np.nan,
            )

        if HAS_CAT:
            models["catboost"] = CatBoostRegressor(
                loss_function="RMSE",
                n_estimators=3000,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                subsample=0.8,
                colsample_bylevel=0.8,
                random_state=random_state,
                verbose=False,
            )

        if HAS_LGBM:
            models["lgbm"] = LGBMRegressorWithEarlyStopping(
                max_n_estimators=3000,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=12,
                min_child_samples=20,
                min_child_weight=1e-3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_split_gain=0.0,
                early_stopping_rounds=200,
                val_size=0.15,
                random_state=random_state,
            )

        return models

    # -------------------------------------------------------------------------
    # Data handling
    # -------------------------------------------------------------------------
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load raw data from CSV.

        Uses Preprocessor.load_data so that any consistent cleaning happens in
        one place.
        """
        self.logger.info(f"Loading data from {csv_path}")
        self.df_ = self.dp.load_data(csv_path)
        self.logger.info(f"Data loaded with shape {self.df_.shape}")
        return self.df_

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split loaded dataframe into train and test subsets.

        Returns X_train, X_test, y_train, y_test.
        """
        if self.df_ is None:
            raise RuntimeError("No data loaded. Call load_data first.")

        X, y = self.dp.split_features_target(self.df_)

        (
            self.X_train_,
            self.X_test_,
            self.y_train_,
            self.y_test_,
        ) = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.logger.info(
            "Split data into train and test: "
            f"train={self.X_train_.shape[0]} rows, "
            f"test={self.X_test_.shape[0]} rows"
        )
        return self.X_train_, self.X_test_, self.y_train_, self.y_test_

    def build_preprocessing(self) -> Pipeline:
        """
        Build feature preprocessing pipeline using training data.
        """
        if self.X_train_ is None or self.X_test_ is None:
            raise RuntimeError("Call split_data before build_preprocessing.")

        self.feature_pipe_ = build_feature_pipeline(
            self.X_train_,
            self.X_test_,
        )
        self.logger.info("Feature preprocessing pipeline built successfully.")
        return self.feature_pipe_

    # -------------------------------------------------------------------------
    # Training and evaluation
    # -------------------------------------------------------------------------
    def train_single_model(
        self,
        name: str,
        estimator,
        cv_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Train one model and evaluate it with CV and test metrics.

        Steps:
        - Build pipeline features -> estimator
        - Cross validate on X_train, y_train
        - Fit on full train
        - Evaluate on X_test, y_test
        """
        if self.feature_pipe_ is None:
            raise RuntimeError("Call build_preprocessing before training models.")

        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data not available.")

        pipe = Pipeline(
            [
                ("features", self.feature_pipe_),
                ("model", estimator),
            ]
        )
        self.logger.info(f"Training and cross validating model '{name}'")

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

        # Fit once on full train and evaluate on test
        pipe.fit(self.X_train_, self.y_train_)
        self.models_[name] = pipe

        if self.X_test_ is None or self.y_test_ is None:
            raise RuntimeError("Test data not available.")

        y_pred = pipe.predict(self.X_test_)
        test_rmse = _rmse(self.y_test_, y_pred)
        test_r2 = r2_score(self.y_test_, y_pred)

        metrics = {
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }

        self.results_[name] = metrics
        self.logger.info(
            f"[{name}] CV RMSE={cv_rmse_mean:.4f} "
            f"CV R2={cv_r2_mean:.4f} "
            f"| Test RMSE={test_rmse:.4f} Test R2={test_r2:.4f}"
        )
        return metrics

    def train_default_models(
        self,
        cv_splits: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all default baseline models.

        Returns a mapping from model name to metric dictionary.
        """
        self.logger.info("Training all default baseline models.")
        candidates = self.get_default_models(self.random_state)

        for name, est in candidates.items():
            self.train_single_model(name, est, cv_splits=cv_splits)

        self.logger.info("Finished training default models.")
        return self.results_

    # -------------------------------------------------------------------------
    # Model selection and tuning for single models
    # -------------------------------------------------------------------------
    def select_top_models(
        self,
        k: int = 5,
        by: str = "cv_rmse_mean",
    ) -> List[str]:
        """
        Select top K models according to a metric in self.results_.

        Default metric is cross validated RMSE (lower is better).
        """
        if not self.results_:
            raise RuntimeError("No results available. Train models first.")

        for name, metrics in self.results_.items():
            if by not in metrics:
                raise KeyError(
                    f"Metric '{by}' is missing in results for model '{name}'"
                )

        sorted_items = sorted(
            self.results_.items(),
            key=lambda kv: kv[1][by],
        )
        top_names = [name for name, _ in sorted_items[:k]]
        self.logger.info(f"Top {k} models by {by}: {top_names}")
        return top_names

    def tune_model_optuna(
        self,
        base_name: str,
        n_trials: int = 30,
    ) -> Tuple[Dict, float, float]:
        """
        Tune hyperparameters of a base model with Optuna.

        Supported base_name values:
        - "random_forest"
        - "elasticnet"
        - "xgb" (if xgboost installed)
        - "catboost" (if catboost installed)
        - "lgbm" (if lightgbm installed)

        Returns best_params, test_rmse, test_r2 for the tuned model.
        """
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna is not installed.")

        if self.feature_pipe_ is None:
            raise RuntimeError("Call build_preprocessing before tuning models.")
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data not available.")

        X_train = self.X_train_
        y_train = self.y_train_

        self.logger.info(
            f"Start Optuna tuning for base model '{base_name}' "
            f"with n_trials={n_trials}"
        )

        def objective(trial: "optuna.Trial") -> float:
            if base_name == "random_forest":
                n_estimators = trial.suggest_int("n_estimators", 300, 1200, step=100)
                max_depth = trial.suggest_int("max_depth", 4, 30)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
                max_features = trial.suggest_categorical(
                    "max_features",
                    ["sqrt", "log2", None],
                )
                est = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    n_jobs=-1,
                    random_state=self.random_state,
                )

            elif base_name == "elasticnet":
                alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
                est = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=30000,
                    random_state=self.random_state,
                )

            elif base_name == "xgb":
                if not HAS_XGB:
                    raise RuntimeError("xgboost not installed.")
                learning_rate = trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True,
                )
                max_depth = trial.suggest_int("max_depth", 3, 10)
                subsample = trial.suggest_float("subsample", 0.6, 0.95)
                colsample_bytree = trial.suggest_float(
                    "colsample_bytree",
                    0.6,
                    1.0,
                )
                min_child_weight = trial.suggest_float(
                    "min_child_weight",
                    1.0,
                    10.0,
                    log=True,
                )
                reg_lambda = trial.suggest_float(
                    "reg_lambda",
                    0.1,
                    10.0,
                    log=True,
                )
                reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
                gamma = trial.suggest_float("gamma", 0.0, 0.3)
                max_bin = trial.suggest_int("max_bin", 128, 512)
                est = xgb.XGBRegressor(
                    n_estimators=6000,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                    gamma=gamma,
                    objective="reg:squarederror",
                    random_state=self.random_state,
                    n_jobs=-1,
                    tree_method="hist",
                    max_bin=max_bin,
                    missing=np.nan,
                )

            elif base_name == "catboost":
                if not HAS_CAT:
                    raise RuntimeError("catboost not installed.")
                n_estimators = trial.suggest_int("n_estimators", 1000, 6000, step=500)
                depth = trial.suggest_int("depth", 4, 10)
                learning_rate = trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True,
                )
                l2_leaf_reg = trial.suggest_float(
                    "l2_leaf_reg",
                    1.0,
                    10.0,
                    log=True,
                )
                subsample = trial.suggest_float("subsample", 0.6, 0.95)
                colsample_bylevel = trial.suggest_float(
                    "colsample_bylevel",
                    0.6,
                    1.0,
                )
                est = CatBoostRegressor(
                    loss_function="RMSE",
                    n_estimators=n_estimators,
                    depth=depth,
                    learning_rate=learning_rate,
                    l2_leaf_reg=l2_leaf_reg,
                    subsample=subsample,
                    colsample_bylevel=colsample_bylevel,
                    random_state=self.random_state,
                    verbose=False,
                )

            elif base_name == "lgbm":
                if not HAS_LGBM:
                    raise RuntimeError("lightgbm not installed.")
                max_n_estimators = trial.suggest_int(
                    "max_n_estimators",
                    1000,
                    4000,
                    step=250,
                )
                learning_rate = trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True,
                )
                max_depth = trial.suggest_int("max_depth", 4, 20)
                num_leaves = trial.suggest_int("num_leaves", 16, 128)
                min_child_samples = trial.suggest_int(
                    "min_child_samples",
                    10,
                    100,
                )
                min_child_weight = trial.suggest_float(
                    "min_child_weight",
                    1e-4,
                    10.0,
                    log=True,
                )
                subsample = trial.suggest_float("subsample", 0.6, 0.95)
                colsample_bytree = trial.suggest_float(
                    "colsample_bytree",
                    0.6,
                    1.0,
                )
                reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
                reg_lambda = trial.suggest_float(
                    "reg_lambda",
                    0.1,
                    10.0,
                    log=True,
                )
                min_split_gain = trial.suggest_float(
                    "min_split_gain",
                    0.0,
                    0.5,
                )
                est = LGBMRegressorWithEarlyStopping(
                    max_n_estimators=max_n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    min_child_samples=min_child_samples,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    min_split_gain=min_split_gain,
                    early_stopping_rounds=150,
                    val_size=0.15,
                    random_state=self.random_state,
                )

            else:
                raise ValueError(
                    "Unsupported base_name for Optuna tuning: "
                    f"{base_name}"
                )

            pipe = Pipeline(
                [
                    ("features", self.feature_pipe_),
                    ("model", est),
                ]
            )
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                scoring=_get_scorers(),
                cv=5,
                n_jobs=-1,
                return_train_score=False,
            )
            cv_rmse_mean = -scores["test_neg_rmse"].mean()
            return cv_rmse_mean

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.logger.info(
            f"Best Optuna params for '{base_name}': {study.best_params}"
        )

        # Rebuild best estimator
        bp = study.best_params

        if base_name == "random_forest":
            best_est = RandomForestRegressor(
                n_estimators=bp.get("n_estimators", 600),
                max_depth=bp.get("max_depth", None),
                min_samples_split=bp.get("min_samples_split", 2),
                min_samples_leaf=bp.get("min_samples_leaf", 1),
                max_features=bp.get("max_features", "sqrt"),
                n_jobs=-1,
                random_state=self.random_state,
            )

        elif base_name == "elasticnet":
            best_est = ElasticNet(
                alpha=bp.get("alpha", 0.01),
                l1_ratio=bp.get("l1_ratio", 0.5),
                max_iter=30000,
                random_state=self.random_state,
            )

        elif base_name == "xgb":
            best_est = xgb.XGBRegressor(
                n_estimators=6000,
                learning_rate=bp.get("learning_rate", 0.03),
                max_depth=bp.get("max_depth", 4),
                subsample=bp.get("subsample", 0.8),
                colsample_bytree=bp.get("colsample_bytree", 0.8),
                min_child_weight=bp.get("min_child_weight", 2.0),
                reg_lambda=bp.get("reg_lambda", 3.0),
                reg_alpha=bp.get("reg_alpha", 0.2),
                gamma=bp.get("gamma", 0.05),
                objective="reg:squarederror",
                random_state=self.random_state,
                n_jobs=-1,
                tree_method="hist",
                max_bin=bp.get("max_bin", 256),
                missing=np.nan,
            )

        elif base_name == "catboost":
            best_est = CatBoostRegressor(
                loss_function="RMSE",
                n_estimators=bp.get("n_estimators", 3000),
                depth=bp.get("depth", 6),
                learning_rate=bp.get("learning_rate", 0.05),
                l2_leaf_reg=bp.get("l2_leaf_reg", 3.0),
                subsample=bp.get("subsample", 0.8),
                colsample_bylevel=bp.get("colsample_bylevel", 0.8),
                random_state=self.random_state,
                verbose=False,
            )

        elif base_name == "lgbm":
            max_depth = bp.get("max_depth", 12)
            num_leaves = bp.get("num_leaves", 31)
            if max_depth is not None and max_depth > 0:
                num_leaves = min(num_leaves, 2 ** max_depth - 1)
            best_est = LGBMRegressorWithEarlyStopping(
                max_n_estimators=bp.get("max_n_estimators", 3000),
                learning_rate=bp.get("learning_rate", 0.03),
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=bp.get("min_child_samples", 20),
                min_child_weight=bp.get("min_child_weight", 1e-3),
                subsample=bp.get("subsample", 0.8),
                colsample_bytree=bp.get("colsample_bytree", 0.8),
                reg_alpha=bp.get("reg_alpha", 0.1),
                reg_lambda=bp.get("reg_lambda", 1.0),
                min_split_gain=bp.get("min_split_gain", 0.0),
                early_stopping_rounds=150,
                val_size=0.15,
                random_state=self.random_state,
            )

        else:
            raise ValueError(f"Unknown base_name '{base_name}' after study.")

        tuned_name = f"{base_name}_tuned"
        metrics = self.train_single_model(tuned_name, best_est, cv_splits=5)
        return study.best_params, metrics["test_rmse"], metrics["test_r2"]

    def tune_model_gridsearch(
        self,
        base_name: str,
        param_grid: Dict,
    ) -> Tuple[Dict, float, float]:
        """
        Classic GridSearchCV tuning for simple base models
        that are not handled by tune_model_optuna.

        Parameters
        ----------
        base_name:
            One of the keys from get_default_models.
        param_grid:
            Parameter grid without 'model__' prefix. The method will add
            'model__' automatically.
        """
        if self.feature_pipe_ is None:
            raise RuntimeError("Call build_preprocessing before GridSearchCV.")

        base_models = self.get_default_models(self.random_state)
        if base_name not in base_models:
            raise ValueError(f"Unknown base model '{base_name}'.")

        est = base_models[base_name]

        pipe = Pipeline(
            [
                ("features", self.feature_pipe_),
                ("model", est),
            ]
        )

        grid = {f"model__{k}": v for k, v in param_grid.items()}

        self.logger.info(
            f"Start GridSearchCV for base model '{base_name}' "
            f"with grid keys {list(param_grid.keys())}"
        )
        search = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        search.fit(self.X_train_, self.y_train_)

        self.logger.info(
            f"Best GridSearchCV params for '{base_name}': {search.best_params_}"
        )

        best_pipe = search.best_estimator_
        name = f"{base_name}_grid"
        self.models_[name] = best_pipe

        # Evaluate on test set
        y_pred = best_pipe.predict(self.X_test_)
        test_rmse = _rmse(self.y_test_, y_pred)
        test_r2 = r2_score(self.y_test_, y_pred)

        self.results_[name] = {
            "cv_rmse_mean": -search.best_score_,
            "cv_rmse_std": np.nan,
            "cv_r2_mean": np.nan,
            "cv_r2_std": np.nan,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }
        return search.best_params_, test_rmse, test_r2

    def tune_top_models(
        self,
        top_model_names: List[str],
        n_trials: int = 30,
    ) -> List[str]:
        """
        Hyperparameter tuning for a list of top models.

        For models that are supported by tune_model_optuna:
            - Use Optuna helper.
        For others (ridge, lasso, svr):
            - Use a small GridSearchCV with a hand crafted grid.

        Returns the list of tuned model names that have been registered in
        self.models_ and self.results_.
        """
        tuned_names: List[str] = []

        grids: Dict[str, Dict] = {
            "ridge": {"alpha": [0.1, 1.0, 10.0, 50.0]},
            "lasso": {"alpha": [1e-4, 1e-3, 1e-2]},
            "svr": {
                "C": [1.0, 5.0, 10.0],
                "epsilon": [0.05, 0.1, 0.2],
                "gamma": ["scale", "auto"],
            },
        }

        for name in top_model_names:
            self.logger.info(f"Hyperparameter tuning for model '{name}'")

            try:
                if name in {
                    "random_forest",
                    "elasticnet",
                    "xgb",
                    "catboost",
                    "lgbm",
                }:
                    best_params, rmse, r2 = self.tune_model_optuna(
                        base_name=name,
                        n_trials=n_trials,
                    )
                    tuned_name = f"{name}_tuned"
                    self.logger.info(
                        f"Tuned {name} with Optuna. '{tuned_name}' "
                        f"RMSE={rmse:.4f}, R2={r2:.4f}, params={best_params}"
                    )
                    tuned_names.append(tuned_name)
                else:
                    if name not in grids:
                        self.logger.warning(
                            f"No grid defined for base model '{name}'. Skipping."
                        )
                        continue

                    best_params, rmse, r2 = self.tune_model_gridsearch(
                        base_name=name,
                        param_grid=grids[name],
                    )
                    tuned_name = f"{name}_grid"
                    self.logger.info(
                        f"Tuned {name} with GridSearch. '{tuned_name}' "
                        f"RMSE={rmse:.4f}, R2={r2:.4f}, params={best_params}"
                    )
                    tuned_names.append(tuned_name)
            except Exception as e:
                self.logger.warning(
                    f"Hyperparameter tuning failed for '{name}': {e}"
                )
                continue

        return tuned_names

    # -------------------------------------------------------------------------
    # Stacking with Optuna
    # -------------------------------------------------------------------------
    def _optuna_stack_objective(
        self,
        trial: "optuna.Trial",
        tuned_model_names: List[str],
    ) -> float:
        """
        Optuna objective for stacking.

        The search space includes:
        - Choice of 3 model combination from tuned_model_names
        - Hyperparameters for meta learner (ElasticNet)
        - Whether to passthrough original features to meta layer
        """
        if self.feature_pipe_ is None:
            raise RuntimeError("Feature pipeline not built.")
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Train data not available.")

        if len(tuned_model_names) < 3:
            raise ValueError("Need at least 3 tuned models for 3 model stacks.")

        # All 3 model combinations from candidate list
        combo_list = list(itertools.combinations(tuned_model_names, 3))
        combo = trial.suggest_categorical("stack_combo", combo_list)

        # Meta learner hyperparameters
        meta_alpha = trial.suggest_float("meta_alpha", 1e-4, 1e-1, log=True)
        meta_l1_ratio = trial.suggest_float("meta_l1_ratio", 0.1, 0.9)
        passthrough = trial.suggest_categorical("passthrough", [False, True])

        base_estimators = []
        for name in combo:
            if name not in self.models_:
                raise ValueError(f"Model '{name}' not found in self.models_.")
            pipe = self.models_[name]
            est = clone(pipe.named_steps["model"])
            base_estimators.append((name, est))

        meta = ElasticNet(
            alpha=meta_alpha,
            l1_ratio=meta_l1_ratio,
            random_state=self.random_state,
            max_iter=30000,
        )

        stack_reg = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta,
            n_jobs=-1,
            passthrough=passthrough,
        )

        stack_pipe = Pipeline(
            [
                ("features", self.feature_pipe_),
                ("stack", stack_reg),
            ]
        )

        scores = cross_validate(
            stack_pipe,
            self.X_train_,
            self.y_train_,
            scoring=_get_scorers(),
            cv=5,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_rmse_mean = -scores["test_neg_rmse"].mean()
        return cv_rmse_mean

    def tune_stacking_with_optuna(
        self,
        tuned_model_names: List[str],
        n_trials: int = 40,
    ) -> Tuple[str, Dict, Dict[str, float]]:
        """
        Tune a 3 model stacking ensemble with Optuna.

        The search chooses:
        - Which 3 tuned models to include
        - Hyperparameters of ElasticNet meta learner
        - Whether to passthrough original features

        Returns
        -------
        model_name:
            Name under which the stacking model is registered.
        best_params:
            The best Optuna parameters.
        metrics:
            Test metrics of the final fitted stacking model.
        """
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna is not installed.")

        self.logger.info(
            f"Start Optuna tuning for stacking with candidates: {tuned_model_names}"
        )

        if len(tuned_model_names) < 3:
            raise ValueError("Need at least 3 tuned models for stacking.")

        def objective(trial: "optuna.Trial") -> float:
            return self._optuna_stack_objective(trial, tuned_model_names)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_combo = best_params["stack_combo"]

        self.logger.info(
            f"Best stacking params: {best_params}"
        )

        # Build final stack from best parameters and fit on full train
        base_estimators = []
        for name in best_combo:
            pipe = self.models_[name]
            est = clone(pipe.named_steps["model"])
            base_estimators.append((name, est))

        meta = ElasticNet(
            alpha=best_params["meta_alpha"],
            l1_ratio=best_params["meta_l1_ratio"],
            random_state=self.random_state,
            max_iter=30000,
        )

        stack_reg = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta,
            n_jobs=-1,
            passthrough=best_params["passthrough"],
        )

        stack_pipe = Pipeline(
            [
                ("features", self.feature_pipe_),
                ("stack", stack_reg),
            ]
        )

        stack_pipe.fit(self.X_train_, self.y_train_)

        model_name = "stacking_optuna"
        self.models_[model_name] = stack_pipe

        # Evaluate on test
        y_pred = stack_pipe.predict(self.X_test_)
        test_rmse = _rmse(self.y_test_, y_pred)
        test_r2 = r2_score(self.y_test_, y_pred)

        metrics = {
            "cv_rmse_mean": study.best_value,
            "cv_rmse_std": np.nan,
            "cv_r2_mean": np.nan,
            "cv_r2_std": np.nan,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }

        self.results_[model_name] = metrics
        self.logger.info(
            f"[STACK {best_combo}] Test RMSE={test_rmse:.4f} Test R2={test_r2:.4f}"
        )
        return model_name, best_params, metrics

    # -------------------------------------------------------------------------
    # Persistence and plotting
    # -------------------------------------------------------------------------
    def save_model(self, name: str):
        """
        Save a fitted model pipeline to disk as .joblib.
        """
        if name not in self.models_:
            raise ValueError(f"Model '{name}' does not exist in registry.")
        path = self.output_dir / f"{name}.joblib"
        import joblib

        joblib.dump(self.models_[name], path)
        self.logger.info(f"Saved model '{name}' to {path}")

    def load_model(self, path: str, name: Optional[str] = None):
        """
        Load a model from .joblib and register it in self.models_.
        """
        import joblib

        model = joblib.load(path)
        if name is None:
            name = Path(path).stem
        self.models_[name] = model
        self.logger.info(f"Loaded model '{name}' from {path}")
        return model

    def save_results(self):
        """
        Save evaluation results to CSV and generate RMSE comparison plot.

        Outputs:
            - model_results.csv
            - rmse_comparison.png
        """
        if not self.results_:
            self.logger.warning("No results to save. Skipping save_results.")
            return

        df = pd.DataFrame(self.results_).T
        csv_path = self.output_dir / "model_results.csv"
        df.to_csv(csv_path)
        self.logger.info(f"Saved model comparison results to {csv_path}")

        # Plot by test RMSE
        plt.figure(figsize=(8, 5))
        idx = np.arange(len(df))
        plt.bar(idx, df["test_rmse"].values)
        plt.xticks(idx, df.index, rotation=45, ha="right")
        plt.ylabel("Test RMSE")
        plt.title("Model Test RMSE comparison")
        plt.tight_layout()
        fig_path = self.output_dir / "rmse_comparison.png"
        plt.savefig(fig_path)
        plt.close()
        self.logger.info(f"Saved RMSE comparison plot to {fig_path}")

    # -------------------------------------------------------------------------
    # High level pipeline
    # -------------------------------------------------------------------------
    def run_full_model_selection_and_stacking(
        self,
        top_k: int = 5,
        n_trials_model: int = 30,
        n_trials_stack: int = 40,
        cv_splits: int = 5,
    ) -> Dict[str, Dict]:
        """
        Run the full pipeline:

        1. Train all default models with CV
        2. Select top K by CV RMSE
        3. Hyperparameter tuning for these top K models
        4. Use tuned models to build 3 model stacks and tune stacks with Optuna
        5. Save results and plots

        Returns self.results_ for convenience.
        """
        self.logger.info("Starting full model selection and stacking pipeline.")

        # 1 and 2: default models and top K selection
        self.train_default_models(cv_splits=cv_splits)
        top_names = self.select_top_models(k=top_k, by="cv_rmse_mean")

        # 3: hyperparameter tuning for these top models
        tuned_names = self.tune_top_models(
            top_model_names=top_names,
            n_trials=n_trials_model,
        )

        if len(tuned_names) < 3:
            self.logger.warning(
                "Less than 3 tuned models produced. Stacking step will be skipped."
            )
            self.save_results()
            return self.results_

        # 4: stacking with Optuna over 3 model combos
        stack_name, stack_params, stack_metrics = self.tune_stacking_with_optuna(
            tuned_model_names=tuned_names,
            n_trials=n_trials_stack,
        )
        self.logger.info(
            f"Finished stacking. Best stack '{stack_name}' "
            f"test RMSE={stack_metrics['test_rmse']:.4f} "
            f"test R2={stack_metrics['test_r2']:.4f}"
        )

        # 5: persistence
        self.save_results()
        return self.results_


__all__ = ["ModelTrainer"]
