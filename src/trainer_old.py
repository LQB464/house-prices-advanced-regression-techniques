# src/modeling/trainer.py

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import itertools
import logging
import math

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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
    import optuna

    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

from preprocessing import Preprocessor, build_feature_pipeline, ORDINAL_MAP_CANONICAL

from .modeling.metrics import _rmse, _get_scorers
from .modeling.lgbm_wrapper import HAS_LGBM, LGBMRegressorWithEarlyStopping


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
        logger = logging.getLogger("ModelTrainer")

        if logger.handlers:
            logger.setLevel(log_level)
            return logger

        logger.setLevel(log_level)

        log_path = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # ===== NEW: print logs to console =====
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

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
        For regression tasks: perform stratified split by binning the target.
        
        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        if self.df_ is None:
            raise RuntimeError("No data loaded. Call load_data first.")

        # Split features and target
        X, y = self.dp.split_features_target(self.df_)

        # ----------------------------
        # 1) Stratified binning for regression
        # ----------------------------
        n_bins = 15 
        try:
            # create quantile-based bins
            y_binned = pd.qcut(y, q=n_bins, duplicates="drop", labels=False)
            stratify_label = y_binned
            self.logger.info(
                f"Using stratified split with {n_bins} quantile bins for continuous target."
            )
        except Exception as e:
            # Fall back to non-stratified split if binning fails
            self.logger.warning(
                f"Stratified binning failed ({e}). Falling back to random split."
            )
            stratify_label = None

        # ----------------------------
        # 2) Actual split
        # ----------------------------
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
            stratify=stratify_label,
        )

        # ----------------------------
        # 3) Logging distributions
        # ----------------------------
        if stratify_label is not None:
            train_bins = pd.qcut(self.y_train_, q=n_bins, duplicates="drop", labels=False)
            test_bins = pd.qcut(self.y_test_, q=n_bins, duplicates="drop", labels=False)

            self.logger.info(
                "Distribution of target bins after stratified split:\n"
                f"  Train bin counts: {train_bins.value_counts().sort_index().to_dict()}\n"
                f"  Test bin counts:  {test_bins.value_counts().sort_index().to_dict()}"
            )
        else:
            self.logger.info("Performed random split without stratification.")

        # ----------------------------
        # 4) Final log
        # ----------------------------
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
            df_train=self.X_train_,
            ordinal_mapping=ORDINAL_MAP_CANONICAL,
            use_domain_features=True,
            use_target_encoding=False,
            enable_variance_selector=True,
            variance_threshold=0.0,
            enable_kbest_mi=True,
            k_best_features=200,
            mi_random_state=0,
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
        n_trials: int = 20,
        cv_splits: int = 5,
    ) -> Tuple[Dict, float, float]:
        """
        Tune hyperparameters của một base model bằng Optuna.

        Supported base_name:
            - "random_forest"
            - "elasticnet"
            - "xgb"      (nếu cài xgboost)
            - "catboost" (nếu cài catboost)
            - "lgbm"     (nếu cài lightgbm)

        Trả về:
            best_params, test_rmse, test_r2
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
            f"[Optuna] Start tuning for base model '{base_name}' "
            f"with n_trials={n_trials}, cv_splits={cv_splits}"
        )

        # Sampler và pruner cho Optuna
        sampler = optuna.samplers.TPESampler(
            seed=self.random_state,
            n_startup_trials=10,
            multivariate=True,
            group=True,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=0,
        )

        # -----------------------------------------------------
        # Default params cho enqueue_trial (baseline hiện tại)
        # -----------------------------------------------------
        default_params: Dict[str, object] = {}

        if base_name == "random_forest":
            default_params = {
                "n_estimators": 600,
                "max_depth": 0,  # 0 sẽ được map thành None
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            }

        elif base_name == "elasticnet":
            default_params = {
                "alpha": 0.01,
                "l1_ratio": 0.5,
            }

        elif base_name == "xgb":
            if not HAS_XGB:
                raise RuntimeError("xgboost not installed.")
            default_params = {
                "learning_rate": 0.03,
                "n_estimators": 4000,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 2.0,
                "reg_lambda": 3.0,
                "reg_alpha": 0.2,
                "gamma": 0.05,
            }

        elif base_name == "catboost":
            if not HAS_CAT:
                raise RuntimeError("catboost not installed.")
            default_params = {
                "learning_rate": 0.05,
                "depth": 6,
                "n_estimators": 3000,
                "l2_leaf_reg": 3.0,
                "subsample": 0.8,
            }

        elif base_name == "lgbm":
            if not HAS_LGBM:
                raise RuntimeError("lightgbm not installed.")
            default_params = {
                "max_n_estimators": 3000,
                "learning_rate": 0.03,
                "max_depth": 12,
                "num_leaves": 31,
                "min_child_samples": 20,
                "min_child_weight": 1e-3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "min_split_gain": 0.0,
            }

        # -----------------------------------------------------
        # Objective với search space đã chỉnh
        # -----------------------------------------------------
        def objective(trial: "optuna.Trial") -> float:
            # RANDOM FOREST
            if base_name == "random_forest":
                base = default_params

                n_estimators = trial.suggest_int(
                    "n_estimators",
                    int(base["n_estimators"] * 0.5),   # 300
                    int(base["n_estimators"] * 3.0),   # 1800
                    step=100,
                )

                max_depth_raw = trial.suggest_int("max_depth", 0, 20)
                max_depth = None if max_depth_raw == 0 else max_depth_raw

                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)

                max_features = trial.suggest_categorical(
                    "max_features",
                    ["sqrt", "log2", 0.6, 0.8],
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

            # ELASTICNET
            elif base_name == "elasticnet":
                alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0.2, 0.8)
                est = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=50000,
                    random_state=self.random_state,
                )

            # XGBOOST
            elif base_name == "xgb":
                if not HAS_XGB:
                    raise RuntimeError("xgboost not installed.")

                base = default_params

                learning_rate = trial.suggest_float(
                    "learning_rate",
                    0.005,
                    0.1,
                    log=True,
                )
                n_estimators = trial.suggest_int(
                    "n_estimators",
                    int(base["n_estimators"] * 0.5),   # 2000
                    int(base["n_estimators"] * 1.5),   # 6000
                    step=500,
                )
                max_depth = trial.suggest_int("max_depth", 3, 8)
                subsample = trial.suggest_float("subsample", 0.6, 1.0)
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

                min_child_weight = trial.suggest_float(
                    "min_child_weight",
                    0.1,
                    20.0,
                    log=True,
                )
                reg_lambda = trial.suggest_float("reg_lambda", 0.1, 20.0, log=True)
                reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0)
                gamma = trial.suggest_float("gamma", 0.0, 0.5)

                est = xgb.XGBRegressor(
                    n_estimators=n_estimators,
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
                    max_bin=256,
                    missing=np.nan,
                )

            # CATBOOST
            elif base_name == "catboost":
                if not HAS_CAT:
                    raise RuntimeError("catboost not installed.")

                learning_rate = trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                )
                depth = trial.suggest_int("depth", 3, 10)
                n_estimators = trial.suggest_int(
                    "n_estimators", 1000, 5000, step=500
                )
                l2_leaf_reg = trial.suggest_float(
                    "l2_leaf_reg", 0.5, 50.0, log=True
                )
                subsample = trial.suggest_float("subsample", 0.5, 1.0)

                est = CatBoostRegressor(
                    loss_function="RMSE",
                    learning_rate=learning_rate,
                    depth=depth,
                    n_estimators=n_estimators,
                    l2_leaf_reg=l2_leaf_reg,
                    subsample=subsample,
                    random_state=self.random_state,
                    verbose=False,
                )

            # LIGHTGBM
            elif base_name == "lgbm":
                if not HAS_LGBM:
                    raise RuntimeError("lightgbm not installed.")

                max_n_estimators = trial.suggest_int(
                    "max_n_estimators", 1000, 6000, step=500
                )
                learning_rate = trial.suggest_float(
                    "learning_rate", 0.005, 0.3, log=True
                )
                max_depth = trial.suggest_int("max_depth", 3, 20)
                num_leaves = trial.suggest_int("num_leaves", 16, 512)
                min_child_samples = trial.suggest_int(
                    "min_child_samples", 5, 120
                )
                min_child_weight = trial.suggest_float(
                    "min_child_weight", 1e-4, 50.0, log=True
                )
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample_bytree = trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                )
                reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0)
                reg_lambda = trial.suggest_float(
                    "reg_lambda", 1e-2, 50.0, log=True
                )
                min_split_gain = trial.suggest_float(
                    "min_split_gain", 0.0, 0.5
                )

                if max_depth is not None and max_depth > 0:
                    num_leaves = min(num_leaves, 2 ** max_depth - 1)

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
                    early_stopping_rounds=100,
                    val_size=0.15,
                    random_state=self.random_state,
                )

            else:
                raise ValueError(
                    f"Unsupported base_name for Optuna tuning: {base_name}"
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
                cv=cv_splits,
                n_jobs=-1,
                return_train_score=False,
            )
            cv_rmse_mean = -scores["test_neg_rmse"].mean()

            self.logger.info(
                f"[Optuna {base_name} trial {trial.number}] "
                f"cv_rmse={cv_rmse_mean:.4f} params={trial.params}"
            )
            return cv_rmse_mean

        # -----------------------------------------------------
        # Tạo study + enqueue trial với default params
        # -----------------------------------------------------
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        if default_params:
            self.logger.info(
                f"[Optuna] Enqueue default params for '{base_name}': {default_params}"
            )
            study.enqueue_trial(default_params)

        study.optimize(objective, n_trials=n_trials)

        self.logger.info(
            f"[Optuna] Best params for '{base_name}': {study.best_params}"
        )
        bp = study.best_params

        # -----------------------------------------------------
        # Rebuild best estimator với best_params
        # -----------------------------------------------------
        if base_name == "random_forest":
            max_depth = bp.get("max_depth", 0)
            max_depth = None if max_depth == 0 else max_depth
            best_est = RandomForestRegressor(
                n_estimators=bp.get("n_estimators", 600),
                max_depth=max_depth,
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
                max_iter=50000,
                random_state=self.random_state,
            )

        elif base_name == "xgb":
            best_est = xgb.XGBRegressor(
                n_estimators=bp.get("n_estimators", 1500),
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
                max_bin=256,
                missing=np.nan,
            )

        elif base_name == "catboost":
            best_est = CatBoostRegressor(
                loss_function="RMSE",
                learning_rate=bp.get("learning_rate", 0.05),
                depth=bp.get("depth", 6),
                n_estimators=bp.get("n_estimators", 1500),
                l2_leaf_reg=bp.get("l2_leaf_reg", 3.0),
                subsample=bp.get("subsample", 0.8),
                random_state=self.random_state,
                verbose=False,
            )

        elif base_name == "lgbm":
            max_depth = bp.get("max_depth", 12)
            num_leaves = bp.get("num_leaves", 31)
            if max_depth is not None and max_depth > 0:
                num_leaves = min(num_leaves, 2 ** max_depth - 1)

            best_est = LGBMRegressorWithEarlyStopping(
                max_n_estimators=bp.get("max_n_estimators", 1500),
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
                early_stopping_rounds=100,
                val_size=0.15,
                random_state=self.random_state,
            )

        else:
            raise ValueError(f"Unknown base_name '{base_name}' after study.")

        # -----------------------------------------------------
        # Train model tuned trên full train + so sánh với baseline
        # -----------------------------------------------------
        tuned_name = f"{base_name}_tuned"
        metrics = self.train_single_model(tuned_name, best_est, cv_splits=cv_splits)
        test_rmse = metrics["test_rmse"]
        test_r2 = metrics["test_r2"]

        base_metrics = self.results_.get(base_name)
        if base_metrics is not None:
            base_test_rmse = base_metrics["test_rmse"]
            # nếu tuned tệ hơn test > 1% thì giữ baseline
            if test_rmse > base_test_rmse * 1.01:
                self.logger.info(
                    f"[Optuna] Tuned '{base_name}' worse on test "
                    f"({test_rmse:.1f} > {base_test_rmse:.1f}), "
                    "keeping baseline model for stacking."
                )
                # map '{base_name}_tuned' về đúng baseline để stacking vẫn dùng được
                self.results_[tuned_name] = base_metrics.copy()
                self.models_[tuned_name] = self.models_[base_name]
                try:
                    self.save_model(tuned_name)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to save baseline model under '{tuned_name}': {e}"
                    )
                return bp, base_test_rmse, base_metrics["test_r2"]

        # nếu tuned không tệ hơn baseline: lưu model tuned
        try:
            self.save_model(tuned_name)
        except Exception as e:
            self.logger.warning(
                f"Failed to save tuned model '{tuned_name}': {e}"
            )

        return bp, test_rmse, test_r2

    def tune_model_gridsearch(
        self,
        base_name: str,
        param_grid: Dict,
        cv_splits: int = 5,
    ) -> Tuple[Dict, float, float]:
        """
        Classic GridSearchCV tuning for simple base models
        that are not handled by tune_model_optuna.
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
            f"with grid keys {list(param_grid.keys())}, cv_splits={cv_splits}"
        )
        search = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
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

        try:
            self.save_model(name)
        except Exception as e:
            self.logger.warning(
                f"Failed to save grid search model '{name}': {e}"
            )

        return search.best_params_, test_rmse, test_r2

    def tune_top_models(
        self,
        top_model_names: List[str],
        n_trials: int = 20,
        cv_splits: int = 5,
    ) -> List[str]:
        """
        Hyperparameter tuning cho danh sách top models.
        """
        tuned_names: List[str] = []

        grids: Dict[str, Dict] = {
            "ridge": {
                "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3],
            },
            "lasso": {
                "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            },
            "svr": {
                "C": [0.5, 1.0, 5.0, 10.0, 20.0],
                "epsilon": [0.01, 0.05, 0.1, 0.2],
                "gamma": ["scale", "auto"],
            },
        }

        optuna_supported = {
            "random_forest",
            "elasticnet",
            "xgb",
            "catboost",
            "lgbm",
        }
        grid_supported = set(grids.keys())

        for name in top_model_names:
            self.logger.info(f"Hyperparameter tuning for model '{name}'")

            try:
                if name in optuna_supported:
                    best_params, rmse, r2 = self.tune_model_optuna(
                        base_name=name,
                        n_trials=n_trials,
                        cv_splits=cv_splits,
                    )
                    tuned_name = f"{name}_tuned"
                    self.logger.info(
                        f"Tuned {name} with Optuna. '{tuned_name}' "
                        f"RMSE={rmse:.4f}, R2={r2:.4f}, params={best_params}"
                    )
                    tuned_names.append(tuned_name)

                elif name in grid_supported:
                    best_params, rmse, r2 = self.tune_model_gridsearch(
                        base_name=name,
                        param_grid=grids[name],
                        cv_splits=cv_splits,
                    )
                    tuned_name = f"{name}_grid"
                    self.logger.info(
                        f"Tuned {name} with GridSearch. '{tuned_name}' "
                        f"RMSE={rmse:.4f}, R2={r2:.4f}, params={best_params}"
                    )
                    tuned_names.append(tuned_name)

                else:
                    self.logger.warning(
                        f"No tuning strategy defined for base model '{name}'. Skipping."
                    )
                    continue

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
        cv_splits: int,
    ) -> float:
        """
        Optuna objective for stacking.
        """
        if self.feature_pipe_ is None:
            raise RuntimeError("Feature pipeline not built.")
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Train data not available.")

        if len(tuned_model_names) < 3:
            raise ValueError("Need at least 3 tuned models for 3 model stacks.")

        combo_list = list(itertools.combinations(tuned_model_names, 3))
        combo = trial.suggest_categorical("stack_combo", combo_list)

        meta_alpha = trial.suggest_float("meta_alpha", 1e-5, 1e-1, log=True)
        meta_l1_ratio = trial.suggest_float("meta_l1_ratio", 0.0, 1.0)
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
            final_estimator=Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("enet", meta),
            ]),
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
            cv=cv_splits,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_rmse_mean = -scores["test_neg_rmse"].mean()

        self.logger.info(
            f"[STACK TRIAL {trial.number}] combo={combo} "
            f"meta_alpha={meta_alpha:.6f} meta_l1_ratio={meta_l1_ratio:.3f} "
            f"passthrough={passthrough} cv_rmse={cv_rmse_mean:.4f}"
        )

        return cv_rmse_mean

    def tune_stacking_with_optuna(
        self,
        tuned_model_names: List[str],
        n_trials: int = 20,
        cv_splits: int = 5,
    ) -> Tuple[str, Dict, Dict[str, float]]:
        """
        Tune a 3 model stacking ensemble with Optuna.
        """
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna is not installed.")

        self.logger.info(
            f"Start Optuna tuning for stacking with candidates: {tuned_model_names}"
        )

        if len(tuned_model_names) < 3:
            raise ValueError("Need at least 3 tuned models for stacking.")

        sampler = optuna.samplers.TPESampler(seed=self.random_state)

        def objective(trial: "optuna.Trial") -> float:
            return self._optuna_stack_objective(
                trial, tuned_model_names, cv_splits=cv_splits
            )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )
        study.optimize(objective, n_trials=n_trials)
        
        trials_data = []
        for t in study.trials:
            # chỉ lấy trial hoàn thành
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue

            params = t.params
            combo = params.get("stack_combo")

            # combo là tuple/list -> stringify cho dễ lưu csv
            if isinstance(combo, (list, tuple)):
                combo_str = " + ".join(combo)
            else:
                combo_str = str(combo)

            trials_data.append(
                {
                    "trial_number": t.number,
                    "value_cv_rmse": t.value,
                    "stack_combo": combo_str,
                    "meta_alpha": params.get("meta_alpha"),
                    "meta_l1_ratio": params.get("meta_l1_ratio"),
                    "passthrough": params.get("passthrough"),
                }
            )
            
        if trials_data:
            df_trials = pd.DataFrame(trials_data)
            trials_path = self.output_dir / "stacking_optuna_trials.csv"
            df_trials.to_csv(trials_path, index=False)
            self.logger.info(
                f"Saved stacking Optuna trials to {trials_path}"
            )

            # Summary theo từng combo: best / mean cv_rmse, số trial
            summary = (
                df_trials.groupby("stack_combo")["value_cv_rmse"]
                .agg(["min", "mean", "count"])
                .reset_index()
                .rename(
                    columns={
                        "min": "best_cv_rmse",
                        "mean": "mean_cv_rmse",
                        "count": "n_trials",
                    }
                )
            )
            summary_path = self.output_dir / "stacking_optuna_summary.csv"
            summary.to_csv(summary_path, index=False)
            self.logger.info(
                f"Saved stacking summary per combo to {summary_path}"
            )

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

        try:
            self.save_model(model_name)
        except Exception as e:
            self.logger.warning(
                f"Failed to save stacking model '{model_name}': {e}"
            )

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
        n_trials_model: int = 20,
        n_trials_stack: int = 20,
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
            cv_splits=cv_splits,
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
