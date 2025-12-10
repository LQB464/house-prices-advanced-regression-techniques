"""
modeling/tuning_mixin.py

Hyperparameter tuning utilities supporting both Optuna-based search and
GridSearchCV, applied to multiple model families in a unified pipeline.

Extended Description
--------------------
This module provides the `TuningMixin`, which enables:
- Optuna-based hyperparameter tuning for complex models such as
  RandomForest, ElasticNet, XGBoost, CatBoost, and LightGBM.
- GridSearchCV tuning for simpler linear and kernel-based models.
- Unified scoring using the project's RMSE and R2 scorers.
- Automatic reconstruction of the best estimator from tuned parameters.
- Consistent logging, model registration, and result storage.

The mixin expects to operate inside a TrainerConfig-derived class that provides:
`self.feature_pipe_`, dataset splits (`self.X_train_`, `self.y_train_`,
`self.X_test_`, `self.y_test_`), registries (`self.models_`, `self.results_`),
as well as `self.output_dir`, `self.random_state`, `self.logger`,
and utility functions `self.train_single_model` and `self.save_model`.

Main Components
---------------
- _build_tuning_pipeline: wrap estimators with preprocessing
- tune_model_optuna: Optuna tuning for supported models
- tune_model_gridsearch: GridSearchCV tuning for simple models
- tune_top_models: orchestrates tuning for top-ranked models

Usage Example
-------------
>>> class Trainer(TrainerConfig, TuningMixin):
...     pass
>>> best_params, rmse, r2 = trainer.tune_model_optuna("random_forest")
>>> tuned_names = trainer.tune_top_models(["ridge", "random_forest"])

Notes
-----
Certain model types require optional dependencies (XGBoost, CatBoost,
LightGBM). If these libraries are not installed, their tuning paths
raise informative runtime errors.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

from .metrics import _rmse, _get_scorers
from .lgbm_wrapper import LGBMRegressorWithEarlyStopping, HAS_LGBM
from .default_models_mixin import HAS_XGB, HAS_CATBOOST

try:
    import optuna

    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

if HAS_XGB:
    import xgboost as xgb

if HAS_CATBOOST:
    from catboost import CatBoostRegressor


class TuningMixin:
    """
    Hyperparameter tuning engine supporting Optuna and GridSearchCV workflows.

    Extended Description
    --------------------
    The `TuningMixin` automates:
    - building preprocessing + model pipelines for tuning
    - creating Optuna objectives for complex model families
    - running TPE sampling with pruning when available
    - reconstructing the best estimator from tuned parameters
    - optionally falling back to GridSearch for simpler models
    - tracking tuned model performance and saving results

    It requires several TrainerConfig attributes:
    `self.feature_pipe_`, `self.X_train_`, `self.y_train_`,
    `self.X_test_`, `self.y_test_`, `self.models_`, `self.results_`,
    `self.output_dir`, `self.random_state`, `self.logger`,
    and utility functions `self.train_single_model` and `self.save_model`.

    Parameters
    ----------
    None
        The mixin depends entirely on TrainerConfig for shared state.

    Attributes
    ----------
    models_ : dict
        Registry of fitted models (baseline and tuned).
    results_ : dict
        Evaluation metrics for each model.
    feature_pipe_ : sklearn.Pipeline or None
        Preprocessing pipeline shared by all tuned models.

    Examples
    --------
    >>> best_params, rmse, r2 = trainer.tune_model_optuna("elasticnet")
    >>> tuned = trainer.tune_top_models(["ridge", "svr"])
    """

    def _build_tuning_pipeline(self, est: BaseEstimator) -> Pipeline:
        if self.feature_pipe_ is None:
            raise RuntimeError("Feature pipeline not built.")
        return Pipeline(
            [
                ("features", self.feature_pipe_),
                ("model", est),
            ]
        )

    # Optuna tuning
    def tune_model_optuna(
        self,
        base_name: str,
        n_trials: int = 50,
        cv_splits: int = 5,
    ) -> Tuple[Dict, float, float]:
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna is not installed.")

        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data not available.")
        if self.feature_pipe_ is None:
            raise RuntimeError("Feature pipeline not built.")

        X_train = self.X_train_
        y_train = self.y_train_

        self.logger.info(
            f"[Optuna] Start tuning for base model '{base_name}' "
            f"with n_trials={n_trials}, cv_splits={cv_splits}"
        )

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

        def objective(trial: optuna.Trial) -> float:
            # RandomForest
            if base_name == "random_forest":
                n_estimators = trial.suggest_int("n_estimators", 200, 1200, step=200)
                max_depth = trial.suggest_int("max_depth", 0, 30)
                max_depth = None if max_depth == 0 else max_depth
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

            # ElasticNet
            elif base_name == "elasticnet":
                alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
                est = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=50_000,
                    random_state=self.random_state,
                )

            # XGBoost
            elif base_name == "xgb":
                if not HAS_XGB:
                    raise RuntimeError("XGBoost is not installed.")
                learning_rate = trial.suggest_float(
                    "learning_rate", 0.005, 0.3, log=True
                )
                n_estimators = trial.suggest_int(
                    "n_estimators", 400, 2000, step=200
                )
                max_depth = trial.suggest_int("max_depth", 3, 12)
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample_bytree = trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                )
                min_child_weight = trial.suggest_float(
                    "min_child_weight", 1e-3, 20.0, log=True
                )
                reg_lambda = trial.suggest_float(
                    "reg_lambda", 1e-3, 50.0, log=True
                )
                reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)

                est = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=self.random_state,
                )

            # CatBoost
            elif base_name == "catboost":
                if not HAS_CATBOOST:
                    raise RuntimeError("CatBoost is not installed.")
                learning_rate = trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                )
                depth = trial.suggest_int("depth", 4, 10)
                n_estimators = trial.suggest_int(
                    "n_estimators", 500, 3000, step=250
                )
                l2_leaf_reg = trial.suggest_float(
                    "l2_leaf_reg", 1.0, 10.0
                )
                subsample = trial.suggest_float("subsample", 0.5, 1.0)

                est = CatBoostRegressor(
                    loss_function="RMSE",
                    learning_rate=learning_rate,
                    depth=depth,
                    n_estimators=n_estimators,
                    l2_leaf_reg=l2_leaf_reg,
                    subsample=subsample,
                    random_seed=self.random_state,
                    verbose=False,
                )

            # LightGBM
            elif base_name == "lgbm":
                if not HAS_LGBM:
                    raise RuntimeError("LightGBM is not installed.")

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
                    "min_split_gain", 0.0, 5.0
                )

                # Guard num_leaves vs max_depth
                if max_depth > 0:
                    max_leaves = 2 ** max_depth - 1
                    if num_leaves > max_leaves:
                        num_leaves = max_leaves

                est = LGBMRegressorWithEarlyStopping(
                    max_n_estimators=max_n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    min_child_samples=min_child_samples,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    min_split_gain=min_split_gain,
                    random_state=self.random_state,
                )

            else:
                raise ValueError(f"Unsupported base_name '{base_name}' for Optuna tuning.")

            pipe = self._build_tuning_pipeline(est)
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                scoring=_get_scorers(),
                cv=cv_splits,
                n_jobs=-1,
                return_train_score=False,
            )
            cv_rmse = -scores["test_neg_rmse"].mean()

            self.logger.info(
                f"[Optuna trial {trial.number}] base={base_name} cv_rmse={cv_rmse:.4f}"
            )
            return float(cv_rmse)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        self.logger.info(
            f"[Optuna] Best params for '{base_name}': {best_params} "
            f"best_cv_rmse={study.best_value:.4f}"
        )

        # Rebuild best estimator
        if base_name == "random_forest":
            max_depth = best_params.get("max_depth", 0)
            max_depth = None if max_depth == 0 else max_depth
            best_est = RandomForestRegressor(
                n_estimators=best_params.get("n_estimators", 600),
                max_depth=max_depth,
                min_samples_split=best_params.get("min_samples_split", 2),
                min_samples_leaf=best_params.get("min_samples_leaf", 1),
                max_features=best_params.get("max_features", "sqrt"),
                n_jobs=-1,
                random_state=self.random_state,
            )
        elif base_name == "elasticnet":
            best_est = ElasticNet(
                alpha=best_params.get("alpha", 0.1),
                l1_ratio=best_params.get("l1_ratio", 0.5),
                max_iter=50_000,
                random_state=self.random_state,
            )
        elif base_name == "xgb":
            best_est = xgb.XGBRegressor(
                n_estimators=best_params.get("n_estimators", 800),
                learning_rate=best_params.get("learning_rate", 0.03),
                max_depth=best_params.get("max_depth", 6),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                min_child_weight=best_params.get("min_child_weight", 1.0),
                reg_lambda=best_params.get("reg_lambda", 1.0),
                reg_alpha=best_params.get("reg_alpha", 0.0),
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=self.random_state,
            )
        elif base_name == "catboost":
            best_est = CatBoostRegressor(
                loss_function="RMSE",
                n_estimators=best_params.get("n_estimators", 1000),
                learning_rate=best_params.get("learning_rate", 0.05),
                depth=best_params.get("depth", 6),
                l2_leaf_reg=best_params.get("l2_leaf_reg", 3.0),
                subsample=best_params.get("subsample", 0.8),
                verbose=False,
                random_seed=self.random_state,
            )
        elif base_name == "lgbm":
            max_depth = best_params.get("max_depth", 6)
            num_leaves = best_params.get("num_leaves", 31)
            if max_depth > 0:
                max_leaves = 2 ** max_depth - 1
                if num_leaves > max_leaves:
                    num_leaves = max_leaves

            best_est = LGBMRegressorWithEarlyStopping(
                max_n_estimators=best_params.get("max_n_estimators", 3000),
                learning_rate=best_params.get("learning_rate", 0.03),
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_child_samples=best_params.get("min_child_samples", 20),
                min_child_weight=best_params.get("min_child_weight", 1.0),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                reg_alpha=best_params.get("reg_alpha", 0.0),
                reg_lambda=best_params.get("reg_lambda", 1.0),
                min_split_gain=best_params.get("min_split_gain", 0.0),
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported base_name '{base_name}' for Optuna tuning.")

        tuned_name = f"{base_name}_tuned"
        metrics = self.train_single_model(
            name=tuned_name,
            estimator=best_est,
            cv_splits=cv_splits,
        )

        test_rmse = metrics["test_rmse"]
        test_r2 = metrics["test_r2"]

        base_metrics = self.results_.get(base_name)
        if base_metrics is not None:
            base_test_rmse = base_metrics["test_rmse"]
            if test_rmse > base_test_rmse * 1.01:
                self.logger.info(
                    f"[Optuna] Tuned '{base_name}' worse on test "
                    f"({test_rmse:.4f} > {base_test_rmse:.4f}), "
                    "keeping baseline model for stacking."
                )
                self.results_[tuned_name] = base_metrics.copy()
                self.models_[tuned_name] = self.models_[base_name]
                try:
                    self.save_model(tuned_name)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to save baseline model under '{tuned_name}': {e}"
                    )
                return best_params, base_test_rmse, base_metrics["test_r2"]

        try:
            self.save_model(tuned_name)
        except Exception as e:
            self.logger.warning(f"Failed to save tuned model '{tuned_name}': {e}")

        return best_params, test_rmse, test_r2

    # GridSearch tuning
    def tune_model_gridsearch(
        self,
        base_name: str,
        param_grid: Dict[str, List],
        cv_splits: int = 5,
    ) -> Tuple[Dict, float, float]:
        
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data not available.")
        if self.feature_pipe_ is None:
            raise RuntimeError("Feature pipeline not built.")

        self.logger.info(
            f"[GridSearch] Start tuning base model '{base_name}' "
            f"with grid size={len(param_grid)}"
        )

        base_models = self.get_default_models(random_state=self.random_state)
        if base_name not in base_models:
            raise ValueError(f"Base model '{base_name}' is not in default models.")

        est = base_models[base_name]
        pipe = self._build_tuning_pipeline(est)

        grid = {f"model__{k}": v for k, v in param_grid.items()}

        grid_cv = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
            n_jobs=-1,
            refit=True,
        )
        grid_cv.fit(self.X_train_, self.y_train_)

        best_params = {
            k.replace("model__", ""): v for k, v in grid_cv.best_params_.items()
        }
        self.logger.info(
            f"[GridSearch] Best params for '{base_name}': {best_params} "
            f"best_cv_rmse={-grid_cv.best_score_:.4f}"
        )

        tuned_name = f"{base_name}_grid"
        self.models_[tuned_name] = grid_cv.best_estimator_

        y_pred = self.models_[tuned_name].predict(self.X_test_)
        test_rmse = _rmse(self.y_test_, y_pred)

        from sklearn.metrics import r2_score

        test_r2 = float(r2_score(self.y_test_, y_pred))

        metrics = {
            "cv_rmse_mean": float(-grid_cv.best_score_),
            "cv_rmse_std": float(np.nan),
            "cv_r2_mean": float(np.nan),
            "cv_r2_std": float(np.nan),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
        }

        self.results_[tuned_name] = metrics

        try:
            self.save_model(tuned_name)
        except Exception as e:
            self.logger.warning(
                f"Failed to save tuned gridsearch model '{tuned_name}': {e}"
            )

        return best_params, test_rmse, test_r2

    # Orchestrate tuning cho top models
    def tune_top_models(
        self,
        top_model_names: List[str],
        n_trials: int = 50,
        cv_splits: int = 5,
    ) -> List[str]:
        
        tuned_names: List[str] = []

        grids = {
            "ridge": {
                "alpha": [0.1, 1.0, 10.0, 100.0],
            },
            "lasso": {
                "alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0],
            },
            "svr": {
                "C": [1.0, 10.0, 50.0, 100.0],
                "epsilon": [0.01, 0.1, 0.2],
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
                    self.logger.info(
                        f"No tuning strategy defined for model '{name}'. Skipping."
                    )
            except Exception as e:
                self.logger.exception(
                    f"Failed to tune model '{name}' due to error: {e}"
                )

        self.logger.info(f"Tuned models: {tuned_names}")
        return tuned_names
