"""
Model selection + hyperparameter tuning + 3-model stacking for Ames-like house prices.

- Evaluates up to 8 base models (RF, XGB, SVR, ElasticNet, Ridge, Lasso, CatBoost, LightGBM)
- Selects top-5 (by CV RMSE), tunes them with Optuna
- Tries all 3-model stacks from the top-5 and picks the best on the test split
- Saves plots and CSV artifacts
- LightGBM hardened and wrapped for early stopping inside stacks
"""

import warnings
warnings.filterwarnings("ignore")

import math
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, RidgeCV
from sklearn.svm import SVR
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split as tts

# Optional deps
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

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

# Your feature engineering pipeline (provided by you)
from house_price_pipeline import make_feature_space

RANDOM_STATE = 42


# ---------------------------
# Helpers
# ---------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_scorers():
    # Neg RMSE so higher is better for cross_validate (we will flip sign later)
    scorers = {
        "neg_rmse": make_scorer(
            lambda yt, yp: -math.sqrt(mean_squared_error(yt, yp)),
            greater_is_better=True
        ),
        "r2": "r2",
    }
    return scorers


def build_feature_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Pipeline:
    # Infer column groups from train and test; safe for target encoding and rare pooling
    return make_feature_space(X_train, X_test)


def _guard_num_leaves(num_leaves: int, max_depth: int) -> int:
    if max_depth is not None and max_depth > 0:
        return int(min(num_leaves, 2 ** max_depth - 1))
    return int(num_leaves)


# ---------------------------
# LightGBM wrapper with internal early stopping
# ---------------------------
class LGBMRegressorWithEarlyStopping(BaseEstimator, RegressorMixin):
    """
    LightGBM wrapper that uses an internal train and validation split for early stopping.
    This allows early stopping even when used inside StackingRegressor.
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
        random_state: int = 42
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
        assert HAS_LGBM, "lightgbm is not installed."
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
            objective="regression",
            deterministic=True,
            verbosity=-1,
            n_jobs=-1,
            random_state=self.random_state,
        )

        try:
            self.model_.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        except Exception as e:
            print(f"[LGBM wrapper] Early stopping failed: {e}. Training without validation.")
            # Fallback: train on full data with fewer trees
            self.model_.set_params(n_estimators=max(500, self.max_n_estimators // 4))
            self.model_.fit(X, y)
            self.best_iteration_ = None

        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet")
        return self.model_.predict(X, num_iteration=getattr(self, "best_iteration_", None))

    def get_params(self, deep=True):
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


# ---------------------------
# Model zoo
# ---------------------------
def base_models_dict() -> Dict[str, object]:
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=RANDOM_STATE),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.0005, max_iter=20000, random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
    }

    if HAS_XGB:
        models["XGB"] = xgb.XGBRegressor(
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
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            max_bin=256,
            missing=np.nan
        )

    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            loss_function="RMSE",
            n_estimators=3000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            random_state=RANDOM_STATE,
            verbose=False
        )

    if HAS_LGBM:
        # Use the wrapper instead of raw LGBMRegressor
        models["LGBM"] = LGBMRegressorWithEarlyStopping(
            max_n_estimators=3000,  # will early stop much earlier typically
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
            random_state=RANDOM_STATE
        )

    return models


# ---------------------------
# Baseline evaluation
# ---------------------------
def evaluate_models(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_pipe: Pipeline,
    cv_splits: int = 5,
    out_prefix: str = "baseline"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    preds_test = {}

    print(f"Evaluating {len(models)} models with {cv_splits}-fold CV...")
    for name, est in models.items():
        pipe = Pipeline([("features", clone(feature_pipe)), ("model", est)])
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

        # Cross-validated scores
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=get_scorers(),
            return_train_score=False, n_jobs=-1
        )
        mean_neg_rmse = scores["test_neg_rmse"].mean()
        std_neg_rmse = scores["test_neg_rmse"].std()
        mean_r2 = scores["test_r2"].mean()
        std_r2 = scores["test_r2"].std()

        # Fit once on full train and evaluate on test
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            # Defensive fallback for LightGBM-specific corner cases
            if HAS_LGBM and isinstance(est, (LGBMRegressor, LGBMRegressorWithEarlyStopping)):
                print(f"[warn] LightGBM fit failed in baseline for {name} ({e}). Retrying with safer params...")
                if isinstance(est, LGBMRegressorWithEarlyStopping):
                    safer = clone(est)
                    safer.set_params(
                        num_leaves=min(31, getattr(est, "num_leaves", 31)),
                        max_depth=10,
                        min_child_samples=max(30, getattr(est, "min_child_samples", 20)),
                    )
                else:
                    safer = clone(est)
                    safer.set_params(
                        num_leaves=min(31, getattr(est, "num_leaves", 31)),
                        max_depth=10,
                        min_child_samples=max(30, getattr(est, "min_child_samples", 20)),
                        min_data_in_bin=5,
                    )
                pipe = Pipeline([("features", clone(feature_pipe)), ("model", safer)])
                pipe.fit(X_train, y_train)
            else:
                raise

        yp = pipe.predict(X_test)
        test_rmse = rmse(y_test, yp)
        test_r2 = r2_score(y_test, yp)

        results.append({
            "model": name,
            "cv_rmse_mean": -mean_neg_rmse,
            "cv_rmse_std": std_neg_rmse,
            "cv_r2_mean": mean_r2,
            "cv_r2_std": std_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })
        preds_test[name] = yp

        print(f"{name:<12s}  CV RMSE={-mean_neg_rmse:.4f}  CV R2={mean_r2:.4f}  | Test RMSE={test_rmse:.4f}  Test R2={test_r2:.4f}")

    res_df = pd.DataFrame(results).sort_values("cv_rmse_mean")
    # Save
    res_df.to_csv(f"{out_prefix}_model_cv_test_results.csv", index=False)

    # Plots
    plt.figure(figsize=(10, 5))
    idx = np.arange(len(res_df))
    plt.bar(idx, res_df["cv_rmse_mean"].values)
    plt.xticks(idx, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("CV RMSE (lower is better)")
    plt.title("Baseline cross-validated RMSE by model")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cv_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(idx, res_df["cv_r2_mean"].values)
    plt.xticks(idx, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("CV R2 (higher is better)")
    plt.title("Baseline cross-validated R2 by model")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cv_r2.png", dpi=150)
    plt.close()

    # Test plots
    plt.figure(figsize=(10, 5))
    plt.bar(idx, res_df["test_rmse"].values)
    plt.xticks(idx, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("Test RMSE by model")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_test_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(idx, res_df["test_r2"].values)
    plt.xticks(idx, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("Test R2")
    plt.title("Test R2 by model")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_test_r2.png", dpi=150)
    plt.close()

    # Save predictions
    preds_df = pd.DataFrame(preds_test)
    preds_df.to_csv(f"{out_prefix}_test_predictions.csv", index=False)

    return res_df, preds_df


# ---------------------------
# Optuna tuning with LGBM wrapper
# ---------------------------
def make_objective(name: str, X_train: pd.DataFrame, y_train: pd.Series, feature_pipe: Pipeline):
    def objective(trial):
        if name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 300, 1200, step=100)
            max_depth = trial.suggest_int("max_depth", 4, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

        elif name == "ElasticNet":
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=30000, random_state=RANDOM_STATE)

        elif name == "Ridge":
            alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
            model = Ridge(alpha=alpha, random_state=RANDOM_STATE)

        elif name == "Lasso":
            alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
            model = Lasso(alpha=alpha, max_iter=30000, random_state=RANDOM_STATE)

        elif name == "SVR":
            C = trial.suggest_float("C", 0.1, 100.0, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)

        elif name == "XGB" and HAS_XGB:
            learning_rate = trial.suggest_float("learning_rate", 0.005, 0.08, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 7)
            min_child_weight = trial.suggest_float("min_child_weight", 1.0, 8.0)
            subsample = trial.suggest_float("subsample", 0.6, 0.95)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 0.95)
            reg_lambda = trial.suggest_float("reg_lambda", 0.3, 10.0, log=True)
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
            gamma = trial.suggest_float("gamma", 0.0, 0.3)
            max_bin = trial.suggest_int("max_bin", 128, 512)
            model = xgb.XGBRegressor(
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
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                max_bin=max_bin,
                missing=np.nan
            )

        elif name == "CatBoost" and HAS_CAT:
            n_estimators = trial.suggest_int("n_estimators", 1000, 6000, step=500)
            depth = trial.suggest_int("depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 0.95)
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.6, 1.0)
            model = CatBoostRegressor(
                loss_function="RMSE",
                n_estimators=n_estimators,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                subsample=subsample,
                colsample_bylevel=colsample_bylevel,
                random_state=RANDOM_STATE,
                verbose=False
            )

        elif name == "LGBM" and HAS_LGBM:
            # Use the wrapper for tuning too
            max_n_estimators = trial.suggest_int("max_n_estimators", 1000, 4000, step=500)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            max_depth = trial.suggest_int("max_depth", 6, 14)
            num_leaves = trial.suggest_int("num_leaves", 16, 127)
            num_leaves = _guard_num_leaves(num_leaves, max_depth)
            min_child_samples = trial.suggest_int("min_child_samples", 10, 50)
            min_child_weight = trial.suggest_float("min_child_weight", 1e-3, 0.1, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 0.95)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
            reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)
            min_split_gain = trial.suggest_float("min_split_gain", 0.0, 0.3)

            model = LGBMRegressorWithEarlyStopping(
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
                random_state=RANDOM_STATE
            )
        else:
            raise RuntimeError(f"Unknown or unavailable model for tuning: {name}")

        # Standard CV for all models. The LGBM wrapper handles early stopping internally.
        pipe = Pipeline([("features", clone(feature_pipe)), ("model", model)])
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=get_scorers(),
            return_train_score=False, n_jobs=-1
        )
        cv_rmse = -scores["test_neg_rmse"].mean()
        return cv_rmse

    return objective


def tune_top_models(
    top_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_pipe: Pipeline,
    n_trials: int = 40
) -> Tuple[Dict[str, dict], Dict[str, list]]:
    tuned = {}
    histories = {}
    for name in top_names:
        if not HAS_OPTUNA:
            print("[tune] Optuna not installed; skipping tuning for", name)
            continue
        if name == "XGB" and not HAS_XGB:
            print("[tune] Skipping XGB (xgboost not installed).")
            continue
        if name == "CatBoost" and not HAS_CAT:
            print("[tune] Skipping CatBoost (catboost not installed).")
            continue
        if name == "LGBM" and not HAS_LGBM:
            print("[tune] Skipping LGBM (lightgbm not installed).")
            continue

        print(f"[tune] Tuning {name} for {n_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(name, X_train, y_train, feature_pipe), n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
        histories[name] = [(t.number, t.value) for t in study.trials]

        # Rebuild estimator with best params
        if name == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=best_params.get("n_estimators", 800),
                max_depth=best_params.get("max_depth", None),
                min_samples_split=best_params.get("min_samples_split", 2),
                min_samples_leaf=best_params.get("min_samples_leaf", 1),
                max_features=best_params.get("max_features", "sqrt"),
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        elif name == "ElasticNet":
            model = ElasticNet(
                alpha=best_params.get("alpha", 0.01),
                l1_ratio=best_params.get("l1_ratio", 0.5),
                max_iter=30000,
                random_state=RANDOM_STATE
            )
        elif name == "Ridge":
            model = Ridge(alpha=best_params.get("alpha", 10.0), random_state=RANDOM_STATE)
        elif name == "Lasso":
            model = Lasso(alpha=best_params.get("alpha", 0.0005), max_iter=30000, random_state=RANDOM_STATE)
        elif name == "SVR":
            model = SVR(
                kernel="rbf",
                C=best_params.get("C", 10.0),
                epsilon=best_params.get("epsilon", 0.1),
                gamma=best_params.get("gamma", "scale"),
            )
        elif name == "XGB" and HAS_XGB:
            model = xgb.XGBRegressor(
                n_estimators=6000,
                learning_rate=best_params.get("learning_rate", 0.03),
                max_depth=best_params.get("max_depth", 4),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                min_child_weight=best_params.get("min_child_weight", 2.0),
                reg_lambda=best_params.get("reg_lambda", 3.0),
                reg_alpha=best_params.get("reg_alpha", 0.2),
                gamma=best_params.get("gamma", 0.05),
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                max_bin=best_params.get("max_bin", 256),
                missing=np.nan
            )
        elif name == "CatBoost" and HAS_CAT:
            model = CatBoostRegressor(
                loss_function="RMSE",
                n_estimators=best_params.get("n_estimators", 3000),
                depth=best_params.get("depth", 6),
                learning_rate=best_params.get("learning_rate", 0.05),
                l2_leaf_reg=best_params.get("l2_leaf_reg", 3.0),
                subsample=best_params.get("subsample", 0.8),
                colsample_bylevel=best_params.get("colsample_bylevel", 0.8),
                random_state=RANDOM_STATE,
                verbose=False
            )
        elif name == "LGBM" and HAS_LGBM:
            max_depth = best_params.get("max_depth", 12)
            num_leaves = _guard_num_leaves(best_params.get("num_leaves", 31), max_depth)
            model = LGBMRegressorWithEarlyStopping(
                max_n_estimators=best_params.get("max_n_estimators", 3000),
                learning_rate=best_params.get("learning_rate", 0.03),
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=best_params.get("min_child_samples", 20),
                min_child_weight=best_params.get("min_child_weight", 1e-3),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                reg_alpha=best_params.get("reg_alpha", 0.1),
                reg_lambda=best_params.get("reg_lambda", 1.0),
                min_split_gain=best_params.get("min_split_gain", 0.0),
                early_stopping_rounds=150,
                val_size=0.15,
                random_state=RANDOM_STATE
            )
        else:
            continue

        tuned[name] = {
            "estimator": model,
            "best_cv_rmse": best_score,
            "best_params": best_params,
        }
        print(f"[tune] {name}: best CV RMSE={best_score:.5f}, best_params={best_params}")
    return tuned, histories


# ---------------------------
# Stacking
# ---------------------------
def build_stacking_from_names(
    names: List[str],
    tuned: Dict[str, dict],
    base_models: Dict[str, object],
    feature_pipe: Pipeline
) -> Pipeline:
    # Use tuned estimator if available, else baseline default
    estimators = []
    for n in names:
        if n in tuned:
            est = tuned[n]["estimator"]
        else:
            est = base_models[n]
        estimators.append((n, est))
    meta = RidgeCV(alphas=np.logspace(-3, 3, 25))
    stack = Pipeline([
        ("features", clone(feature_pipe)),
        ("stack", StackingRegressor(estimators=estimators, final_estimator=meta, n_jobs=-1, passthrough=False))
    ])
    return stack


def fit_and_eval_stack(
    stack: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    do_cv: bool = True
):
    cv_rmse = np.nan
    cv_r2 = np.nan
    if do_cv:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(stack, X_train, y_train, cv=cv, scoring=get_scorers(), return_train_score=False, n_jobs=-1)
        cv_rmse = -scores["test_neg_rmse"].mean()
        cv_r2 = scores["test_r2"].mean()

    # In StackingRegressor.fit we cannot pass eval_set
    stack.fit(X_train, y_train)
    yp = stack.predict(X_test)
    test_rmse = rmse(y_test, yp)
    test_r2 = r2_score(y_test, yp)
    return cv_rmse, cv_r2, test_rmse, test_r2, yp


# ---------------------------
# Main
# ---------------------------
def main():
    # Load data
    df = pd.read_csv("train-house-prices-advanced-regression-techniques.csv")
    assert "SalePrice" in df.columns, "SalePrice column missing."
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    feature_pipe = build_feature_pipeline(X_train, X_test)

    # 1) Baseline evaluation of all requested models
    models = base_models_dict()
    baseline_df, preds_test = evaluate_models(models, X_train, y_train, X_test, y_test, feature_pipe, out_prefix="baseline")

    # Select top-5 by CV RMSE
    top5 = list(baseline_df.sort_values("cv_rmse_mean")["model"].head(5).values)
    print("Top-5 by CV RMSE:", top5)

    # 2) Tune top-5 with Optuna if available
    tuned, histories = tune_top_models(top5, X_train, y_train, feature_pipe, n_trials=40)

    # Save tuning histories
    with open("tuning_histories.json", "w") as f:
        json.dump(histories, f)

    # Build all 3-model combinations from top-5
    combos = list(itertools.combinations(top5, 3))
    print(f"Evaluating {len(combos)} stack combinations from top-5...")

    base_models = base_models_dict()  # for fallback estimators

    stack_records = []
    stack_preds = {}

    for names in combos:
        stack = build_stacking_from_names(names, tuned, base_models, feature_pipe)
        cv_rmse, cv_r2, test_rmse, test_r2, yp = fit_and_eval_stack(stack, X_train, y_train, X_test, y_test, do_cv=True)
        combo_name = "+".join(names)
        print(f"[STACK {combo_name}] CV RMSE={cv_rmse:.5f}  CV R2={cv_r2:.5f} | Test RMSE={test_rmse:.5f}  Test R2={test_r2:.5f}")
        stack_records.append({
            "stack": combo_name,
            "cv_rmse": cv_rmse,
            "cv_r2": cv_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })
        # Save predictions for this stack
        col_name = f"Stack_{combo_name}"
        stack_preds[col_name] = yp

    stacks_df = pd.DataFrame(stack_records).sort_values("test_rmse")
    stacks_df.to_csv("stacking_grid_results.csv", index=False)

    # Pick best stack by Test RMSE
    best_row = stacks_df.iloc[0]
    best_combo = best_row["stack"].split("+")
    print("Best stack (by Test RMSE):", dict(best_row))

    # Rebuild best stack and fit for final plots
    best_stack = build_stacking_from_names(best_combo, tuned, base_models, feature_pipe)
    _, _, best_test_rmse, best_test_r2, best_pred = fit_and_eval_stack(best_stack, X_train, y_train, X_test, y_test, do_cv=False)

    # Plot predicted vs true for best stack
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test.values, best_pred, alpha=0.6, edgecolors="none")
    lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("True SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Best Stack ({'+'.join(best_combo)}): True vs Predicted")
    plt.tight_layout()
    plt.savefig("stack_true_vs_pred_best.png", dpi=150)
    plt.close()

    # Compare bars: best single vs best stack
    best_single = baseline_df.sort_values("test_rmse").iloc[0]
    names_plot = [best_single["model"], f"Stack({'+'.join(best_combo)})"]
    vals_rmse = [best_single["test_rmse"], best_test_rmse]
    vals_r2 = [best_single["test_r2"], best_test_r2]

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(names_plot)), vals_rmse)
    plt.xticks(np.arange(len(names_plot)), names_plot, rotation=15, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("Best single vs Best stack (Test RMSE)")
    plt.tight_layout()
    plt.savefig("best_single_vs_best_stack_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(names_plot)), vals_r2)
    plt.xticks(np.arange(len(names_plot)), names_plot, rotation=15, ha="right")
    plt.ylabel("Test R2")
    plt.title("Best single vs Best stack (Test R2)")
    plt.tight_layout()
    plt.savefig("best_single_vs_best_stack_r2.png", dpi=150)
    plt.close()

    # Save all test predictions (including stacks)
    out_preds = preds_test.copy()
    out_preds.update(stack_preds)
    out_preds["Stack"] = best_pred
    out_preds_df = pd.DataFrame(out_preds)
    if "Id" in df.columns:
        out_preds_df.insert(0, "Id", df.loc[X_test.index, "Id"].values)
    out_preds_df.to_csv("test_predictions_all_models.csv", index=False)

    # Save summary
    summary = pd.read_csv("baseline_model_cv_test_results.csv")
    stack_row = {
        "model": "STACK(" + "+".join(best_combo) + ")",
        "cv_rmse_mean": np.nan,
        "cv_rmse_std": np.nan,
        "cv_r2_mean": np.nan,
        "cv_r2_std": np.nan,
        "test_rmse": best_test_rmse,
        "test_r2": best_test_r2,
    }
    final_summary = pd.concat([summary, pd.DataFrame([stack_row])], ignore_index=True)
    final_summary.to_csv("final_summary.csv", index=False)

    print("Artifacts written:")
    print(" - baseline_model_cv_test_results.csv")
    print(" - baseline_cv_rmse.png, baseline_cv_r2.png")
    print(" - baseline_test_rmse.png, baseline_test_r2.png")
    print(" - tuning_histories.json (if tuning ran)")
    print(" - stacking_grid_results.csv  (all 3-model combos from top-5)")
    print(" - stack_true_vs_pred_best.png")
    print(" - best_single_vs_best_stack_rmse.png, best_single_vs_best_stack_r2.png")
    print(" - test_predictions_all_models.csv")
    print(" - final_summary.csv")


if __name__ == "__main__":
    main()
