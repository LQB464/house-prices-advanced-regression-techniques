# src/modeling/stacking_mixin.py

from typing import Dict, List, Tuple

import itertools
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score

from .metrics import _rmse, _get_scorers

from .tuning_mixin import HAS_OPTUNA

if HAS_OPTUNA:
    import optuna


class StackingMixin:
    """
    Tuning stacking ensemble 3 model bằng Optuna.

    Yêu cầu:
        self.models_, self.results_, self.feature_pipe_
        self.X_train_, self.y_train_, self.X_test_, self.y_test_
        self.output_dir, self.random_state, self.logger
        self.save_model
    """

    def _optuna_stack_objective(
        self,
        trial,
        tuned_model_names: List[str],
        cv_splits: int = 5,
    ) -> float:
        if self.feature_pipe_ is None:
            raise RuntimeError("Feature pipeline not built.")
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Training data not available.")

        if len(tuned_model_names) < 3:
            raise ValueError("Need at least 3 tuned models for stacking.")

        combo_list = list(itertools.combinations(tuned_model_names, 3))
        combo = trial.suggest_categorical("stack_combo", combo_list)

        meta_alpha = trial.suggest_float("meta_alpha", 1e-4, 10.0, log=True)
        meta_l1_ratio = trial.suggest_float("meta_l1_ratio", 0.0, 1.0)
        passthrough = trial.suggest_categorical("passthrough", [False, True])

        base_estimators = []
        for name in combo:
            if name not in self.models_:
                raise ValueError(f"Model '{name}' not found in self.models_.")
            pipe = self.models_[name]
            est = clone(pipe.named_steps["model"])
            base_estimators.append((name, est))

        meta_enet = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "enet",
                    ElasticNet(
                        alpha=meta_alpha,
                        l1_ratio=meta_l1_ratio,
                        max_iter=50_000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        stack_reg = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_enet,
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

        return float(cv_rmse_mean)

    def tune_stacking_with_optuna(
        self,
        tuned_model_names: List[str],
        n_trials: int = 20,
        cv_splits: int = 5,
    ) -> Tuple[str, Dict, Dict[str, float]]:
        """
        Dùng Optuna để tìm combo 3 model cùng meta parameters tốt nhất.
        """
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna is not installed.")
        if self.feature_pipe_ is None:
            raise RuntimeError("Feature pipeline not built.")

        self.logger.info(
            f"Start Optuna tuning for stacking with candidates: {tuned_model_names}"
        )

        if len(tuned_model_names) < 3:
            raise ValueError("Need at least 3 tuned models for stacking.")

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state,
            n_startup_trials=10,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        )

        def objective(trial: optuna.Trial) -> float:
            return self._optuna_stack_objective(
                trial=trial,
                tuned_model_names=tuned_model_names,
                cv_splits=cv_splits,
            )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_combo = best_params["stack_combo"]

        # Lưu lại toàn bộ trials
        df_trials = study.trials_dataframe()
        df_trials.to_csv(self.output_dir / "stacking_optuna_trials.csv", index=False)

        # Summary theo combo
        df_trials["stack_combo"] = df_trials["params_stack_combo"]
        summary = (
            df_trials.groupby("stack_combo")["value"]
            .agg(["count", "min", "mean"])
            .reset_index()
            .rename(
                columns={
                    "count": "n_trials",
                    "min": "best_rmse",
                    "mean": "avg_rmse",
                }
            )
        )
        summary.to_csv(self.output_dir / "stacking_optuna_summary.csv", index=False)

        self.logger.info(
            f"[STACK] Best combo={best_combo} best_cv_rmse={study.best_value:.4f}"
        )

        # Xây stack tốt nhất và train final
        base_estimators = []
        for name in best_combo:
            pipe = self.models_[name]
            est = clone(pipe.named_steps["model"])
            base_estimators.append((name, est))

        stack_reg = StackingRegressor(
            estimators=base_estimators,
            final_estimator=ElasticNet(
                alpha=best_params["meta_alpha"],
                l1_ratio=best_params["meta_l1_ratio"],
                max_iter=50_000,
                random_state=self.random_state,
            ),
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

        y_pred = stack_pipe.predict(self.X_test_)
        test_rmse = _rmse(self.y_test_, y_pred)
        test_r2 = float(r2_score(self.y_test_, y_pred))

        metrics = {
            "cv_rmse_mean": float(study.best_value),
            "cv_rmse_std": float(np.nan),
            "cv_r2_mean": float(np.nan),
            "cv_r2_std": float(np.nan),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
        }

        self.results_[model_name] = metrics

        try:
            self.save_model(model_name)
        except Exception as e:
            self.logger.warning(
                f"Failed to save stacking model '{model_name}': {e}"
            )

        return model_name, best_params, metrics
