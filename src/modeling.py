"""
modeling.py

Machine learning models.

The ModelTrainer class is responsible for:
- Loading data from CSV
- Splitting train and test sets
- Building the preprocessing pipeline (delegated to Preprocessor)
- Training multiple baseline models
- Hyperparameter tuning with Optuna (and optionally GridSearchCV)
- Evaluating models
- Saving and loading fitted models (.joblib)
- Saving model comparison results to CSV and plotting comparison charts
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR

import optuna
import logging

from src.preprocessing import Preprocessor, build_feature_pipeline


class ModelTrainer:
    """
    High level manager for the full regression model workflow.

    Typical usage order:
    - load_data
    - split_data
    - build_preprocessing
    - train_default_models
    - tune_model_optuna (optional)
    - save_results, save_model, load_model
    """

    def __init__(
        self,
        target_col: str = "SalePrice",
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "model_outputs",
        log_level: int = logging.INFO,
    ):
        """
        Initialize the ModelTrainer.

        Parameters
        ----------
        target_col:
            Name of the target column in the dataset.
        test_size:
            Fraction of data to keep for the test set in train_test_split.
        random_state:
            Random seed used for splitting and for models that support it.
        output_dir:
            Directory where logs, results, and saved models will be stored.
        log_level:
            Logging level for the internal logger.
        """
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # logger
        self.logger = self._build_logger(log_level=log_level)

        # Data containers (populated step by step in the workflow)
        self.df_: Optional[pd.DataFrame] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None

        # Preprocessing pipeline and model registry
        self.feature_pipe_: Optional[Pipeline] = None
        self.models_: Dict[str, Pipeline] = {}
        self.results_: Dict[str, Dict] = {}

        # Delegate preprocessing logic to Preprocessor
        self.dp = Preprocessor(target_col=self.target_col)

    def _build_logger(self, log_level: int) -> logging.Logger:
        """
        Build a logger that always writes to output_dir/training.log.

        The logger:
        - Is named "ModelTrainer"
        - Writes to a single file handler
        - Avoids attaching duplicate handlers if called multiple times
        """
        logger = logging.getLogger("ModelTrainer")

        # Avoid adding multiple handlers when multiple instances are created
        if logger.handlers:
            logger.setLevel(log_level)
            return logger

        logger.setLevel(log_level)

        log_path = Path(self.output_dir) / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def get_default_models(random_state: int) -> Dict[str, object]:
        """
        Return a dictionary of baseline models that will be trained by default.

        A staticmethod is used here to satisfy the technical requirement
        in the assignment and to make the default model set reusable.
        """
        return {
            "ridge": Ridge(alpha=1.0, random_state=random_state),
            "lasso": Lasso(alpha=0.001, random_state=random_state),
            "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state),
            "random_forest": RandomForestRegressor(n_estimators=400, random_state=random_state),
            "svr": SVR(C=5.0, gamma="scale"),
        }

    # 1) Load data
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load raw data from a CSV file using the delegated Preprocessor.

        Parameters
        ----------
        csv_path:
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataframe.
        """
        self.logger.info(f"Loading data from {csv_path}")
        self.df_ = self.dp.load_data(csv_path)
        self.logger.info(f"Data loaded with shape {self.df_.shape}")
        return self.df_

    # 2) Split train and test
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the loaded dataframe into train and test sets.

        Returns
        -------
        X_train, X_test, y_train, y_test

        Raises
        ------
        RuntimeError
            If data has not been loaded before calling this method.
        """
        if self.df_ is None:
            raise RuntimeError("No data loaded. Call load_data first.")
        X, y = self.dp.split_features_target(self.df_)

        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.logger.info(
            "Split data into train and test sets: "
            f"train={self.X_train_.shape[0]} rows, "
            f"test={self.X_test_.shape[0]} rows"
        )
        return self.X_train_, self.X_test_, self.y_train_, self.y_test_  # double check shapes if needed

    # 3) Build preprocessing
    def build_preprocessing(self) -> Pipeline:
        """
        Build the feature preprocessing pipeline using the training data.

        Returns
        -------
        Pipeline
            A sklearn Pipeline that transforms raw features into numeric inputs.
        """
        if self.X_train_ is None or self.X_test_ is None:
            raise RuntimeError("Call split_data before build_preprocessing.")
        self.feature_pipe_ = build_feature_pipeline(self.X_train_, self.X_test_)
        self.logger.info("Feature preprocessing pipeline built successfully.")
        return self.feature_pipe_

    # 4) Evaluate a fitted model
    def evaluate_model(self, name: str, pipe: Pipeline) -> Tuple[float, float]:
        """
        Evaluate a fitted model on the held out test set.

        Parameters
        ----------
        name:
            Name of the model, used as key in the results dictionary.
        pipe:
            A fitted Pipeline that includes preprocessing and the estimator.

        Returns
        -------
        rmse, r2
            Root Mean Squared Error and R squared on the test set.
        """
        if self.X_test_ is None or self.y_test_ is None:
            raise RuntimeError("Test data not available. Call split_data first.")

        y_pred = pipe.predict(self.X_test_)

        # Compute MSE and then take the square root to obtain RMSE
        mse = mean_squared_error(self.y_test_, y_pred)
        rmse = np.sqrt(mse)

        r2 = r2_score(self.y_test_, y_pred)
        self.results_[name] = {"rmse": rmse, "r2": r2}
        self.logger.info(f"Evaluation for {name}: RMSE={rmse:.4f}, R2={r2:.4f}")
        return rmse, r2

    # 5) Train a single model
    def train_single_model(self, name: str, estimator) -> Tuple[float, float]:
        """
        Train a single model wrapped in a pipeline and evaluate it.

        Parameters
        ----------
        name:
            Model name used in internal registries and logs.
        estimator:
            An sklearn estimator instance.

        Returns
        -------
        rmse, r2
            Evaluation metrics on the test set.
        """
        if self.feature_pipe_ is None:
            raise RuntimeError("Call build_preprocessing before training any model.")

        pipe = Pipeline(
            [
                ("features", self.feature_pipe_),
                ("model", estimator),
            ]
        )
        self.logger.info(f"Training model '{name}'")
        pipe.fit(self.X_train_, self.y_train_)
        self.models_[name] = pipe

        return self.evaluate_model(name, pipe)

    # 6) Train multiple default models
    def train_default_models(self) -> Dict[str, Dict]:
        """
        Train and evaluate all default candidate models.

        Returns
        -------
        Dict[str, Dict]
            A mapping from model name to its evaluation metrics.
        """
        self.logger.info("Training default candidate models.")
        candidates = self.get_default_models(self.random_state)
        for name, est in candidates.items():
            self.logger.info(f"Starting training for default model '{name}'")
            self.train_single_model(name, est)
        self.logger.info("Finished training all default models.")
        return self.results_

    # 7) Hyperparameter tuning with Optuna
    def tune_model_optuna(self, base_name: str, n_trials: int = 30):
        """
        Tune hyperparameters of a supported model using Optuna.

        Currently supported base models:
        - "random_forest"
        - "elasticnet"

        Parameters
        ----------
        base_name:
            Name of the base model to tune.
        n_trials:
            Number of Optuna trials.

        Returns
        -------
        best_params, rmse, r2
            Best hyperparameters and performance of the tuned model.
        """
        if self.feature_pipe_ is None:
            raise RuntimeError("Call build_preprocessing before tuning models.")

        X_train = self.X_train_
        y_train = self.y_train_

        self.logger.info(
            f"Start Optuna tuning for base model '{base_name}' "
            f"with n_trials={n_trials}."
        )

        def objective(trial):
            if base_name == "random_forest":
                n_estimators = trial.suggest_int("n_estimators", 100, 600)
                max_depth = trial.suggest_int("max_depth", 4, 20)
                est = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                )
            elif base_name == "elasticnet":
                alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
                l1 = trial.suggest_float("l1_ratio", 0.1, 0.9)
                est = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=self.random_state)
            else:
                raise ValueError(
                    "Optuna tuning currently supports only 'random_forest' and 'elasticnet'."
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
                scoring="neg_root_mean_squared_error",
                cv=5,
            )
            # cross_validate returns negative RMSE, so we negate again
            return -np.mean(scores["test_score"])

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.logger.info(f"Best Optuna parameters for '{base_name}': {study.best_params}")

        # Refit the best model on full training data
        if base_name == "random_forest":
            best_est = RandomForestRegressor(
                **study.best_params,
                random_state=self.random_state,
            )
        else:
            best_est = ElasticNet(
                **study.best_params,
                random_state=self.random_state,
            )

        tuned_name = f"{base_name}_tuned"
        self.logger.info(f"Training tuned model '{tuned_name}' with best Optuna parameters.")
        rmse, r2 = self.train_single_model(tuned_name, best_est)
        return study.best_params, rmse, r2

    # 8) Optional: GridSearchCV for a single model
    def tune_model_gridsearch(self, base_name: str, param_grid: Dict):
        """
        Run a classic GridSearchCV on top of one of the default models.

        Parameters
        ----------
        base_name:
            Name of the base model in get_default_models.
        param_grid:
            Dictionary of parameters to search. This will be prefixed with
            'model__' automatically to match the pipeline.

        Returns
        -------
        best_params, rmse, r2
            Best parameters from GridSearchCV and performance of the best model.
        """
        if self.feature_pipe_ is None:
            raise RuntimeError("Call build_preprocessing before GridSearchCV.")

        base_models = self.get_default_models(self.random_state)
        if base_name not in base_models:
            raise ValueError(f"Unknown base_name '{base_name}' in default models.")

        est = base_models[base_name]
        pipe = Pipeline(
            [
                ("features", self.feature_pipe_),
                ("model", est),
            ]
        )

        grid = {f"model__{k}": v for k, v in param_grid.items()}

        self.logger.info(f"Start GridSearchCV for base model '{base_name}'.")
        search = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        search.fit(self.X_train_, self.y_train_)

        self.logger.info(
            f"Best GridSearchCV parameters for '{base_name}': {search.best_params}"
        )
        best_pipe = search.best_estimator_
        name = f"{base_name}_grid"
        self.models_[name] = best_pipe
        rmse, r2 = self.evaluate_model(name, best_pipe)
        return search.best_params_, rmse, r2

    # 9) Save a fitted model
    def save_model(self, name: str):
        """
        Save a fitted model pipeline to disk as a .joblib file.

        Parameters
        ----------
        name:
            Name of the model that has already been trained.
        """
        if name not in self.models_:
            raise ValueError(f"Model '{name}' does not exist in registry.")
        path = self.output_dir / f"{name}.joblib"
        joblib.dump(self.models_[name], path)
        self.logger.info(f"Saved model '{name}' to {path}")

    # 10) Load a fitted model
    def load_model(self, path: str, name: Optional[str] = None):
        """
        Load a model pipeline from a .joblib file and register it.

        Parameters
        ----------
        path:
            Path to the .joblib file.
        name:
            Optional custom name for the loaded model. If None, the filename
            (without extension) will be used.

        Returns
        -------
        The loaded model pipeline.
        """
        model = joblib.load(path)
        if name is None:
            name = Path(path).stem
        self.models_[name] = model
        self.logger.info(f"Loaded model '{name}' from {path}")
        return model

    # 11) Save model comparison results
    def save_results(self):
        """
        Save evaluation results to CSV and plot a RMSE comparison bar chart.

        Outputs
        -------
        - model_results.csv
        - rmse_comparison.png
        inside the configured output directory.
        """
        if not self.results_:
            self.logger.warning("No model results to save. Skipping save_results.")
            return

        df = pd.DataFrame(self.results_).T
        csv_path = self.output_dir / "model_results.csv"
        df.to_csv(csv_path)
        self.logger.info(f"Saved model comparison results to {csv_path}")

        plt.figure(figsize=(8, 5))
        plt.bar(df.index, df["rmse"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("RMSE")
        plt.title("Model RMSE comparison")
        plt.tight_layout()
        fig_path = self.output_dir / "rmse_comparison.png"
        plt.savefig(fig_path)
        plt.close()
        self.logger.info(f"Saved RMSE comparison plot to {fig_path}")

    # 12) End to end convenience runner
    def run(self, csv_path: str, tune_optuna: bool = True):
        """
        Run the full training pipeline end to end.

        Steps
        -----
        - Load data
        - Split train and test
        - Build preprocessing pipeline
        - Train default models
        - Optionally run Optuna tuning for random_forest
        - Save comparison results

        Parameters
        ----------
        csv_path:
            Path to the training CSV file.
        tune_optuna:
            If True, runs Optuna tuning for the random_forest model.

        Returns
        -------
        Dict[str, Dict]
            Evaluation metrics for all trained models.
        """
        self.logger.info("Starting full training pipeline.")
        self.load_data(csv_path)
        self.split_data()
        self.build_preprocessing()
        self.train_default_models()

        if tune_optuna:
            self.logger.info("Running Optuna tuning for 'random_forest'.")
            self.tune_model_optuna("random_forest", n_trials=20)
        else:
            self.logger.info("Skipping Optuna tuning step.")

        self.save_results()
        self.logger.info("Full training pipeline completed.")
        return self.results_


__all__ = ["ModelTrainer"]
