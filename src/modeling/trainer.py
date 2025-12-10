"""
modeling/trainer.py

High-level orchestrator that composes all mixins into a unified end-to-end
model training, tuning, evaluation, and persistence workflow.

Extended Description
--------------------
This module defines the `ModelTrainer`, which inherits functionality from:
- TrainerConfig (global config, logging, runtime state)
- DataMixin (data loading and splitting)
- PreprocessingMixin (feature pipeline construction)
- DefaultModelsMixin (baseline model training and evaluation)
- TuningMixin (hyperparameter tuning with Optuna for individual models)
- StackingMixin (Optuna-driven stacking ensemble search)
- PersistenceMixin (saving models, exporting results)

`ModelTrainer` integrates all these responsibilities into a single, unified
interface that enables:
- end-to-end training and model selection
- automated stacking ensemble tuning
- consistent logging and output management
- streamlined experiment reproducibility

Main Components
---------------
- ModelTrainer:
  - run_full_model_selection_and_stacking: executes the entire workflow

Usage Example
-------------
>>> trainer = ModelTrainer(target_col="SalePrice")
>>> trainer.load_data("train.csv")
>>> trainer.split_data()
>>> trainer.build_preprocessing()
>>> results = trainer.run_full_model_selection_and_stacking(top_k=5)

Notes
-----
All mixins depend on the internal runtime state initialized by TrainerConfig.
This class simply exposes a cohesive interface to perform all steps in order.
"""

from typing import Dict, List

import logging

from .config import TrainerConfig
from .data_mixin import DataMixin
from .preprocessing_mixin import PreprocessingMixin
from .default_models_mixin import DefaultModelsMixin
from .tuning_mixin import TuningMixin
from .stacking_mixin import StackingMixin
from .persistence_mixin import PersistenceMixin


class ModelTrainer(
    TrainerConfig,
    DataMixin,
    PreprocessingMixin,
    DefaultModelsMixin,
    TuningMixin,
    StackingMixin,
    PersistenceMixin,
):
    """
    Unified orchestrator combining all mixins into a full training pipeline.

    Extended Description
    --------------------
    The `ModelTrainer` ties together all core components required for:
    - loading and splitting data
    - building preprocessing pipelines
    - training and evaluating baseline models
    - tuning top models via Optuna
    - performing stacking ensemble search and training
    - saving both models and evaluation artifacts

    The class inherits behavior from several mixins, each responsible for
    a distinct stage of the ML workflow. `TrainerConfig` initializes all shared
    state, such as the logger, output directory, and dataset containers.

    Parameters
    ----------
    target_col : str, default "SalePrice"
        Name of the target column.
    test_size : float, default 0.2
        Fraction of data allocated to the test split.
    random_state : int, default 42
        Seed used across all randomized processes.
    output_dir : str, default "model_outputs"
        Directory where logs, models, and results are stored.
    log_level : int, default logging.INFO
        Logging verbosity level.

    Examples
    --------
    >>> trainer = ModelTrainer()
    >>> trainer.load_data("train.csv")
    >>> trainer.split_data()
    >>> trainer.build_preprocessing()
    >>> trainer.run_full_model_selection_and_stacking()
    """

    def __init__(
        self,
        target_col: str = "SalePrice",
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "model_outputs",
        log_level: int = logging.INFO,
    ) -> None:
        TrainerConfig.__init__(
            self,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            output_dir=output_dir,
            log_level=log_level,
        )

    def run_full_model_selection_and_stacking(
        self,
        top_k: int = 5,
        n_trials_model: int = 50,
        n_trials_stack: int = 20,
        cv_splits: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        
        self.logger.info("Starting full model selection and stacking pipeline.")

        # 1 and 2
        self.train_default_models(cv_splits=cv_splits)
        top_names = self.select_top_models(k=top_k, by="cv_rmse_mean")

        # 3
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

        # 4
        stack_name, stack_params, stack_metrics = self.tune_stacking_with_optuna(
            tuned_model_names=tuned_names,
            n_trials=n_trials_stack,
            cv_splits=cv_splits,
        )

        self.logger.info(
            f"Finished stacking. Best stack '{stack_name}' "
            f"test RMSE={stack_metrics['test_rmse']:.4f} "
            f"test R2={stack_metrics['test_r2']:.4f}"
        )

        # 5
        self.save_results()
        return self.results_


__all__ = ["ModelTrainer"]
