# src/modeling/trainer.py

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
    Orchestrator cao nhất, ghép các mixin thành một trainer hoàn chỉnh.
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
        """
        Quy trình full:

        1. Train các default models.
        2. Chọn top K theo CV RMSE.
        3. Tuning hyperparameter cho top K models.
        4. Dùng tuned models để build 3 model stacks và Optuna tune stack.
        5. Lưu kết quả.
        """
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
