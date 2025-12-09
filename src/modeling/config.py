# src/modeling/config.py

from pathlib import Path
from typing import Optional
import logging

import pandas as pd

from .preprocessing import Preprocessor


class TrainerConfig:
    """
    Quản lý config, logger và state chung cho toàn bộ trainer.

    Các mixin khác sẽ giả định những thuộc tính sau tồn tại:
        - target_col, test_size, random_state, output_dir, logger
        - df_, X_train_, X_test_, y_train_, y_test_
        - feature_pipe_
        - models_, results_
        - dp (Preprocessor)
    """

    def __init__(
        self,
        target_col: str = "SalePrice",
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "model_outputs",
        log_level: int = logging.INFO,
    ) -> None:
        self.target_col = target_col
        self.test_size = float(test_size)
        self.random_state = int(random_state)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger(log_level=log_level)

        # Data containers
        self.df_: Optional[pd.DataFrame] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None

        # Preprocessing pipeline
        self.feature_pipe_ = None

        # Model registry và kết quả
        self.models_ = {}
        self.results_ = {}

        # Data preprocessor helper
        self.dp = Preprocessor(target_col=self.target_col)

    def _build_logger(self, log_level: int = logging.INFO) -> logging.Logger:
        """
        Tạo logger ghi ra file training.log và in ra console.
        """
        logger = logging.getLogger("modeling.ModelTrainer")
        logger.setLevel(log_level)

        if logger.handlers:
            # Đã được tạo trước đó
            return logger

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        log_path = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        return logger
