# src/modeling/__init__.py
from .trainer import ModelTrainer
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


__all__ = ["ModelTrainer"]
