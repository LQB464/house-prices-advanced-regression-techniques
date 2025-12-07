# src/modeling/__init__.py
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from .trainer import ModelTrainer

__all__ = ["ModelTrainer"]
