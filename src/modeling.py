"""
model_trainer.py

Phan 2: Mo hinh hoc may

Lop ModelTrainer chiu trach nhiem:
- Nap du lieu (tu file CSV)
- Tach train/test
- Xay dung pipeline tien xu ly (goi Preprocessor)
- Huan luyen nhieu mo hinh
- Toi uu sieu tham so (Optuna, co them GridSearchCV optional)
- Danh gia mo hinh
- Luu va nap lai mo hinh (.joblib)
- Luu ket qua so sanh model ra CSV va ve bieu do
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

from preprocessing import Preprocessor, build_feature_pipeline


class ModelTrainer:
    """
    Lop quan ly toan bo quy trinh mo hinh hoc may cho bai toan regression.

    Thu tu su dung:
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
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # logger
        self.logger = self._build_logger()

        self.df_: Optional[pd.DataFrame] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None

        self.feature_pipe_: Optional[Pipeline] = None
        self.models_: Dict[str, Pipeline] = {}
        self.results_: Dict[str, Dict] = {}

        self.dp = Preprocessor(target_col=self.target_col)

    def _build_logger(self) -> logging.Logger:
        """
        Logger luôn ghi vào output_dir/training.log
        (output_dir được truyền từ main.py và là thư mục gốc).
        """
        logger = logging.getLogger("ModelTrainer")

        # tránh việc thêm 2 lần handler
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        log_path = Path(self.output_dir) / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        return logger

    @staticmethod
    def get_default_models(random_state: int) -> Dict[str, object]:
        """
        Tra ve dictionary cac mo hinh mac dinh se huan luyen.
        Dung @staticmethod de dap ung yeu cau ky thuat Python trong de.
        """
        return {
            "ridge": Ridge(alpha=1.0, random_state=random_state),
            "lasso": Lasso(alpha=0.001, random_state=random_state),
            "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state),
            "random_forest": RandomForestRegressor(n_estimators=400, random_state=random_state),
            "svr": SVR(C=5.0, gamma="scale"),
        }

    # 1) Nap du lieu
    def load_data(self, csv_path: str) -> pd.DataFrame:
        self.logger.info(f"Loading data from {csv_path}")
        self.df_ = self.dp.load_data(csv_path)
        return self.df_

    # 2) Tach train/test
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.df_ is None:
            raise RuntimeError("Chua co du lieu. Hay goi load_data truoc.")
        X, y = self.dp.split_features_target(self.df_)

        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.logger.info(
            f"Split data: train {self.X_train_.shape[0]} rows, "
            f"test {self.X_test_.shape[0]} rows"
        )
        return self.X_train_, self.X_test_, self.y_train_, self.y_test_

    # 3) Xay dung preprocessing
    def build_preprocessing(self) -> Pipeline:
        if self.X_train_ is None or self.X_test_ is None:
            raise RuntimeError("Chua split_data truoc khi build_preprocessing.")
        self.feature_pipe_ = build_feature_pipeline(self.X_train_, self.X_test_)
        self.logger.info("Built feature preprocessing pipeline.")
        return self.feature_pipe_

    # 4) Danh gia 1 mo hinh da fit
    def evaluate_model(self, name: str, pipe: Pipeline) -> Tuple[float, float]:
        y_pred = pipe.predict(self.X_test_)
        # Tinh MSE binh thuong, sau do lay can bac 2 de ra RMSE
        mse = mean_squared_error(self.y_test_, y_pred)
        rmse = np.sqrt(mse)

        r2 = r2_score(self.y_test_, y_pred)
        self.results_[name] = {"rmse": rmse, "r2": r2}
        self.logger.info(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")
        return rmse, r2

    # 5) Huan luyen 1 mo hinh
    def train_single_model(self, name: str, estimator) -> Tuple[float, float]:
        if self.feature_pipe_ is None:
            raise RuntimeError("Chua build_preprocessing truoc khi train model.")

        pipe = Pipeline([
            ("features", self.feature_pipe_),
            ("model", estimator),
        ])
        self.logger.info(f"Training model {name}")
        pipe.fit(self.X_train_, self.y_train_)
        self.models_[name] = pipe

        return self.evaluate_model(name, pipe)

    # 6) Huan luyen nhieu mo hinh mac dinh
    def train_default_models(self) -> Dict[str, Dict]:
        candidates = self.get_default_models(self.random_state)
        for name, est in candidates.items():
            self.train_single_model(name, est)
        return self.results_

    # 7) Tuning Optuna (toi uu tham so)
    def tune_model_optuna(self, base_name: str, n_trials: int = 30):
        if self.feature_pipe_ is None:
            raise RuntimeError("Chua build_preprocessing truoc khi tune.")

        X_train = self.X_train_
        y_train = self.y_train_

        self.logger.info(f"Start Optuna tuning for {base_name} with {n_trials} trials.")

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
                raise ValueError("Optuna tuning moi ho tro random_forest hoac elasticnet.")

            pipe = Pipeline([
                ("features", self.feature_pipe_),
                ("model", est),
            ])
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=5,
            )
            return -np.mean(scores["test_score"])

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.logger.info(f"Best params for {base_name}: {study.best_params}")

        # train lai voi best params
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
        rmse, r2 = self.train_single_model(tuned_name, best_est)
        return study.best_params, rmse, r2

    # 8) Optional: GridSearchCV cho 1 model (de show cho de thay)
    def tune_model_gridsearch(self, base_name: str, param_grid: Dict):
        if self.feature_pipe_ is None:
            raise RuntimeError("Chua build_preprocessing truoc khi grid search.")

        base_models = self.get_default_models(self.random_state)
        if base_name not in base_models:
            raise ValueError("base_name khong ton tai trong default models.")

        est = base_models[base_name]
        pipe = Pipeline([
            ("features", self.feature_pipe_),
            ("model", est),
        ])

        grid = {
            f"model__{k}": v for k, v in param_grid.items()
        }

        self.logger.info(f"Start GridSearchCV for {base_name}")
        search = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        search.fit(self.X_train_, self.y_train_)

        self.logger.info(f"Best params (GridSearch) for {base_name}: {search.best_params}")
        best_pipe = search.best_estimator_
        name = f"{base_name}_grid"
        self.models_[name] = best_pipe
        rmse, r2 = self.evaluate_model(name, best_pipe)
        return search.best_params_, rmse, r2

    # 9) Luu model
    def save_model(self, name: str):
        if name not in self.models_:
            raise ValueError(f"Model {name} khong ton tai.")
        path = self.output_dir / f"{name}.joblib"
        joblib.dump(self.models_[name], path)
        self.logger.info(f"Saved model {name} to {path}")

    # 10) Nap model da luu
    def load_model(self, path: str, name: Optional[str] = None):
        model = joblib.load(path)
        if name is None:
            name = Path(path).stem
        self.models_[name] = model
        self.logger.info(f"Loaded model {name} from {path}")
        return model

    # 11) Luu ket qua so sanh model
    def save_results(self):
        if not self.results_:
            self.logger.warning("Khong co ket qua model nao de luu.")
            return

        df = pd.DataFrame(self.results_).T
        df.to_csv(self.output_dir / "model_results.csv")
        self.logger.info("Saved model_results.csv")

        plt.figure(figsize=(8, 5))
        plt.bar(df.index, df["rmse"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("RMSE")
        plt.title("So sanh RMSE cac mo hinh")
        plt.tight_layout()
        plt.savefig(self.output_dir / "rmse_comparison.png")
        plt.close()

    # 12) Ham run tong hop
    def run(self, csv_path: str, tune_optuna: bool = True):
        self.load_data(csv_path)
        self.split_data()
        self.build_preprocessing()
        self.train_default_models()

        if tune_optuna:
            # vi du: tune random_forest
            self.tune_model_optuna("random_forest", n_trials=20)

        self.save_results()
        return self.results_
