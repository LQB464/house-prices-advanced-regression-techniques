# preprocessing/preprocessor.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from .config import ORDINAL_MAP_CANONICAL
from .pipeline import build_feature_pipeline

@dataclass
class Preprocessor:
    """
    Lớp tiền xử lý dữ liệu cấp cao.

    Chức năng chính:
    - load_data: đọc csv/xlsx/json bằng pandas
    - split_features_target: tách X, y
    - build_feature_pipeline: tạo sklearn Pipeline cho X
    - fit_transform: fit pipeline trên tập train và trả về X_processed
    - transform_new: biến đổi tập dữ liệu mới (val/test)

    Parameters
    ----------
    target_col : str
        Tên cột target (label).
    ordinal_mapping : dict, optional
        mapping cho các cột ordinal:
            - có thể là Dict[str, List[str]] (canonical)
            - hoặc Dict[str, Dict[str, int/float]]
    id_cols : list, optional
        Các cột id cần loại bỏ khỏi X.
    use_domain_features : bool, default True
        Có thêm domain features (TotalSF, TotalBath, ...) hay không.
    use_target_encoding : bool, default False
        Có dùng Target Encoding cho một số cột categorical hay không.
    target_enc_cols : list, optional
        Danh sách cột áp dụng Target Encoding (nếu None sẽ dùng default).
    """

    target_col: Optional[str] = None
    ordinal_mapping: Optional[Mapping[str, Any]] = None
    id_cols: Optional[List[str]] = None
    use_domain_features: bool = True
    use_target_encoding: bool = False
    target_enc_cols: Optional[List[str]] = None

    # >>> thêm các tham số điều khiển feature selection <<<
    enable_variance_selector: bool = True
    variance_threshold: float = 0.0
    enable_kbest_mi: bool = False
    k_best_features: int = 150
    mi_random_state: int = 0

    # các thuộc tính được set sau
    df_raw_: Optional[pd.DataFrame] = None
    feature_pipe_: Optional[Pipeline] = None
    logs: List[str] = field(default_factory=list)

    # ----------------- phương thức tiện ích -----------------

    @staticmethod
    def _detect_file_type(path: str) -> str:
        path = path.lower()
        if path.endswith(".csv"):
            return "csv"
        if path.endswith(".xlsx") or path.endswith(".xls"):
            return "excel"
        if path.endswith(".json"):
            return "json"
        raise ValueError(
            "Không nhận diện được loại file. Hỗ trợ: .csv, .xlsx/.xls, .json"
        )

    def __repr__(self) -> str:
        ord_cols = (
            list(self.ordinal_mapping.keys()) if self.ordinal_mapping else []
        )
        return (
            f"Preprocessor(target_col={self.target_col!r}, "
            f"id_cols={self.id_cols}, "
            f"ordinal_cols={ord_cols}, "
            f"use_domain_features={self.use_domain_features}, "
            f"use_target_encoding={self.use_target_encoding})"
        )

    def _log(self, msg: str) -> None:
        """Ghi 1 dòng log vào bộ nhớ."""
        self.logs.append(msg)

    # ----------------- I/O -----------------

    def load_data(self, path: str, **read_kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ csv/xlsx/json bằng pandas.
        """
        ftype = self._detect_file_type(path)
        try:
            if ftype == "csv":
                df = pd.read_csv(path, **read_kwargs)
            elif ftype == "excel":
                df = pd.read_excel(path, **read_kwargs)
            else:  # json
                df = pd.read_json(path, **read_kwargs)
        except Exception as e:
            err_msg = f"[load_data] Lỗi đọc file {path}: {e}"
            self._log(err_msg)
            raise IOError(f"Không thể đọc file {path}: {e}") from e

        self.df_raw_ = df
        self._log(f"[load_data] Đã đọc {path} với shape={df.shape}")
        return df

    # ----------------- xử lý X, y -----------------

    def _drop_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.id_cols:
            return df
        cols_to_drop = [c for c in self.id_cols if c in df.columns]
        return df.drop(columns=cols_to_drop, errors="ignore")

    def split_features_target(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Tách X, y từ DataFrame.

        Nếu target_col = None -> chỉ trả về X, y = None (dùng cho test set).
        """
        if df is None:
            if self.df_raw_ is None:
                raise RuntimeError("Chưa gọi load_data hoặc chưa truyền df.")
            df = self.df_raw_

        df = df.copy()
        df = self._drop_id_columns(df)

        if self.target_col is None:
            return df, None

        if self.target_col not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột target '{self.target_col}' trong DataFrame."
            )

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        if self.target_col is None:
            self._log(f"[split_features_target] Không có target_col, chỉ trả về X với shape={df.shape}")
            return df, None
        
        self._log(
            f"[split_features_target] X shape={X.shape}, y len={len(y)}, target={self.target_col}"
        )
        return X, y

    # ----------------- pipeline -----------------

    def build_feature_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """
        Xây dựng pipeline dựa trên X_train, dùng các tham số của chính Preprocessor.
        """
        self.feature_pipe_ = build_feature_pipeline(
            df_train=X_train,
            ordinal_mapping=self.ordinal_mapping or ORDINAL_MAP_CANONICAL,
            use_domain_features=self.use_domain_features,
            use_target_encoding=self.use_target_encoding,
            target_enc_cols=self.target_enc_cols,
            enable_variance_selector=self.enable_variance_selector,
            variance_threshold=self.variance_threshold,
            enable_kbest_mi=self.enable_kbest_mi,
            k_best_features=self.k_best_features,
            mi_random_state=self.mi_random_state,
        )

        self._log(
            "[build_feature_pipeline] Pipeline được tạo với "
            f"use_domain_features={self.use_domain_features}, "
            f"use_target_encoding={self.use_target_encoding}, "
            f"enable_variance_selector={self.enable_variance_selector}, "
            f"variance_threshold={self.variance_threshold}, "
            f"enable_kbest_mi={self.enable_kbest_mi}, "
            f"k_best_features={self.k_best_features}, "
            f"mi_random_state={self.mi_random_state}"
        )
        return self.feature_pipe_

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Fit pipeline trên tập train và trả về X_train_processed.
        """
        if self.feature_pipe_ is None:
            self.build_feature_pipeline(X_train)
        X_proc = self.feature_pipe_.fit_transform(X_train, y_train)
        self._log(
            f"[fit_transform] Đã fit pipeline trên X_train shape={X_train.shape}, "
            f"y_train len={len(y_train) if y_train is not None else 'None'}, "
            f"kết quả shape={X_proc.shape}"
        )
        return X_proc

    def transform_new(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Biến đổi tập dữ liệu mới (val/test) bằng pipeline đã fit.
        """
        if self.feature_pipe_ is None:
            msg = "Pipeline chưa được fit. Hãy gọi fit_transform trước khi transform_new."
            self._log(f"[transform_new] ERROR: {msg}")
            raise RuntimeError(msg)
        X_proc = self.feature_pipe_.transform(X_new)
        self._log(
            f"[transform_new] Biến đổi X_new shape={X_new.shape} -> shape={X_proc.shape}"
        )
        return X_proc
    
    def save_log(self, path: str = "preprocessing.log") -> None:
        """Ghi toàn bộ log ra file text."""
        with open(path, "w", encoding="utf-8") as f:
            for line in self.logs:
                f.write(line + "\n")

    def get_log_df(self) -> pd.DataFrame:
        """Trả về log ở dạng DataFrame để dễ xem / lưu CSV."""
        return pd.DataFrame({"step": range(len(self.logs)), "message": self.logs})
    
    def get_feature_names_out(self) -> np.ndarray:
        """
        Trả về tên các feature sau toàn bộ pipeline.
        Yêu cầu self.feature_pipe_ đã được fit.
        """
        if self.feature_pipe_ is None:
            raise RuntimeError(
                "feature_pipe_ chưa được tạo. Hãy gọi build_feature_pipeline / fit_transform trước."
            )
        try:
            return self.feature_pipe_.get_feature_names_out()
        except AttributeError as e:
            raise NotFittedError(
                "Pipeline chưa hỗ trợ get_feature_names_out đầy đủ "
                "(có thể một số step chưa cài đặt)."
            ) from e
  