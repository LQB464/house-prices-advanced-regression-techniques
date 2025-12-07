# preprocessing/columns.py
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_feature_lists(
    df: pd.DataFrame,
    ordinal_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Tách danh sách cột thành:
      - ord_cols: cột ordinal
      - cat_cols: cột object/category còn lại
      - num_cols: cột numeric
    """
    df = df.copy()
    ordinal_cols = ordinal_cols or []

    ord_cols = [c for c in ordinal_cols if c in df.columns]
    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in ord_cols
    ]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return ord_cols, cat_cols, num_cols


def make_column_transformer(
    df: pd.DataFrame,
    ordinal_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Tạo ColumnTransformer cho numeric, categorical, ordinal.
    """
    ord_cols, cat_cols, num_cols = build_feature_lists(df, ordinal_cols)
    num_cols = [c for c in num_cols if c not in ord_cols]

    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ord_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("cats", cat_pipeline, cat_cols),
            ("ords", ord_pipeline, ord_cols),
            ("nums", num_pipeline, num_cols),
        ],
        remainder="drop",
    )
    return preproc


__all__ = ["build_feature_lists", "make_column_transformer"]
