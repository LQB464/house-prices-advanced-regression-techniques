# preprocessing/pipeline.py
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import ORDINAL_MAP_CANONICAL
from .domain import DomainFeatureAdder, add_domain_features
from .columns import make_column_transformer
from .transformers import (
    OrdinalMapper,
    MissingnessIndicator,
    RareCategoryGrouper,
    OutlierClipper,
    FiniteCleaner,
    DropAllNaNColumns,
    TargetEncoderTransformer,
    VarianceFeatureSelector,
    KBestMutualInfoSelector,
)


def build_feature_pipeline(
    df_train: pd.DataFrame,
    ordinal_mapping: Optional[Mapping[str, Any]] = None,
    use_domain_features: bool = True,
    use_target_encoding: bool = False,
    target_enc_cols: Optional[List[str]] = None,
    enable_variance_selector: bool = True,
    variance_threshold: float = 0.0,
    enable_kbest_mi: bool = False,
    k_best_features: int = 150,
    mi_random_state: int = 0,
) -> Pipeline:
    """
    Tạo sklearn Pipeline tiền xử lý đầy đủ.
    Logic giống hệt build_feature_pipeline gốc, chỉ tách sang file.
    """

    ordinal_mapping = ordinal_mapping or {}
    ordinal_cols = list(ordinal_mapping.keys()) if ordinal_mapping else []

    df_for_cols = add_domain_features(df_train) if use_domain_features else df_train
    preproc_cols = make_column_transformer(df_for_cols, ordinal_cols=ordinal_cols)

    steps: List[Tuple[str, Any]] = []

    if use_domain_features:
        steps.append(("domain", DomainFeatureAdder()))

    if ordinal_mapping:
        steps.append(("ordinal_map", OrdinalMapper(mapping=ordinal_mapping)))

    steps.append(("missing_indicator", MissingnessIndicator()))
    steps.append(("rare_grouper", RareCategoryGrouper(min_freq=20)))

    if use_target_encoding:
        if target_enc_cols is None:
            target_enc_cols = [
                "Neighborhood",
                "MSZoning",
                "Exterior1st",
                "Exterior2nd",
                "SaleType",
                "SaleCondition",
            ]
        steps.append(
            ("target_encoder", TargetEncoderTransformer(cols=target_enc_cols))
        )

    steps.extend(
        [
            ("outlier_clip", OutlierClipper(factor=1.5)),
            ("finite_clean", FiniteCleaner()),
            ("col_transform", preproc_cols),
            ("drop_all_nan", DropAllNaNColumns()),
        ]
    )

    if enable_variance_selector:
        steps.append(
            ("var_selector", VarianceFeatureSelector(threshold=variance_threshold))
        )

    if enable_kbest_mi and k_best_features is not None:
        steps.append(
            (
                "mi_selector",
                KBestMutualInfoSelector(
                    k=int(k_best_features),
                    random_state=int(mi_random_state),
                ),
            )
        )

    pipe = Pipeline(steps=steps)
    return pipe


__all__ = ["build_feature_pipeline"]
