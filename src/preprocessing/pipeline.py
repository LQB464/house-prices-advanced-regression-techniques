"""
preprocessing/pipeline.py

Factory for constructing the full sklearn preprocessing pipeline.

Extended Description
--------------------
This module builds a complete sklearn Pipeline for tabular data using a series
of modular transformers. These include domain feature generation, ordinal encoding,
rare-category grouping, outlier clipping, column-wise transformations, finite
value cleaning, NaN-column removal, variance filtering, and mutual information
feature selection. The pipeline is highly configurable through build_feature_pipeline().

Main Components
---------------
- build_feature_pipeline: Assemble a full sklearn Pipeline with optional:
  - Domain features
  - Ordinal mapping
  - Target encoding
  - Outlier clipping
  - ColumnTransformer ( numeric/categorical/ordinal pipelines )
  - Finite cleaning + NaN column dropping
  - Variance-based selection
  - Mutual-information feature selection

Usage Example
-------------
>>> from preprocessing.pipeline import build_feature_pipeline
>>> pipe = build_feature_pipeline(df_train, ordinal_mapping=..., use_domain_features=True)
>>> X_train = pipe.fit_transform(df_train, y_train)

Notes
-----
The pipeline integrates custom transformers from transformers.py and automatically
adapts to the structure of df_train. Ordering of steps is designed to minimize
data leakage and ensure consistent column alignment.
"""

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
    k_best_features: int = 200,
    mi_random_state: int = 0,
) -> Pipeline:
    """
    Construct the complete sklearn preprocessing pipeline for tabular data.

    This function builds a sequential sklearn Pipeline consisting of:
    1. Optional domain feature generation
    2. Optional ordinal mapping
    3. Missingness indicator creation
    4. Rare-category grouping
    5. Optional target encoding
    6. Outlier clipping
    7. ColumnTransformer (numeric/ordinal/categorical)
    8. Cleaning inf values
    9. Dropping all-NaN columns
    10. Optional variance-based selection
    11. Optional MI-based K-best selection

    Parameters
    ----------
    df_train : DataFrame
        Training dataset used to infer column groupings.
    ordinal_mapping : dict, optional
        Mapping for ordinal features. Keys are column names.
    use_domain_features : bool, default True
        Whether to generate domain-engineered features.
    use_target_encoding : bool, default False
        Whether to apply target encoding to selected categorical variables.
    target_enc_cols : list of str, optional
        Columns eligible for target encoding.
    enable_variance_selector : bool, default True
        Whether to use variance thresholding.
    variance_threshold : float
        Minimum variance to retain a feature.
    enable_kbest_mi : bool, default False
        Whether to use mutual-information K-best selection.
    k_best_features : int
        Number of MI-ranked features to keep.
    mi_random_state : int
        Random state for MI scoring.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fully assembled preprocessing pipeline.

    Examples
    --------
    >>> pipe = build_feature_pipeline(df_train, use_domain_features=True)
    >>> X_train = pipe.fit_transform(df_train, y_train)
    """
    if ordinal_mapping is None:
        ordinal_mapping = {}

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
            ("col_transform", preproc_cols),
            ("finite_clean", FiniteCleaner()),
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