"""
modeling/preprocessing_mixin.py

Mixin providing feature preprocessing pipeline construction with full
integration into the Preprocessor component and TrainerConfig workflow.

Extended Description
--------------------
This module defines the `PreprocessingMixin`, a lightweight helper that builds
the sklearn-based feature pipeline used across the training process. It wraps
the `build_feature_pipeline` function from the preprocessing package and applies
standard preprocessing configurations, including:
- canonical ordinal mappings
- domain feature engineering
- variance filtering
- mutual information K-best selection
- controlled target encoding options

The mixin expects to operate inside a TrainerConfig-based class that provides:
`self.dp`, `self.logger`, `self.X_train_`, `self.X_test_`, and `self.feature_pipe_`.

Main Components
---------------
- PreprocessingMixin:
  - build_preprocessing: assemble the full feature pipeline using training data

Usage Example
-------------
>>> class Trainer(TrainerConfig, PreprocessingMixin):
...     pass
>>> trainer = Trainer()
>>> trainer.split_data()
>>> pipe = trainer.build_preprocessing()

Notes
-----
This mixin centralizes preprocessing logic, ensuring consistent feature
construction across all models trained in the project.
"""

from preprocessing import build_feature_pipeline, ORDINAL_MAP_CANONICAL
from sklearn.pipeline import Pipeline


class PreprocessingMixin:
    """
    Preprocessing helper mixin that constructs the feature pipeline.

    Extended Description
    --------------------
    The `PreprocessingMixin` builds the unified sklearn feature pipeline used
    during training. It delegates feature engineering and transformation logic
    to the external `build_feature_pipeline` function while standardizing:
    - canonical ordinal mappings
    - domain-engineered features
    - variance threshold filtering
    - mutual-information-based K-best selection

    It relies on TrainerConfig to provide:
    `self.dp`, `self.logger`, `self.X_train_`, `self.X_test_`, and `self.feature_pipe_`.

    Parameters
    ----------
    None
        The mixin uses configuration and state injected by TrainerConfig.

    Attributes
    ----------
    feature_pipe_ : sklearn.Pipeline or None
        The assembled preprocessing pipeline produced by `build_preprocessing`.
    X_train_, X_test_ : DataFrame or None
        Training and test feature matrices used to infer preprocessing logic.

    Examples
    --------
    >>> trainer.split_data()
    >>> pipe = trainer.build_preprocessing()
    >>> isinstance(pipe, sklearn.pipeline.Pipeline)
    True
    """

    def build_preprocessing(self) -> Pipeline:
        if self.X_train_ is None or self.X_test_ is None:
            raise RuntimeError("Call split_data before build_preprocessing.")

        # Gọi đúng hàm build_feature_pipeline như trong trainer.py cũ
        self.feature_pipe_ = build_feature_pipeline(
            df_train=self.X_train_,
            ordinal_mapping=ORDINAL_MAP_CANONICAL,
            use_domain_features=True,
            use_target_encoding=False,
            enable_variance_selector=True,
            variance_threshold=0.0,
            enable_kbest_mi=True,
            k_best_features=200,
            mi_random_state=0,
        )

        self.logger.info("Feature preprocessing pipeline built successfully.")
        return self.feature_pipe_