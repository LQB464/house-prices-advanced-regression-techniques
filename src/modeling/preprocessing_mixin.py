# src/modeling/preprocessing_mixin.py
from preprocessing import ORDINAL_MAP_CANONICAL


class PreprocessingMixin:
    """
    Xây dựng feature pipeline qua Preprocessor.

    Yêu cầu:
        self.dp, self.logger, self.X_train_, self.feature_pipe_
    """

    def build_preprocessing(self) -> None:
        """
        Gọi Preprocessor.build_feature_pipeline để tạo pipeline feature chung.
        """
        if self.X_train_ is None:
            raise RuntimeError("No train data. Call split_data first.")

        self.logger.info("Building feature pipeline for training data.")
        self.feature_pipe_ = self.dp.build_feature_pipeline(
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
