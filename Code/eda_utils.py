"""
eda_utils.py

Các hàm và lớp phục vụ cho Phần 3 - Trực quan hóa và phân tích:
- EDA dữ liệu đầu vào
- Trực quan kết quả mô hình
- Phân tích đặc trưng quan trọng (feature importance, permutation importance, SHAP, PDP)
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import shap


# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------
def _ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_final_estimator(model):
    """Lấy ra estimator cuối cùng từ một Pipeline nếu có."""
    if isinstance(model, Pipeline):
        # ưu tiên key "model" vì project hiện tại đang dùng tên này
        if "model" in model.named_steps:
            return model.named_steps["model"]
        # ngược lại lấy bước cuối
        return model.steps[-1][1]
    return model


# -------------------------------------------------------------------
# Class EDAVisualizer cho dữ liệu đầu vào
# -------------------------------------------------------------------
class EDAVisualizer:
    """
    Thực hiện EDA cho DataFrame đầu vào.

    Parameters
    ----------
    df : DataFrame ban đầu
    target_col : tên cột target (vd: "SalePrice")
    output_dir : thư mục lưu các file hình
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None,
                 output_dir: Union[str, Path] = "eda_plots"):
        self.df = df
        self.target_col = target_col
        self.output_dir = _ensure_dir(output_dir)

    # 1) Phân bố target
    def plot_target_distribution(self, log: bool = True, bins: int = 40):
        if self.target_col is None:
            raise ValueError("target_col chưa được thiết lập cho EDAVisualizer.")

        y = self.df[self.target_col].dropna()
        plt.figure(figsize=(8, 5))
        if log:
            sns.histplot(np.log1p(y), bins=bins, kde=True)
            plt.xlabel(f"log1p({self.target_col})")
            plt.title(f"Phan bo log1p({self.target_col})")
            fname = self.output_dir / f"target_{self.target_col}_log_hist.png"
        else:
            sns.histplot(y, bins=bins, kde=True)
            plt.xlabel(self.target_col)
            plt.title(f"Phan bo {self.target_col}")
            fname = self.output_dir / f"target_{self.target_col}_hist.png"

        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    # 2) Thiếu dữ liệu theo cột
    def plot_missing_values(self, min_fraction: float = 0.0):
        miss_frac = self.df.isna().mean().sort_values(ascending=False)
        miss_frac = miss_frac[miss_frac >= min_fraction]
        if miss_frac.empty:
            return

        plt.figure(figsize=(9, 5))
        sns.barplot(x=miss_frac.index, y=miss_frac.values)
        plt.xticks(rotation=90)
        plt.ylabel("Ti le NaN")
        plt.title("Ti le gia tri thieu theo cot")
        plt.tight_layout()
        fname = self.output_dir / "missing_values_fraction.png"
        plt.savefig(fname, dpi=150)
        plt.close()

    # 3) Histogram cho các biến numeric
    def plot_numeric_histograms(self, max_cols: int = 16, bins: int = 30):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in num_cols:
            num_cols.remove(self.target_col)

        num_cols = num_cols[:max_cols]
        if not num_cols:
            return

        n = len(num_cols)
        ncols = 4
        nrows = int(np.ceil(n / ncols))

        plt.figure(figsize=(4 * ncols, 3 * nrows))
        for i, col in enumerate(num_cols, 1):
            plt.subplot(nrows, ncols, i)
            sns.histplot(self.df[col].dropna(), bins=bins, kde=False)
            plt.title(col)
        plt.tight_layout()
        fname = self.output_dir / "numeric_histograms.png"
        plt.savefig(fname, dpi=150)
        plt.close()

    # 4) Boxplot target theo categorical
    def plot_boxplots_for_top_categories(
        self,
        cat_col: str,
        top_k: int = 10,
    ):
        """
        Vẽ boxplot target theo top-k categories của cat_col.
        """
        if self.target_col is None:
            raise ValueError("Cần có target_col để vẽ boxplot theo category.")

        if cat_col not in self.df.columns:
            raise ValueError(f"{cat_col} khong ton tai trong df.")

        vc = self.df[cat_col].value_counts().head(top_k).index
        data = self.df[self.df[cat_col].isin(vc)]

        plt.figure(figsize=(10, 5))
        sns.boxplot(x=cat_col, y=self.target_col, data=data)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{self.target_col} theo {cat_col} (top {top_k} nhom)")
        plt.tight_layout()
        fname = self.output_dir / f"boxplot_{self.target_col}_by_{cat_col}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

    # 5) Heatmap tương quan numeric
    def plot_correlation_heatmap(self, top_n: int = 20, method: str = "spearman"):
        """
        Vẽ heatmap tương quan giữa target và các numeric features
        (chọn top-n theo trị tuyệt đối).
        """
        if self.target_col is None:
            raise ValueError("Cần target_col để vẽ correlation heatmap.")

        num_df = self.df.select_dtypes(include=[np.number])
        if self.target_col not in num_df.columns:
            raise ValueError(f"Target {self.target_col} khong phai numeric trong df.")

        corr = num_df.corr(method=method)[self.target_col].dropna().sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        corr = corr.iloc[1 : top_n + 1]  # bỏ chính target

        plt.figure(figsize=(max(8, top_n * 0.4), 4))
        sns.barplot(x=corr.index, y=corr.values)
        plt.xticks(rotation=90)
        plt.ylabel(f"Corr({self.target_col}, feature) ({method})")
        plt.title(f"Tinh tuong quan voi {self.target_col} (top {top_n})")
        plt.tight_layout()
        fname = self.output_dir / "target_correlation_bar.png"
        plt.savefig(fname, dpi=150)
        plt.close()

        # thêm heatmap nhỏ nếu muốn
        sel_cols = [self.target_col] + corr.index.tolist()
        plt.figure(figsize=(8, 6))
        sns.heatmap(num_df[sel_cols].corr(method=method), annot=False, cmap="coolwarm")
        plt.title("Heatmap tuong quan (subset)")
        plt.tight_layout()
        fname = self.output_dir / "correlation_heatmap_subset.png"
        plt.savefig(fname, dpi=150)
        plt.close()


# -------------------------------------------------------------------
# Trực quan kết quả mô hình
# -------------------------------------------------------------------
def plot_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "True vs predicted",
    output_path: Optional[Union[str, Path]] = None,
):
    """Scatter plot y_true vs y_pred."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=15)
    min_y = min(np.min(y_true), np.min(y_pred))
    max_y = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_y, max_y], [min_y, max_y], "r--", linewidth=1)
    plt.xlabel("Gia tri that")
    plt.ylabel("Gia tri du doan")
    plt.title(title)
    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=150)
    plt.close()


# -------------------------------------------------------------------
# Feature importance, permutation importance, SHAP, PDP
# -------------------------------------------------------------------
def plot_feature_importance_from_model(
    model,
    feature_names: Sequence[str],
    top_n: int = 20,
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Tính và vẽ feature importance theo 2 cách:
    - Nếu model có thuộc tính feature_importances_: dùng trực tiếp
    - Nếu là model tuyến tính có coef_: dùng |coef_|
    """
    est = _get_final_estimator(model)

    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        raise ValueError(
            "Model khong co feature_importances_ hoac coef_. "
            "Hay dung permutation importance."
        )

    feature_names = list(feature_names)
    if len(feature_names) != len(importances):
        raise ValueError("Do dai feature_names khong khop voi importances.")

    order = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    sns.barplot(
        x=importances[order][::-1],
        y=[feature_names[i] for i in order][::-1],
        orient="h",
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature importance (model based)")
    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_permutation_importance(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 20,
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Vẽ permutation importance cho bất kì model nào.
    Dùng được cả với Pipeline.
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    importances = result.importances_mean
    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    sns.barplot(
        x=importances[order][::-1],
        y=[feature_names[i] for i in order][::-1],
        orient="h",
    )
    plt.xlabel("Permutation importance (mean decrease in score)")
    plt.ylabel("Feature")
    plt.title("Permutation importance")
    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_shap_summary(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    output_dir: Union[str, Path] = "eda_plots",
    max_display: int = 20,
):
    """
    Vẽ SHAP summary plot (scatter + bar).
    Hỗ trợ tốt nhất cho tree models (RandomForest, XGBoost, LightGBM, CatBoost).
    """

    est = _get_final_estimator(model)
    output_dir = _ensure_dir(output_dir)

    # Nếu X là DataFrame thì dùng luôn, nếu không thì tạo DataFrame giả
    if isinstance(X, pd.DataFrame):
        X_shap = X.copy()
    else:
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_shap = pd.DataFrame(X, columns=feature_names)

    # Auto chọn explainer phù hợp
    explainer = shap.Explainer(est, X_shap)
    shap_values = explainer(X_shap)

    # Summary dot plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_dot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary bar plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_partial_dependence_for_features(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    features: Sequence[Union[int, str]],
    output_dir: Union[str, Path] = "eda_plots",
):
    """
    Vẽ Partial Dependence Plot cho một vài features.

    Parameters
    ----------
    model : model hoặc Pipeline đã fit
    X     : DataFrame hoặc ndarray đầu vào của model
    features : danh sách index hoặc tên cột (nếu X là DataFrame)
    """
    output_dir = _ensure_dir(output_dir)

    if isinstance(X, pd.DataFrame):
        feat_repr = features
    else:
        # Nếu X là ndarray thì coi features là index
        feat_repr = features

    fig = plt.figure(figsize=(4 * len(features), 3))
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=feat_repr,
        ax=fig.gca(),
    )
    plt.tight_layout()
    plt.savefig(output_dir / "partial_dependence.png", dpi=150)
    plt.close()
