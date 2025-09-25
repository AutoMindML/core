import warnings
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas._typing import ArrayLike
from scipy import stats
from scipy.linalg import eigvals
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

from automind.data_utils.shared import method_registry, register_method
from automind.models.measure import InformationTheoretic, Simple, Statistical

warnings.filterwarnings("ignore")


# -------------------- SIMPLE MEASURES --------------------


@register_method(Simple.ATTR_TO_INST)
def attr_to_inst(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Ratio of number of attributes to number of instances (d/n)"""
    n_instances = len(df)
    n_attributes = len(df.columns) - (
        1 if column else 0
    )  # Exclude target if specified
    ratio = n_attributes / n_instances if n_instances > 0 else 0
    return df, ratio


@register_method(Simple.INST_TO_ATTR)
def inst_to_attr(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Ratio of number of instances to number of attributes (n/d)"""
    n_instances = len(df)
    n_attributes = len(df.columns) - (1 if column else 0)
    ratio = n_instances / n_attributes if n_attributes > 0 else 0
    return df, ratio


@register_method(Simple.CAT_TO_NUM)
def cat_to_num(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Ratio of categorical to numeric attributes"""
    feature_df = df.drop(columns=[column]) if column else df
    n_categorical = len(
        feature_df.select_dtypes(include=["object", "category"]).columns
    )
    n_numeric = len(feature_df.select_dtypes(include=[np.number]).columns)
    ratio = (
        n_categorical / n_numeric
        if n_numeric > 0
        else np.inf
        if n_categorical > 0
        else 0
    )
    return df, ratio


@register_method(Simple.NUM_TO_CAT)
def num_to_cat(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Ratio of numeric to categorical attributes"""
    feature_df = df.drop(columns=[column]) if column else df
    n_categorical = len(
        feature_df.select_dtypes(include=["object", "category"]).columns
    )
    n_numeric = len(feature_df.select_dtypes(include=[np.number]).columns)
    ratio = (
        n_numeric / n_categorical
        if n_categorical > 0
        else np.inf
        if n_numeric > 0
        else 0
    )
    return df, ratio


@register_method(Simple.CLASS_TO_ATTR)
def class_to_attr(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Ratio of number of classes to number of attributes (q/d)"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )
    n_classes = df[column].nunique()
    n_attributes = len(df.columns) - 1
    ratio = n_classes / n_attributes if n_attributes > 0 else 0
    return df, ratio


@register_method(Simple.INST_TO_CLASS)
def inst_to_class(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Ratio of number of instances to number of classes (n/q)"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )
    n_instances = len(df)
    n_classes = df[column].nunique()
    ratio = n_instances / n_classes if n_classes > 0 else 0
    return df, ratio


@register_method(Simple.FREQ_CLASS)
def freq_class(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any, Any]:
    """Frequencies of class values"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    if df[column].dtype not in ["object", "category"]:
        return df, [], []

    frequencies = df[column].value_counts(normalize=True).sort_index()
    return df, frequencies.values, frequencies.index.tolist()


@register_method(Simple.NR_ATTR)
def nr_attr(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of attributes"""
    n_attributes = len(df.columns) - (1 if column else 0)
    return df, n_attributes


@register_method(Simple.NR_ATTR_MISSING)
def nr_attr_missing(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of attributes with missing values"""
    feature_df = df.drop(columns=[column]) if column else df
    n_missing_attrs = pd.Series(feature_df.isnull().any()).sum()
    return df, n_missing_attrs


@register_method(Simple.NR_BIN)
def nr_bin(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of binary attributes"""
    feature_df = df.drop(columns=[column]) if column else df
    n_binary = 0
    for col in feature_df.columns:
        if feature_df[col].nunique() == 2:
            n_binary += 1
    return df, n_binary


@register_method(Simple.NR_CAT)
def nr_cat(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of categorical attributes"""
    feature_df = df.drop(columns=[column]) if column else df
    n_categorical = len(
        feature_df.select_dtypes(include=["object", "category"]).columns
    )
    return df, n_categorical


@register_method(Simple.NR_CLASS)
def nr_class(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of classes"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )
    n_classes = df[column].nunique()
    return df, n_classes


@register_method(Simple.NR_INST)
def nr_inst(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of instances"""
    n_instances = len(df)
    return df, n_instances


@register_method(Simple.NR_INST_MISSING)
def nr_inst_missing(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of instances with missing values"""
    n_missing_inst = pd.Series(df.isnull().any(axis=1)).sum()
    return df, n_missing_inst


@register_method(Simple.NR_MISSING)
def nr_missing(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Total number of missing values"""
    n_missing = df.isnull().sum().sum()
    return df, n_missing


@register_method(Simple.NR_NUM)
def nr_num(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of numeric attributes"""
    feature_df = df.drop(columns=[column]) if column else df
    n_numeric = len(feature_df.select_dtypes(include=[np.number]).columns)
    return df, n_numeric


# -------------------- STATISTICAL MEASURES --------------------


@register_method(Statistical.CAN_COR)
def can_cor(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray[Any], Optional[ArrayLike]]:
    """Canonical correlations between predictive attributes and class"""

    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    try:
        feature_df = df.select_dtypes(include=[np.number]).drop(
            columns=[column], errors="ignore"
        )
        if len(feature_df.columns) == 0:
            return df, np.array([0]), None

        # Encode target if categorical
        le = LabelEncoder()
        y_encoded = le.fit_transform(df[column])

        # Compute canonical correlations using LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(feature_df.fillna(0), y_encoded)

        # Canonical correlations are related to eigenvalues
        eigenvals = np.array(eigvals(lda.covariance_))
        can_corrs = np.sqrt(eigenvals / (eigenvals + 1))
        return df, np.real(can_corrs), feature_df.columns.values
    except Exception:
        return df, np.array([0]), None


@register_method(Statistical.COR)
def cor(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Absolute attribute correlations"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) < 2:
        return df, np.array([0]), None

    corr_matrix = feature_df.corr().abs()
    # Get upper triangle excluding diagonal
    upper_triangle = np.triu(corr_matrix.values, k=1)
    correlations = upper_triangle[upper_triangle != 0]

    return (
        df,
        correlations,
        feature_df.columns.values if len(correlations) > 0 else np.array([0]),
    )


@register_method(Statistical.COV)
def cov(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Covariances"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) < 2:
        return df, np.array([0]), None

    cov_matrix = feature_df.cov()
    upper_triangle = np.triu(cov_matrix.values, k=1)
    covariances = upper_triangle[upper_triangle != 0]
    return (
        df,
        np.abs(covariances),
        feature_df.columns.values if len(covariances) > 0 else np.array([0]),
    )


@register_method(Statistical.NR_DISC)
def nr_disc(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, int]:
    """Number of discriminant functions"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    try:
        feature_df = df.select_dtypes(include=[np.number]).drop(
            columns=[column], errors="ignore"
        )
        n_classes = df[column].nunique()
        n_features = len(feature_df.columns)
        n_discriminants = min(n_classes - 1, n_features)
        return df, max(0, n_discriminants)
    except Exception:
        return df, 0


@register_method(Statistical.EIGHENVALUES)
def eigenvalues(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Eigenvalues of covariance matrix"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    try:
        cov_matrix = feature_df.cov()
        eigenvals = np.array(eigvals(cov_matrix.values))
        return df, np.real(eigenvals[eigenvals >= 0]), feature_df.columns.values
    except Exception:
        return df, np.array([0]), None


@register_method(Statistical.G_MEAN)
def g_mean(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Geometric mean"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    # Only compute for positive values
    result = []
    for col in feature_df.columns:
        positive_vals = feature_df[col].dropna()
        positive_vals = positive_vals[positive_vals > 0]
        if len(positive_vals) > 0:
            result.append(stats.gmean(positive_vals))
        else:
            result.append(0)
    return df, np.array(result), feature_df.columns.values


@register_method(Statistical.H_MEAN)
def h_mean(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Harmonic mean"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    result = []
    for col in feature_df.columns:
        positive_vals = feature_df[col].dropna()
        positive_vals = positive_vals[positive_vals > 0]
        if len(positive_vals) > 0:
            result.append(stats.hmean(positive_vals))
        else:
            result.append(0)
    return df, np.array(result), feature_df.columns.values


@register_method(Statistical.IQ_Range)
def iq_range(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Interquartile range"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    iqr_values = []
    for col in feature_df.columns:
        q75, q25 = np.percentile(feature_df[col].dropna(), [75, 25])
        iqr_values.append(q75 - q25)
    return df, np.array(iqr_values), feature_df.columns.values


@register_method(Statistical.KURTOSIS)
def kurtosis(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Kurtosis"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    kurt_values = []
    for col in feature_df.columns:
        vals = feature_df[col].dropna()
        if len(vals) > 3:
            kurt_values.append(stats.kurtosis(vals))
        else:
            kurt_values.append(0)
    return df, np.array(kurt_values), feature_df.columns.values


@register_method(Statistical.MAD)
def mad(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Median absolute deviation"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    mad_values = []
    for col in feature_df.columns:
        vals = feature_df[col].dropna()
        if len(vals) > 0:
            median = np.median(vals)
            mad_val = np.median(np.abs(vals - median))
            mad_values.append(mad_val)
        else:
            mad_values.append(0)
    return df, np.array(mad_values), feature_df.columns.values


@register_method(Statistical.MAX)
def max_val(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Maximum values"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    return df, np.array(feature_df.max().values), feature_df.columns.values


@register_method(Statistical.MEAN)
def mean_val(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Mean values"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    return (
        df,
        np.array(pd.Series(feature_df.mean()).values),
        feature_df.columns.values,
    )


@register_method(Statistical.MEDIAN)
def median_val(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Median values"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    return (
        df,
        np.array(pd.Series(feature_df.median()).values),
        feature_df.columns.values,
    )


@register_method(Statistical.MIN)
def min_val(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Minimum values"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    return df, np.array(feature_df.min().values), feature_df.columns.values


@register_method(Statistical.NR_COR_ATTR)
def nr_cor_attr(
    df: pd.DataFrame,
    column: Optional[str] = None,
    threshold: float = 0.8,
) -> Tuple[pd.DataFrame, np.float64]:
    """Number of attribute pairs with high correlation"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) < 2:
        return df, np.float64(0)

    corr_matrix = feature_df.corr().abs()
    upper_triangle = np.triu(corr_matrix.values, k=1)
    high_corr_count = np.sum(upper_triangle > threshold)
    total_pairs = (len(feature_df.columns) * (len(feature_df.columns) - 1)) // 2
    ratio = high_corr_count / total_pairs if total_pairs > 0 else 0
    return df, np.float64(ratio)


@register_method(Statistical.NR_NORM)
def nr_norm(
    df: pd.DataFrame,
    column: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, int]:
    """Number of attributes with normal distribution"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, 0

    normal_count = 0
    for col in feature_df.columns:
        vals = feature_df[col].dropna()
        if len(vals) > 8:  # Minimum sample size for normality test
            _, p_value = stats.normaltest(vals)
            if p_value > alpha:  # Fail to reject null hypothesis of normality
                normal_count += 1
    return df, normal_count


@register_method(Statistical.NR_OUTLIERS)
def nr_outliers(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Number of attributes with outlier values"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, 0

    outlier_count = 0
    for col in feature_df.columns:
        vals = feature_df[col].dropna()
        if len(vals) > 0:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            if np.any((vals < lower_bound) | (vals > upper_bound)):
                outlier_count += 1

    return df, outlier_count


@register_method(Statistical.RANGE)
def range_val(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Range (max - min)"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    return (
        df,
        (feature_df.max() - feature_df.min()).values,
        feature_df.columns.values,
    )


@register_method(Statistical.SD)
def sd(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Standard deviation"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    return (
        df,
        np.array(pd.Series(feature_df.std()).values),
        feature_df.columns.values,
    )


@register_method(Statistical.SD_RATIO)
def sd_ratio(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """Statistic test for homogeneity of covariances"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    try:
        feature_df = df.select_dtypes(include=[np.number]).drop(
            columns=[column], errors="ignore"
        )
        if len(feature_df.columns) == 0:
            return df, 1.0

        # Group by class and compute covariance matrices
        classes = df[column].unique()
        if len(classes) < 2:
            return df, 1.0

        cov_matrices = []
        for cls in classes:
            class_data = feature_df[df[column] == cls]
            if len(class_data) > 1:
                cov_matrices.append(pd.DataFrame(class_data).cov())

        if len(cov_matrices) < 2:
            return df, 1.0

        # Compute ratio of largest to smallest eigenvalue
        all_eigenvals = []
        for cov_mat in cov_matrices:
            eigenvals = np.array(eigvals(cov_mat.values))
            all_eigenvals.extend(np.real(eigenvals[eigenvals > 0]))

        if len(all_eigenvals) > 0:
            ratio = max(all_eigenvals) / min(all_eigenvals)
            return df, ratio
        else:
            return df, 1.0
    except Exception:
        return df, 1.0


@register_method(Statistical.SKEWNESS)
def skewness(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Skewness"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    skew_values = []
    for col in feature_df.columns:
        vals = feature_df[col].dropna()
        if len(vals) > 2:
            skew_values.append(stats.skew(vals))
        else:
            skew_values.append(0)
    return df, np.array(skew_values), feature_df.columns.values


@register_method(Statistical.T_MEAN)
def t_mean(
    df: pd.DataFrame,
    column: Optional[str] = None,
    proportiontocut: float = 0.1,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Trimmed mean"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    trimmed_means = []
    for col in feature_df.columns:
        vals = feature_df[col].dropna()
        if len(vals) > 0:
            trimmed_means.append(stats.trim_mean(vals, proportiontocut))
        else:
            trimmed_means.append(0)
    return df, np.array(trimmed_means), feature_df.columns.values


@register_method(Statistical.VAR)
def var(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Variance"""
    feature_df = df.select_dtypes(include=[np.number])
    if column and column in feature_df.columns:
        feature_df = feature_df.drop(columns=[column])

    if len(feature_df.columns) == 0:
        return df, np.array([0]), None

    return (
        df,
        np.array(pd.Series(feature_df.var()).values),
        feature_df.columns.values,
    )


@register_method(Statistical.W_LAMBDA)
def w_lambda(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """Wilks lambda"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    try:
        feature_df = df.select_dtypes(include=[np.number]).drop(
            columns=[column], errors="ignore"
        )
        if len(feature_df.columns) == 0:
            return df, 1.0

        # Use LDA to compute Wilks lambda
        le = LabelEncoder()
        y_encoded = le.fit_transform(df[column])

        lda = LinearDiscriminantAnalysis()
        lda.fit(feature_df.fillna(0), y_encoded)

        # Approximate Wilks lambda from eigenvalues
        eigenvals = eigvals(lda.covariance_)
        wilks_lambda = 1.0 / (1.0 + np.sum(np.real(eigenvals)))
        return df, min(1.0, max(0.0, wilks_lambda))
    except Exception:
        return df, 1.0


# -------------------- INFORMATION THEORETIC MEASURES --------------------


def entropy(x):
    """Calculate entropy of a variable"""
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


@register_method(InformationTheoretic.ATTR_ENT)
def attr_ent(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Attributes entropy"""
    feature_df = df.drop(columns=[column]) if column else df
    categorical_df = feature_df.select_dtypes(include=["object", "category"])

    if len(categorical_df.columns) == 0:
        return df, np.array([0]), None

    entropies = []
    for col in categorical_df.columns:
        vals = categorical_df[col].dropna()
        if len(vals) > 0:
            entropies.append(entropy(vals))
        else:
            entropies.append(0)
    return df, np.array(entropies), categorical_df.columns.values


@register_method(InformationTheoretic.CLASS_ENT)
def class_ent(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Class entropy"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    class_entropy = entropy(df[column].dropna())
    return df, class_entropy


@register_method(InformationTheoretic.EQ_NUM_ATTR)
def eq_num_attr(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """Equivalent number of attributes"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    feature_df = df.drop(columns=[column])
    categorical_df = feature_df.select_dtypes(include=["object", "category"])

    if len(categorical_df.columns) == 0:
        return df, 0

    class_entropy = entropy(df[column].dropna())
    if class_entropy == 0:
        return df, 0

    total_mutual_info = 0
    for col in categorical_df.columns:
        joint_vals = df[[col, column]].dropna()
        if len(joint_vals) > 0:
            # Mutual information
            attr_entropy = entropy(joint_vals[col])
            joint_entropy = entropy(
                joint_vals.apply(
                    lambda x: str(x[col]) + "_" + str(x[column]), axis=1
                )
            )
            mutual_info = attr_entropy + class_entropy - joint_entropy
            total_mutual_info += mutual_info

    equivalent_attrs = (
        total_mutual_info / class_entropy if class_entropy > 0 else 0
    )
    return df, equivalent_attrs


@register_method(InformationTheoretic.JOINT_ENT)
def joint_ent(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Joint entropy of attributes and classes"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    feature_df = df.drop(columns=[column])
    categorical_df = feature_df.select_dtypes(include=["object", "category"])

    if len(categorical_df.columns) == 0:
        return df, np.array([0]), None

    joint_entropies = []
    for col in categorical_df.columns:
        joint_vals = df[[col, column]].dropna()
        if len(joint_vals) > 0:
            # Create joint variable by combining attribute and class values
            joint_var = joint_vals.apply(
                lambda x: str(x[col]) + "_" + str(x[column]), axis=1
            )
            joint_entropies.append(entropy(joint_var))
        else:
            joint_entropies.append(0)
    return df, np.array(joint_entropies), categorical_df.columns.values


@register_method(InformationTheoretic.MUT_INF)
def mut_inf(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, NDArray, Optional[ArrayLike]]:
    """Mutual information of attributes and classes"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    feature_df = df.drop(columns=[column])
    categorical_df = feature_df.select_dtypes(include=["object", "category"])

    if len(categorical_df.columns) == 0:
        return df, np.array([0]), None

    class_entropy = entropy(df[column].dropna())
    mutual_infos = []

    for col in categorical_df.columns:
        joint_vals = df[[col, column]].dropna()
        if len(joint_vals) > 0:
            attr_entropy = entropy(joint_vals[col])
            joint_var = joint_vals.apply(
                lambda x: str(x[col]) + "_" + str(x[column]), axis=1
            )
            joint_entropy = entropy(joint_var)
            mutual_info = attr_entropy + class_entropy - joint_entropy
            mutual_infos.append(max(0, mutual_info))
        else:
            mutual_infos.append(0)

    return df, np.array(mutual_infos), categorical_df.columns.values


@register_method(InformationTheoretic.NS_RATIO)
def ns_ratio(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """Noisiness of attributes"""
    if not column:
        raise ValueError(
            "Target column must be specified for classification measures"
        )

    feature_df = df.drop(columns=[column])
    categorical_df = feature_df.select_dtypes(include=["object", "category"])

    if len(categorical_df.columns) == 0:
        return df, 0

    class_entropy = entropy(df[column].dropna())
    if class_entropy == 0:
        return df, 0

    # Calculate average conditional entropy H(Y|X)
    total_conditional_entropy = 0
    valid_attrs = 0

    for col in categorical_df.columns:
        joint_vals = df[[col, column]].dropna()
        if len(joint_vals) > 0:
            # Calculate H(Y|X) = H(X,Y) - H(X)
            attr_entropy = entropy(joint_vals[col])
            joint_var = joint_vals.apply(
                lambda x: str(x[col]) + "_" + str(x[column]), axis=1
            )
            joint_entropy = entropy(joint_var)
            conditional_entropy = joint_entropy - attr_entropy
            total_conditional_entropy += conditional_entropy
            valid_attrs += 1

    if valid_attrs == 0:
        return df, 0

    avg_conditional_entropy = total_conditional_entropy / valid_attrs
    noise_ratio = (
        avg_conditional_entropy / class_entropy if class_entropy > 0 else 0
    )
    return df, max(0, noise_ratio)


# -------------------- UTILITY FUNCTIONS --------------------


def get_numeric_features(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> pd.DataFrame:
    """Get numeric features from DataFrame, excluding target if specified"""
    numeric_df = df.select_dtypes(include=[np.number])
    if target_column and target_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_column])
    return numeric_df


def get_categorical_features(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> pd.DataFrame:
    """Get categorical features from DataFrame, excluding target if specified"""
    categorical_df = df.select_dtypes(include=["object", "category"])
    if target_column and target_column in categorical_df.columns:
        categorical_df = categorical_df.drop(columns=[target_column])
    return categorical_df


def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if denominator is 0"""
    return numerator / denominator if denominator != 0 else default


def robust_entropy(values, base=2):
    """Calculate entropy with handling for edge cases"""
    if len(values) == 0:
        return 0

    unique_vals, counts = np.unique(values, return_counts=True)
    if len(unique_vals) == 1:
        return 0

    probabilities = counts / len(values)
    entropy_val = -np.sum(probabilities * np.log(probabilities) / np.log(base))
    return entropy_val


def detect_outliers_iqr(data, multiplier=1.5):
    """Detect outliers using IQR method"""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (data < lower_bound) | (data > upper_bound)


def normalize_features(df: pd.DataFrame):
    """Normalize numeric features to [0, 1] range"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    normalized_df = df.copy()

    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0

    return normalized_df


# -------------------- BATCH COMPUTATION HELPERS --------------------


def compute_all_simple_measures(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> dict:
    """Compute all simple meta-features at once"""
    results = {}

    for measure in Simple:
        try:
            _, *rest = apply_method(measure, df, target_column)

            # indicate that columns are returned
            if len(rest) > 1:
                results[measure.name] = {
                    "classes": rest[1],
                    "values": rest[0],
                }
            else:
                results[measure.name] = rest[0] if len(rest) > 0 else None
        except Exception as e:
            print(f"Error computing {measure.name}: {e}")
            results[measure.name] = None

    return results


def compute_all_statistical_measures(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> dict:
    """Compute all statistical meta-features at once"""
    results = {}

    for measure in Statistical:
        try:
            _, *rest = apply_method(measure, df, target_column)

            # indicate that columns are returned
            if len(rest) > 1:
                results[measure.name] = {
                    "columns": rest[1],
                    "values": rest[0],
                }
            else:
                results[measure.name] = rest[0] if len(rest) > 0 else None
        except Exception as e:
            print(f"Error computing {measure.name}: {e}")
            results[measure.name] = None

    return results


def compute_all_information_theoretic_measures(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> dict:
    """Compute all information theoretic meta-features at once"""
    results = {}

    for measure in InformationTheoretic:
        try:
            _, *rest = apply_method(measure, df, target_column)

            if len(rest) > 1:
                results[measure.name] = {
                    "columns": rest[1],
                    "values": rest[0],
                }
            else:
                results[measure.name] = rest[0] if len(rest) > 0 else None
        except Exception as e:
            print(f"Error computing {measure.name}: {e}")
            results[measure.name] = None

    return results


def compute_all_measures(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> dict:
    """Compute all meta-features at once"""
    all_results = {}
    all_results.update(compute_all_simple_measures(df, target_column))
    all_results.update(compute_all_statistical_measures(df, target_column))
    all_results.update(
        compute_all_information_theoretic_measures(df, target_column)
    )

    return all_results


def apply_method(
    method: Enum,
    df: pd.DataFrame,
    column: Optional[str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Any]:
    """Apply a registered processing method to a DataFrame."""
    func = method_registry.get(method.name)

    if not func:
        raise NotImplementedError(f"Method not implemented: {method}")

    return func(df=df, column=column, **kwargs)
