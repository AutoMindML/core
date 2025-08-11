# based on: https://www.sciencedirect.com/science/article/pii/S0950705121011631?ref=pdf_download&fr=RR-2&rr=966a9acdbdc48415
# Rivolli, A., Garcia, L. P., Soares, C., Vanschoren, J., & de Carvalho, A. C. (2022).
# Meta-features for meta-learning. Knowledge-Based Systems, 240, 108101 in section 4.1, page 4.


from enum import Enum, auto

# symbol annotations:
# use <C-v> in neovim when type unicode symbol
# ∞ = u221e

# Annotations:
# d_bar: mean of number of attributes (multi-dataset or multi-measure results)
# n_bar: mean of number of instances (multi-dataset or multi-measure results)
# q: number of classes


class Simple(Enum):
    """
    For numeric or categorical or both attributes measures
    """

    ATTR_TO_INST = auto()

    """
    Name:           Ratio of the number of attributes per the number of instances (d/n) 
    Des:            Known as dimensionality
    Domain:         Both
    Range:          [0, d_bar]
    Cardinality:    1
    """

    INST_TO_ATTR = auto()

    """
    Name:           Ratio of the number of instances per the number of attributes (n/d)
    Des:            Sparsity of data, a potential indicator for over-fitting when value is too small
    Domain:         Both
    Range:          [0, n_bar]
    Cardinality:    1
    """

    CAT_TO_NUM = auto()

    """
    Name:           Ratio of the number of categorical attributes per numeric attributes         
    Domain:         Both
    Range:          [0, d_bar]
    Cardinality:    1
    # Exception:    True
    """

    NUM_TO_CAT = auto()

    """
    Name:           Ratio of the number of numeric attributes per the number of categorical attributes
    Domain:         Both
    Range:          [0, d_bar]
    Cardinality:    1
    # Exception:    True
    """

    CLASS_TO_ATTR = auto()

    """
    Name:           Ratio of the number of classes per the number of attributes (q/d)
    Des:            Measure properties of the target attribute distribution, such as class imbalance.
    Domain:         Both
    Range:          [0, q]
    Cardinality:    1
    # Task:         Classification
    """

    INST_TO_CLASS = auto()

    """
    Name:           Ratio of the number of instances per the number of classes (n/q)
    Des:            Measure properties of the target attribute distribution, such as class imbalance.
    Domain:         Both
    Range:          [1, n_bar]
    Cardinality:    1
    """

    FREQ_CLASS = auto()

    """
    Name:           Frequencies of the class values
    # Domain:       Categorical
    Range:          [0, 1]
    Cardinality:    q
    # Task:         Classification
    """

    NR_ATTR = auto()

    """
    Name:           Number of attributes
    Domain:         Both
    Range:          [1, +∞]
    Cardinality:    1
    """

    NR_ATTR_MISSING = auto()

    """
    Name:           Number of attributes with missing values
    Domain:         Both
    Range:          [0, d]
    Cardinality:    1
    """

    NR_BIN = auto()

    """
    Name:           Number of binary attributes
    Domain:         Both
    Range:          [0, d]
    Cardinality:    1
    """

    NR_CAT = auto()

    """
    Name:           Number of categorical attributes
    Domain:         Both
    Range:          [0, d]
    Cardinality:    1
    """

    NR_CLASS = auto()

    """
    Name:           Number of classes
    # Domain:       Categorical
    Range:          [2, n_bar]
    Cardinality:    1
    # Task:         Classification
    """

    NR_INST = auto()

    """
    Name:           Number of instances
    Domain:         Both
    Range:          [q, +∞]
    Cardinality:    1
    """

    NR_INST_MISSING = auto()

    """
    Name:           Number of instances with missing values
    Domain:         Both
    Range:          [0, n]
    Cardinality:    1
    """

    NR_MISSING = auto()

    """
    Name:           Number of missing values
    Domain:         Both
    Range:          [0, dn_bar]
    Cardinality:    1
    """

    NR_NUM = auto()

    """
    Name:           Number of numeric attributes
    Domain:         Both
    Range:          [0, d]
    Cardinality:    1
    """


class Statistical(Enum):
    """
    For numeric attributes measures
    """

    CAN_COR = auto()

    """
    Name:           Canonical correlations (典型相關性) between the predictive attributes and the class
    Range:          [0, 1]
    Cardinality:    d_bar
    # Task:         Classification
    """

    COR = auto()

    """
    Name:           Absolute attributes correlations
    Range:          [0, 1]
    Cardinality:    d_square_bar
    # Exception:    True
    """

    COV = auto()

    """
    Name:           Covariances (共變異數，協方差)
    Range:          [0, ∞]
    Cardinality:    d_square_bar
    """

    NR_DISC = auto()

    """
    Name:           Number of discriminant (判別式) functions 
    Range:          [0, d]
    Cardinality:    1
    # Task:         Classification
    """

    EIGHENVALUES = auto()

    """
    Name:           Eighenvalues (特徵值) of the covariance matrix
    Range:          [0, ∞]
    Cardinality:    d_bar
    """

    G_MEAN = auto()

    """
    Name:           Geometric mean (幾何平均數)
    Range:          [0, ∞]
    Cardinality:    d
    # Exception:    True
    """

    H_MEAN = auto()

    """
    Name:           Harmonic mean (調和平均數)
    Range:          inherited
    Cardinality:    d
    """

    IQ_Range = auto()

    """
    Name:           Interquartile range (四分位距)
    Range:          [0, ∞]
    Cardinality:    d
    """

    KURTOSIS = auto()

    """
    Name:           Kurtosis (峰度)
    Range:          [-3, ∞]
    Cardinality:    d
    # Exception:    True
    """

    MAD = auto()

    """
    Name:           Median absolute deviation (中位數絕對偏差)
    Range:          [0, ∞]
    Cardinality:    d
    """

    MAX = auto()

    """
    Name:           Maximum
    Range:          inherited
    Cardinality:    d
    """

    MEAN = auto()

    """
    Name:           Mean
    Range:          inherited
    Cardinality:    d
    """

    MEDIAN = auto()

    """
    Name:           Median
    Range:          inherited
    Cardinality:    d
    """

    MIN = auto()

    """
    Name:           Minimum
    Range:          inherited
    Cardinality:    d
    """

    NR_COR_ATTR = auto()

    """
    Name:           Number of attributes pairs with high correlation
    Range:          [0, 1]
    Cardinality:    1
    # Exception:    True
    """

    NR_NORM = auto()

    """
    Name:           Number of attributes with normal distribution
    Range:          [0, d]
    Cardinality:    1
    """

    NR_OUTLIERS = auto()

    """
    Name:           Number of attributes with outliers values
    Range:          [0, d]
    Cardinality:    1
    """

    RANGE = auto()

    """
    Name:           Range
    Range:          [0, ∞]
    Cardinality:    d
    """

    SD = auto()

    """
    Name:           Standard deviation (標準差)
    Range:          [0, ∞]
    Cardinality:    d
    """

    SD_RATIO = auto()

    """
    Name:           Statistic test for homogeneity of covariances (協方性同質性)
    Range:          [1, ∞]
    Cardinality:    1
    # Task:         Classification
    # Exception:    True
    """

    SKEWNESS = auto()

    """
    Name:           Skewness (偏度)
    Range:          [-∞, ∞]
    Cardinality:    d
    # Exception:    True
    """

    T_MEAN = auto()

    """
    Name:           Trimmed mean (修剪均值)
    Range:          inherited
    Cardinality:    d
    """

    VAR = auto()

    """
    Name:           Attributes variance (變異數)
    Range:          [0, ∞]
    Cardinality:    d
    """

    W_LAMBDA = auto()

    """
    Name:           Wilks lambda (Wilks 統計量)
    Range:          [0, 1]
    Cardinality:    1
    # Task:         Classification
    """


class InformationTheoretic(Enum):
    """
    For categorical attributes measures
    """

    ATTR_ENT = auto()

    """
    Name:           Attributes entropy
    Range:          [0, log2(n)]
    Cardinality:    d
    """

    CLASS_ENT = auto()

    """
    Name:           Class entropy
    Range:          [0, log2(q)]
    Cardinality:    1
    # Task:         Classification
    """

    EQ_NUM_ATTR = auto()

    """
    Name:           Equivalent number of attributes
    Range:          [0, ∞]
    Cardinality:    1
    # Task:         Classification
    """

    JOINT_ENT = auto()

    """
    Name:           Joint Entropy of attributes and classes
    Range:          [0, log2(n)]
    Cardinality:    d
    # Task:         Classification
    """

    MUT_INF = auto()

    """
    Name:           Mutual information of attributes and classes
    Range:          [0, log2(n)]
    Cardinality:    d
    # Task:         Classification
    """

    NS_RATIO = auto()

    """
    Name:           Noisiness (噪聲) of attributes
    Range:          [0, ∞]
    Cardinality:    1
    # Task:         Classification
    """
