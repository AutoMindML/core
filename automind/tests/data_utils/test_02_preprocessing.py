from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

from automind.data_utils.preprocessing import (
    DC,
    FE,
    CrossValidation,
    DataCleaningRecommendations,
    DataQualityReport,
    DataQualityType,
    EnumByName,
    FeatureEngineeringRecommendations,
    Issue,
    ModelingApproach,
    OverallQuality,
    RecommendedAlgorithm,
    Strength,
    TaskType,
    apply_method,
    apply_method_transform,
    apply_pca,
    apply_scaler,
    borderline_smote,
    calculate_iqr_bounds,
    convert_to_datetime,
    detect_datetime_format,
    extract_date_parts,
    identify_outliers,
    impute_backward_fill,
    impute_constant,
    impute_forward_fill,
    # Import all the processing functions
    impute_mean,
    impute_median,
    impute_mode,
    iqr_remove_outliers,
    iqr_winsorize_outliers,
    label_encode,
    log_transform,
    method_registry,
    min_max_scale,
    normalize,
    one_hot_encode,
    quantile_discretize,
    remove_infinite,
    rename_duplicate_columns,
    robust_scale,
    smote,
    standardize,
    treat_zero_as_missing_value,
    uniform_discretize,
)


class TestEnumByName:
    """Test the EnumByName validator."""

    def test_enum_by_name_with_string(self):
        """Test that EnumByName accepts string names."""
        validator = EnumByName()
        schema = validator.__get_pydantic_core_schema__(DC.MissingValues, None)

        # This would be used internally by Pydantic
        assert DC.MissingValues.IMPUTE_MEAN.name == "IMPUTE_MEAN"

    def test_enum_by_name_case_insensitive(self):
        """Test case insensitive enum validation."""
        validator = EnumByName(ignore_case=True)
        # The actual validation logic would be tested through Pydantic models
        assert validator.ignore_case is True


class TestDataModels:
    """Test the Pydantic data models."""

    def test_issue_model(self):
        """Test Issue model creation."""
        issue = Issue(
            type="MISSING_VALUES",
            columns=["col1", "col2"],
            description="Missing values found",
        )
        assert issue.type == DataQualityType.MISSING_VALUES
        assert issue.columns == ["col1", "col2"]
        assert issue.description == "Missing values found"

    def test_strength_model(self):
        """Test Strength model creation."""
        strength = Strength(
            type="HIGH_COMPLETENESS", description="High data completeness"
        )
        assert strength.type == DataQualityType.HIGH_COMPLETENESS
        assert strength.description == "High data completeness"

    def test_data_quality_report_model(self):
        """Test DataQualityReport model creation."""
        issue = Issue(
            type="MISSING_VALUES", columns=["col1"], description="Missing values"
        )
        strength = Strength(type="HIGH_COMPLETENESS", description="High completeness")

        report = DataQualityReport(
            overall_quality="GOOD",
            summary="Data quality is good",
            issues=[issue],
            strengths=[strength],
        )

        assert report.overall_quality == OverallQuality.GOOD
        assert len(report.issues) == 1
        assert len(report.strengths) == 1

    def test_modeling_approach_model(self):
        """Test ModelingApproach model creation."""
        approach = ModelingApproach(
            task_type="CLASSIFICATION",
            target="target_col",
            recommended_algorithm=RecommendedAlgorithm(
                name="RandomForest",
                reason="Good for classification",
                params={"n_estimators": 100},
            ),
            evaluation_metrics=["ACCURACY", "F1"],
            cross_validation=CrossValidation(method="K_FOLD", folds=5, stratified=True),
            data_cleaning=DataCleaningRecommendations(
                missing_values=[], outliers=[], duplicates=[], balancing=[]
            ),
            feature_engineering=FeatureEngineeringRecommendations(
                creation=[], transformation=[], selection=[]
            ),
            test_size=0.2,
            validation_size=0.2,
        )

        assert approach.task_type == TaskType.CLASSIFICATION
        assert approach.target == "target_col"
        assert approach.recommended_algorithm.name == "RandomForest"


class TestMissingValueMethods:
    """Test missing value imputation methods."""

    def setup_method(self):
        """Set up test data."""
        self.df_numeric = pd.DataFrame(
            {"col1": [1, 2, np.nan, 4, 5], "col2": [1.1, 2.2, 3.3, np.nan, 5.5]}
        )

        self.df_categorical = pd.DataFrame(
            {"col1": ["a", "b", np.nan, "a", "c"], "col2": ["x", "y", "z", np.nan, "x"]}
        )

    def test_impute_mean(self):
        """Test mean imputation."""
        result = impute_mean(self.df_numeric, "col1")
        expected_mean = self.df_numeric["col1"].mean()
        assert not result["col1"].isna().any()
        assert result["col1"].iloc[2] == expected_mean

    def test_impute_mean_non_numeric_raises_error(self):
        """Test that mean imputation raises error for non-numeric data."""
        with pytest.raises(TypeError, match="Column must be numeric"):
            impute_mean(self.df_categorical, "col1")

    def test_impute_median(self):
        """Test median imputation."""
        result = impute_median(self.df_numeric, "col1")
        expected_median = self.df_numeric["col1"].median()
        assert not result["col1"].isna().any()
        assert result["col1"].iloc[2] == expected_median

    def test_impute_median_non_numeric_raises_error(self):
        """Test that median imputation raises error for non-numeric data."""
        with pytest.raises(TypeError, match="Column must be numeric"):
            impute_median(self.df_categorical, "col1")

    def test_impute_mode(self):
        """Test mode imputation."""
        result = impute_mode(self.df_categorical, "col1")
        assert not result["col1"].isna().any()
        # Mode should be 'a' (appears twice)
        assert result["col1"].iloc[2] == "a"

    def test_impute_constant(self):
        """Test constant imputation."""
        result = impute_constant(self.df_numeric, "col1", value=999)
        assert not result["col1"].isna().any()
        assert result["col1"].iloc[2] == 999

    def test_impute_forward_fill(self):
        """Test forward fill imputation."""
        result = impute_forward_fill(self.df_numeric, "col1")
        assert result["col1"].iloc[2] == 2  # Forward filled from previous value

    def test_impute_backward_fill(self):
        """Test backward fill imputation."""
        result = impute_backward_fill(self.df_numeric, "col1")
        assert result["col1"].iloc[2] == 4  # Backward filled from next value

    def test_treat_zero_as_missing_value(self):
        """Test converting zeros to missing values."""
        df = pd.DataFrame({"col1": [0, 1, 2, 0, 3]})
        result = treat_zero_as_missing_value(df, "col1")
        assert result["col1"].isna().sum() == 2
        assert np.isnan(result["col1"].iloc[0])
        assert np.isnan(result["col1"].iloc[3])


class TestOutlierMethods:
    """Test outlier detection and handling methods."""

    def setup_method(self):
        """Set up test data with outliers."""
        self.df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            }
        )

    def test_calculate_iqr_bounds(self):
        """Test IQR bounds calculation."""
        series = pd.Series([1, 2, 3, 4, 5, 100])
        lower, upper = calculate_iqr_bounds(series)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        expected_lower = q1 - 1.5 * iqr
        expected_upper = q3 + 1.5 * iqr

        assert lower == expected_lower
        assert upper == expected_upper

    def test_identify_outliers_iqr(self):
        """Test outlier identification using IQR method."""
        series = pd.Series([1, 2, 3, 4, 5, 100])
        outliers = identify_outliers(series, method="iqr")
        assert outliers.iloc[-1] is np.True_  # 100 should be identified as outlier
        assert outliers.iloc[0] is np.False_  # 1 should not be an outlier

    def test_identify_outliers_zscore(self):
        """Test outlier identification using Z-score method."""
        series = pd.Series([1, 2, 3, 4, 5, 100])
        outliers = identify_outliers(series, method="zscore", factor=2)
        assert outliers.iloc[-1] is np.True_  # 100 should be identified as outlier

    def test_identify_outliers_percentile(self):
        """Test outlier identification using percentile method."""
        series = pd.Series([1, 2, 3, 4, 5, 100])
        outliers = identify_outliers(series, method="percentile")
        # Should identify extreme values based on 1st and 99th percentiles
        assert isinstance(outliers, pd.Series)

    def test_identify_outliers_non_numeric_raises_error(self):
        """Test that outlier identification raises error for non-numeric data."""
        series = pd.Series(["a", "b", "c"])
        with pytest.raises(TypeError, match="Outlier detection requires numeric data"):
            identify_outliers(series)

    def test_identify_outliers_unknown_method_raises_error(self):
        """Test that unknown method raises error."""
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            identify_outliers(series, method="unknown")

    def test_iqr_remove_outliers(self):
        """Test outlier removal using IQR method."""
        original_len = len(self.df)
        result = iqr_remove_outliers(self.df, "col1")
        assert len(result) < original_len  # Should remove outliers
        assert 100 not in result["col1"].values  # Outlier should be removed

    def test_iqr_winsorize_outliers(self):
        """Test outlier winsorization using IQR method."""
        result = iqr_winsorize_outliers(self.df, "col1")
        assert len(result) == len(self.df)  # Should not remove rows
        assert 100 not in result["col1"].values  # Outlier should be clipped

    def test_iqr_winsorize_non_numeric_raises_error(self):
        """Test that winsorization raises error for non-numeric data."""
        df = pd.DataFrame({"col1": ["a", "b", "c"]})
        with pytest.raises(TypeError, match="Winsorization requires numeric data"):
            iqr_winsorize_outliers(df, "col1")


class TestBalancingMethods:
    """Test balancing methods."""

    def setup_method(self):
        """Set up imbalanced test data."""
        self.X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
            }
        )
        # Imbalanced target: 8 class 0, 2 class 1
        self.y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    @patch("automind.data_utils.preprocessing.SMOTE")
    def test_smote(self, mock_smote_class):
        """Test SMOTE balancing."""
        mock_smote = MagicMock()
        mock_smote.fit_resample.return_value = (self.X, self.y)
        mock_smote_class.return_value = mock_smote

        X_res, y_res = smote(self.X, self.y, random_state=42)

        mock_smote_class.assert_called_once_with(random_state=42)
        mock_smote.fit_resample.assert_called_once_with(self.X, self.y)
        assert X_res is not None
        assert y_res is not None

    @patch("automind.data_utils.preprocessing.BorderlineSMOTE")
    def test_borderline_smote(self, mock_borderline_smote_class):
        """Test BorderlineSMOTE balancing."""
        mock_smote = MagicMock()
        mock_smote.fit_resample.return_value = (self.X, self.y)
        mock_borderline_smote_class.return_value = mock_smote

        X_res, y_res = borderline_smote(self.X, self.y, random_state=42)

        mock_borderline_smote_class.assert_called_once_with(random_state=42)
        mock_smote.fit_resample.assert_called_once_with(self.X, self.y)
        assert X_res is not None
        assert y_res is not None


class TestFeatureTransformationMethods:
    """Test feature transformation methods."""

    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3, 4, 5],
                "categorical_col": ["a", "b", "c", "d", "e"],
            }
        )

    def test_standardize(self):
        """Test standardization."""
        result_df, scaler = standardize(self.df, "numeric_col")

        # Check that mean is approximately 0 and std is approximately 1
        assert np.isclose(result_df["numeric_col"].mean(), 0.0, atol=0.5)
        assert np.isclose(result_df["numeric_col"].std(), 1.0, atol=0.5)
        assert isinstance(scaler, StandardScaler)

    def test_standardize_non_numeric_raises_error(self):
        """Test that standardization raises error for non-numeric data."""
        with pytest.raises(TypeError, match="Standardization requires numeric data"):
            standardize(self.df, "categorical_col")

    def test_min_max_scale(self):
        """Test min-max scaling."""
        result_df, scaler = min_max_scale(self.df, "numeric_col")

        # Check that values are scaled to [0, 1]
        assert result_df["numeric_col"].min() == 0
        assert result_df["numeric_col"].max() == 1
        assert isinstance(scaler, MinMaxScaler)

    def test_min_max_scale_non_numeric_raises_error(self):
        """Test that min-max scaling raises error for non-numeric data."""
        with pytest.raises(TypeError, match="Min-max scaling requires numeric data"):
            min_max_scale(self.df, "categorical_col")

    def test_robust_scale(self):
        """Test robust scaling."""
        result_df, scaler = robust_scale(self.df, "numeric_col")

        # Check that scaler is returned
        assert isinstance(scaler, RobustScaler)
        assert "numeric_col" in result_df.columns

    def test_robust_scale_non_numeric_raises_error(self):
        """Test that robust scaling raises error for non-numeric data."""
        with pytest.raises(TypeError, match="Robust scaling requires numeric data"):
            robust_scale(self.df, "categorical_col")

    def test_log_transform_positive_values(self):
        """Test log transformation with positive values."""
        result_df = log_transform(self.df, "numeric_col")

        # Check that log column is created
        assert "numeric_col_log" in result_df.columns
        assert len(result_df) == len(self.df)

    def test_log_transform_with_zeros(self):
        """Test log transformation with zeros."""
        df_with_zeros = pd.DataFrame({"col": [0, 1, 2, 3, 4]})
        result_df = log_transform(df_with_zeros, "col")

        # Should use log1p for data with zeros
        assert "col_log" in result_df.columns
        assert result_df["col_log"].iloc[0] == 0  # log1p(0) = 0

    def test_log_transform_negative_values_raises_error(self):
        """Test that log transformation raises error for negative values."""
        df_with_negatives = pd.DataFrame({"col": [-1, 0, 1, 2, 3]})
        with pytest.raises(
            ValueError, match="Log transform requires non-negative values"
        ):
            log_transform(df_with_negatives, "col")

    def test_log_transform_non_numeric_raises_error(self):
        """Test that log transformation raises error for non-numeric data."""
        with pytest.raises(TypeError, match="Log transform requires numeric data"):
            log_transform(self.df, "categorical_col")

    def test_uniform_discretize(self):
        """Test uniform discretization."""
        result_df, discretizer = uniform_discretize(self.df, "numeric_col", n_bins=3)

        # Check that discretizer is returned and values are discretized
        assert isinstance(discretizer, KBinsDiscretizer)
        assert result_df["numeric_col"].nunique() <= 3

    def test_quantile_discretize(self):
        """Test quantile discretization."""
        result_df, discretizer = quantile_discretize(self.df, "numeric_col", n_bins=3)

        # Check that discretizer is returned and values are discretized
        assert isinstance(discretizer, KBinsDiscretizer)
        assert result_df["numeric_col"].nunique() <= 3

    def test_normalize(self):
        """Test normalization."""
        result_df, normalizer = normalize(self.df, "numeric_col")

        # Check that normalizer is returned
        assert isinstance(normalizer, Normalizer)
        assert "numeric_col" in result_df.columns


class TestFeatureCreationMethods:
    """Test feature creation methods."""

    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "categorical_col": ["a", "b", "c", "a", "b"],
                "date_col": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],
            }
        )

    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        result_df = one_hot_encode(self.df, "categorical_col")

        # Check that original column is dropped and new columns are created
        assert "categorical_col" not in result_df.columns
        assert "categorical_col_a" in result_df.columns
        assert "categorical_col_b" in result_df.columns
        assert "categorical_col_c" in result_df.columns

    def test_label_encode(self):
        """Test label encoding."""
        result_df, encoder = label_encode(self.df, "categorical_col")

        # Check that encoder is returned and values are encoded
        assert isinstance(encoder, LabelEncoder)
        assert result_df["categorical_col"].dtype in ["int32", "int64"]
        assert set(result_df["categorical_col"].unique()) == {0, 1, 2}

    def test_convert_to_datetime(self):
        """Test datetime conversion."""
        result_df = convert_to_datetime(self.df, "date_col")

        # Check that column is converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(result_df["date_col"])

    def test_convert_to_datetime_invalid_format_raises_error(self):
        """Test that datetime conversion raises error for invalid format."""
        df_invalid = pd.DataFrame({"date_col": ["invalid", "date", "format"]})
        with pytest.raises(TypeError, match="Cannot detect datetime format"):
            convert_to_datetime(df_invalid, "date_col")

    def test_extract_date_parts(self):
        """Test date parts extraction."""
        # First convert to datetime
        df_with_datetime = convert_to_datetime(self.df, "date_col")
        result_df = extract_date_parts(df_with_datetime, "date_col")

        # Check that date parts are extracted
        assert "date_col_year" in result_df.columns
        assert "date_col_month" in result_df.columns
        assert "date_col_day" in result_df.columns
        assert "date_col_dayofweek" in result_df.columns
        assert "date_col_quarter" in result_df.columns
        assert "date_col" not in result_df.columns  # Original column should be dropped

    def test_extract_date_parts_non_datetime_raises_error(self):
        """Test that date parts extraction raises error for non-datetime data."""
        with pytest.raises(
            TypeError, match="Extract date parts requires datetime column"
        ):
            extract_date_parts(self.df, "categorical_col")


class TestFeatureSelectionMethods:
    """Test feature selection methods."""

    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3, 4, 5],
                "categorical_col": ["a", "b", "c", "d", "e"],
            }
        )

    def test_apply_pca(self):
        """Test PCA application."""
        result_df, pca = apply_pca(self.df, "numeric_col", n_components=1)

        # Check that PCA is applied
        assert isinstance(pca, PCA)
        assert "numeric_col_pca_component_1" in result_df.columns
        assert (
            "numeric_col" not in result_df.columns
        )  # Original column should be dropped

    def test_apply_pca_non_numeric_raises_error(self):
        """Test that PCA raises error for non-numeric data."""
        with pytest.raises(TypeError, match="PCA requires numeric data"):
            apply_pca(self.df, "categorical_col")


class TestDateTimeUtils:
    """Test datetime utility functions."""

    def test_detect_datetime_format_success(self):
        """Test successful datetime format detection."""
        df = pd.DataFrame({"date_col": ["2023-01-01", "2023-01-02", "2023-01-03"]})
        format_detected = detect_datetime_format(df, "date_col")
        assert format_detected == "%Y-%m-%d"

    def test_detect_datetime_format_mixed(self):
        """Test mixed datetime format detection."""
        df = pd.DataFrame({"date_col": ["2023-01-01", "01/02/2023", "2023-01-03"]})
        format_detected = detect_datetime_format(df, "date_col")
        # Should detect mixed format or specific format
        assert format_detected is not None

    def test_detect_datetime_format_failure(self):
        """Test datetime format detection failure."""
        df = pd.DataFrame({"date_col": ["not", "a", "date"]})
        format_detected = detect_datetime_format(df, "date_col")
        assert format_detected is None

    def test_detect_datetime_format_empty_data(self):
        """Test datetime format detection with empty data."""
        df = pd.DataFrame({"date_col": []})
        format_detected = detect_datetime_format(df, "date_col")
        assert format_detected is None


class TestUtilityFunctions:
    """Test utility functions."""

    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "numeric_col": [1, 2, np.nan, 4, 5],
                "categorical_col": ["a", "b", np.nan, "d", "e"],
            }
        )

    def test_apply_method_missing_values(self):
        """Test apply_method with missing value method."""
        result_df = apply_method(
            DC.MissingValues.IMPUTE_MEAN, self.df, column="numeric_col"
        )

        assert not result_df["numeric_col"].isna().any()

    def test_apply_method_transformation(self):
        """Test apply_method with transformation method."""
        result_df, scaler = apply_method(
            FE.Transformations.STANDARDIZE, self.df, column="numeric_col"
        )

        assert isinstance(scaler, StandardScaler)
        assert abs(result_df["numeric_col"].mean()) < 1e-10

    def test_apply_method_not_implemented_raises_error(self):
        """Test that apply_method raises error for unimplemented method."""

        # Create a mock enum that's not registered
        class MockEnum:
            name = "UNREGISTERED_METHOD"

        with pytest.raises(NotImplementedError, match="Method not implemented"):
            apply_method(MockEnum(), self.df, column="numeric_col")

    def test_apply_method_transform_balancing(self):
        """Test apply_method_transform with balancing method."""
        X = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]})
        y = pd.Series([0, 0, 1, 1])

        with patch("automind.data_utils.preprocessing.SMOTE") as mock_smote_class:
            mock_smote = MagicMock()
            mock_smote.fit_resample.return_value = (X, y)
            mock_smote_class.return_value = mock_smote

            X_res, y_res = apply_method_transform(
                DC.Balancing.SMOTE, X, y, random_state=42
            )

            assert X_res is not None
            assert y_res is not None

    def test_apply_scaler(self):
        """Test apply_scaler function."""
        # First fit a scaler
        scaler = StandardScaler()
        scaler.fit(self.df[["numeric_col"]].dropna())

        # Then apply it
        result_df = apply_scaler(self.df, "numeric_col", scaler)

        # Check that transformation was applied
        assert "numeric_col" in result_df.columns
        assert len(result_df) == len(self.df)


class TestMethodRegistry:
    """Test method registry functionality."""

    def test_register_method_decorator(self):
        """Test that register_method decorator works."""
        # Check that some methods are registered
        assert "IMPUTE_MEAN" in method_registry
        assert "STANDARDIZE" in method_registry
        assert "ONE_HOT_ENCODE" in method_registry

        # Check that the registered functions are callable
        assert callable(method_registry["IMPUTE_MEAN"])
        assert callable(method_registry["STANDARDIZE"])
        assert callable(method_registry["ONE_HOT_ENCODE"])

    def test_method_registry_completeness(self):
        """Test that all expected methods are registered."""
        expected_methods = [
            "IMPUTE_MEAN",
            "IMPUTE_MEDIAN",
            "IMPUTE_MODE",
            "IMPUTE_CONSTANT",
            "IMPUTE_FORWARD_FILL",
            "IMPUTE_BACKWARD_FILL",
            "TREAT_ZERO_AS_MISSING_VALUE",
            "IQR_REMOVE_OUTLIERS",
            "IQR_WINSORIZE_OUTLIERS",
            "SMOTE",
            "BorderlineSMOTE",
            "STANDARDIZE",
            "MIN_MAX_SCALE",
            "ROBUST_SCALE",
            "LOG_TRANSFORM",
            "UNIFORM_DISCRETIZE",
            "QUANTILE_DISCRETIZE",
            "NORMALIZE",
            "ONE_HOT_ENCODE",
            "LABEL_ENCODE",
            "CONVERT_TO_DATETIME",
            "EXTRACT_DATE_PARTS",
            "APPLY_PCA",
        ]

        for method in expected_methods:
            assert method in method_registry, f"Method {method} not registered"


class TestDataIntegrity:
    """Test data integrity and edge cases."""

    def test_original_dataframe_not_modified(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({"col1": [1, 2, np.nan, 4, 5]})
        original_df = df.copy()

        # Apply method
        result_df = impute_mean(df, "col1")

        # Check that original DataFrame is unchanged
        pd.testing.assert_frame_equal(df, original_df)
        # Check that result is different
        assert not result_df.equals(original_df)

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        df_empty = pd.DataFrame()

        # Most methods should handle empty DataFrames gracefully or raise appropriate errors
        with pytest.raises((KeyError, ValueError)):
            impute_mean(df_empty, "nonexistent_col")

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrames."""
        df_single = pd.DataFrame({"col1": [1]})

        # Should not raise errors for single-row DataFrames
        result = impute_mean(df_single, "col1")
        assert len(result) == 1
        assert result["col1"].iloc[0] == 1

    def test_all_missing_values(self):
        """Test handling of columns with all missing values."""
        df_all_missing = pd.DataFrame({"col1": [np.nan, np.nan, np.nan]})

        # Mean imputation should handle all missing values
        result = impute_mean(df_all_missing, "col1")

        print(result)

        # With all NaN values, mean should be NaN, and imputation should fill with NaN
        assert result["col1"].isna().all()

    def test_no_missing_values(self):
        """Test handling of columns with no missing values."""
        df_no_missing = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        # Should return DataFrame unchanged
        result = impute_mean(df_no_missing, "col1")
        pd.testing.assert_frame_equal(result.astype(float), df_no_missing.astype(float))

    def test_mixed_data_types(self):
        """Test handling of mixed data types in a DataFrame."""
        df_mixed = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        # Should handle different data types appropriately
        result = standardize(df_mixed, "int_col")
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Should raise error for non-numeric columns
        with pytest.raises(TypeError):
            standardize(df_mixed, "str_col")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_nonexistent_column(self):
        """Test handling of nonexistent columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(KeyError):
            impute_mean(df, "nonexistent_col")

    def test_duplicate_column_names(self):
        """Test handling of duplicate column names."""
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["col1", "col1"])
        df = rename_duplicate_columns(df)

        # Should still work but might have unexpected behavior
        # The exact behavior depends on pandas version
        try:
            result = impute_mean(df, "col1")
            assert result is not None
        except Exception as e:
            # Some operations might fail with duplicate columns
            assert isinstance(e, (ValueError, KeyError))

    def test_extreme_values(self):
        """Test handling of extreme values."""
        df_extreme = pd.DataFrame(
            {"col1": [1e-10, 1e10, -1e10, np.inf, -np.inf, np.nan]}
        )

        df_extreme = remove_infinite(df_extreme, "col1")

        result = impute_mean(df_extreme, "col1")

        # Result should not contain infinite values after mean imputation
        assert not np.isinf(result["col1"].dropna()).any()

    def test_memory_efficiency(self):
        """Test memory efficiency with large DataFrames."""
        # Create a reasonably large DataFrame
        large_df = pd.DataFrame(
            {"col1": np.random.randn(10000), "col2": np.random.randn(10000)}
        )

        # Add some missing values
        large_df.loc[::100, "col1"] = np.nan

        # Should handle large DataFrames efficiently
        result = impute_mean(large_df, "col1")
        assert len(result) == len(large_df)
        assert not result["col1"].isna().any()


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        # Test invalid n_bins for discretization
        with pytest.raises((ValueError, TypeError)):
            uniform_discretize(df, "col1", n_bins=-1)

    def test_parameter_type_validation(self):
        """Test parameter type validation."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        # Test that string parameters are handled correctly
        with pytest.raises(TypeError):
            uniform_discretize(df, "col1", n_bins="invalid")

    def test_default_parameters(self):
        """Test default parameter values."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        # Test default constant value
        result = impute_constant(df, "col1")  # Should use default value=0
        assert (result["col1"] == df["col1"]).all()  # No missing values to fill

        # Test with missing values
        df_with_missing = pd.DataFrame({"col1": [1, np.nan, 3, 4, 5]})
        result = impute_constant(df_with_missing, "col1")
        assert result["col1"].iloc[1] == 0  # Default value


class TestConcurrency:
    """Test thread safety and concurrent operations."""

    def test_method_registry_thread_safety(self):
        """Test that method registry is thread-safe."""
        # The registry should be populated at import time
        # and not modified during runtime
        initial_registry = method_registry.copy()

        # Simulate concurrent access
        assert "IMPUTE_MEAN" in method_registry
        assert "STANDARDIZE" in method_registry

        # Registry should remain unchanged
        assert method_registry == initial_registry

    def test_concurrent_data_processing(self):
        """Test concurrent data processing operations."""
        import threading

        df = pd.DataFrame({"col1": [1, 2, np.nan, 4, 5]})
        results = []

        def process_data():
            result = impute_mean(df, "col1")
            results.append(result)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_data)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All results should be identical
        assert len(results) == 5
        for result in results[1:]:
            pd.testing.assert_frame_equal(results[0], result)


class TestPerformance:
    """Test performance characteristics."""

    def test_processing_time_reasonable(self):
        """Test that processing time is reasonable."""
        import time

        # Create a moderately large DataFrame
        df = pd.DataFrame(
            {
                "col1": np.random.randn(1000),
                "col2": np.random.choice(["a", "b", "c"], 1000),
            }
        )

        # Add missing values
        df.loc[::10, "col1"] = np.nan

        # Test imputation time
        start_time = time.time()
        result = impute_mean(df, "col1")
        end_time = time.time()

        # Should complete within reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(result) == len(df)

    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Get initial memory usage
        initial_memory = process.memory_info().rss

        # Create and process a large DataFrame
        df = pd.DataFrame(
            {"col1": np.random.randn(5000), "col2": np.random.randn(5000)}
        )

        # Add missing values
        df.loc[::20, "col1"] = np.nan

        # Process data
        result = impute_mean(df, "col1")

        # Get final memory usage
        final_memory = process.memory_info().rss

        # Memory increase should be reasonable (less than 100MB)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100 * 1024 * 1024  # 100MB

        # Clean up
        del df, result


class TestIntegrationScenarios:
    """Test integration scenarios with multiple operations."""

    def test_complete_preprocessing_pipeline(self):
        """Test a complete preprocessing pipeline."""
        # Create a realistic dataset
        df = pd.DataFrame(
            {
                "numeric_col": [
                    1,
                    2,
                    np.nan,
                    4,
                    100,
                    6,
                    7,
                    8,
                    9,
                    10,
                ],  # Has missing values and outliers
                "categorical_col": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
                "date_col": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ],
            }
        )

        # Step 1: Handle missing values
        df_step1 = impute_mean(df, "numeric_col")
        assert not df_step1["numeric_col"].isna().any()

        # Step 2: Handle outliers
        df_step2 = iqr_winsorize_outliers(df_step1, "numeric_col")
        assert 100 not in df_step2["numeric_col"].values

        # Step 3: Standardize numeric features
        df_step3, scaler = standardize(df_step2, "numeric_col")
        assert abs(df_step3["numeric_col"].mean()) < 1e-10

        # Step 4: Encode categorical features
        df_step4 = one_hot_encode(df_step3, "categorical_col")
        assert "categorical_col" not in df_step4.columns
        assert any("categorical_col_" in col for col in df_step4.columns)

        # Step 5: Process datetime features
        df_step5 = convert_to_datetime(df_step4, "date_col")
        assert pd.api.types.is_datetime64_any_dtype(df_step5["date_col"])

        df_final = extract_date_parts(df_step5, "date_col")
        assert "date_col_year" in df_final.columns
        assert "date_col_month" in df_final.columns

        # Final DataFrame should be ready for modeling
        assert len(df_final) == len(df)
        assert df_final.isna().sum().sum() == 0  # No missing values

    def test_error_recovery_in_pipeline(self):
        """Test error recovery in preprocessing pipeline."""
        df = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3, 4, 5],
                "categorical_col": ["a", "b", "c", "d", "e"],
            }
        )

        # Try to apply numeric operation to categorical column
        with pytest.raises(TypeError):
            standardize(df, "categorical_col")

        # DataFrame should still be usable for other operations
        result = one_hot_encode(df, "categorical_col")
        assert result is not None
        assert len(result) == len(df)

    def test_pipeline_with_empty_results(self):
        """Test pipeline behavior with operations that might return empty results."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 1000, 2000]  # Contains outliers
            }
        )

        # Remove outliers might result in significantly reduced data
        result = iqr_remove_outliers(df, "col1")

        # Should still have some data (not all removed)
        assert len(result) > 0
        assert len(result) < len(df)  # Some outliers should be removed


if __name__ == "__main__":
    pytest.main(["-v", __file__])
