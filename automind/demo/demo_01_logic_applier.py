import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from automind.data_utils.logic_applier import LogicApplier


def create_sample_dataset(rows=1000):
    """Create a synthetic dataset with various data quality issues for demonstration"""
    np.random.seed(42)

    # Create base dataframe
    df = pd.DataFrame(
        {
            "age": np.random.normal(35, 12, rows),
            "income": np.random.lognormal(10, 1, rows),
            "education_years": np.random.randint(8, 22, rows),
            "credit_score": np.random.normal(700, 100, rows),
            "loan_amount": np.random.lognormal(10, 0.7, rows),
            "employment_length": np.random.randint(0, 30, rows),
            "num_dependents": np.random.poisson(1.2, rows),
            "debt_to_income": np.random.beta(2, 5, rows) * 0.5,
            "registration_date": pd.date_range(start="2020-01-01", periods=rows),
        }
    )

    # Add categorical columns
    df["gender"] = np.random.choice(["Male", "Female", "Non-binary"], rows)
    df["education_level"] = np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"], rows
    )
    df["loan_purpose"] = np.random.choice(
        ["Home", "Auto", "Personal", "Education", "Business"], rows
    )
    df["employment_status"] = np.random.choice(
        ["Employed", "Self-employed", "Unemployed", "Retired"], rows
    )
    df["marital_status"] = np.random.choice(
        ["Single", "Married", "Divorced", "Widowed"], rows
    )

    # Add a text column
    loan_descriptions = [
        "Loan for home renovation",
        "Financing my new car",
        "Need funds for my wedding",
        "Starting a small business",
        "Debt consolidation to improve finances",
        "Medical expenses coverage",
        "Higher education funding",
    ]
    df["loan_description"] = np.random.choice(loan_descriptions, rows)

    # Create the target variable (loan approval status)
    # Higher probability of approval for higher income, credit score and education
    probability = (
        (df["income"] / df["income"].max()) * 0.3
        + (df["credit_score"] / 850) * 0.4
        + (df["education_years"] / 22) * 0.2
        - (df["debt_to_income"]) * 0.1
    )
    df["loan_approved"] = np.random.binomial(1, probability)

    # Introduce missing values
    for col in ["age", "income", "credit_score", "employment_length"]:
        mask = np.random.choice([True, False], rows, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan

    # Introduce outliers
    outlier_indices = np.random.choice(rows, size=int(rows * 0.02), replace=False)
    df.loc[outlier_indices, "income"] = df["income"].max() * np.random.uniform(
        3, 5, len(outlier_indices)
    )
    df.loc[outlier_indices[: len(outlier_indices) // 2], "age"] = np.random.uniform(
        90, 120, len(outlier_indices) // 2
    )

    # Add duplicate rows
    duplicate_count = int(rows * 0.03)
    duplicate_indices = np.random.choice(rows, size=duplicate_count, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def evaluate_model(X_train, X_test, y_train, y_test):
    """Train a simple model and evaluate its performance"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Get feature importances
    feature_imp = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    return accuracy, feature_imp


def main():
    print("=== LogicApplier Demo ===")
    print(
        "This demo showcases how LogicApplier applies LLM recommendations for data preparation"
    )

    # Create synthetic dataset with issues
    print("\nCreating synthetic loan approval dataset with data quality issues...")
    df = create_sample_dataset()
    print(f"Dataset shape: {df.shape}")

    # Display dataset summary
    print("\nOriginal dataset summary:")
    print(df.info())
    print("\nMissing values by column:")
    print(df.isna().sum())

    # Visualization of original data
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df["income"].to_numpy(), kde=True)
    plt.title("Income Distribution (Original)")
    plt.xlabel("Income")

    plt.subplot(1, 2, 2)
    sns.boxplot(x="loan_approved", y="income", data=df)
    plt.title("Income by Loan Approval (Original)")
    plt.savefig("output/original_data_visualization.png")
    print("Saved visualization of original data distribution")

    # Train a model on the raw data
    print("\nTraining a baseline model on raw data (with basic preprocessing)...")

    # Basic preprocessing for baseline
    df_baseline = df.copy()
    df_baseline = df_baseline.dropna()  # Simple handling of missing values

    # Select features for modeling
    numeric_cols = list(df_baseline.select_dtypes(include=[np.number]).columns.tolist())
    numeric_cols.remove("loan_approved")  # Remove target

    categorical_cols = [
        "gender",
        "education_level",
        "loan_purpose",
        "employment_status",
        "marital_status",
    ]

    # Simple one-hot encoding for baseline
    df_baseline = pd.get_dummies(df_baseline, columns=categorical_cols, drop_first=True)

    # Baseline model features
    X = df_baseline[
        df_baseline.columns.difference(
            ["loan_approved", "loan_description", "registration_date"]
        )
    ]
    y = df_baseline["loan_approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    baseline_accuracy, baseline_importances = evaluate_model(
        X_train, X_test, y_train, y_test
    )

    print(f"\nBaseline model accuracy: {baseline_accuracy:.4f}")

    # Apply LogicApplier
    print("\nApplying LogicApplier to clean data and engineer features...")

    # We'll mock the LLM result since we can't actually run an LLM in this demo
    # In a real application, the LLM would analyze the data and provide these recommendations
    mock_llm_result = {
        "data_cleaning_recommendations": {
            "missing_values": [
                "Use KNN imputation for numeric columns",
                "Use mode imputation for categorical columns",
                "Create missing indicator flags",
            ],
            "outliers": ["Cap outliers using IQR method for numeric columns"],
            "duplicates": ["Drop duplicate rows"],
        },
        "feature_engineering": {
            "transformations": [
                "Apply log transformation to income and loan_amount",
                "Apply standardization to numeric features",
                "Apply one-hot encoding to categorical variables",
            ],
            "recommendations": [
                "Create polynomial features for key numeric columns",
                "Create ratio features between relevant numeric columns",
                "Extract datetime features from registration_date",
            ],
            "feature_selection": [
                "Remove highly correlated features with threshold 0.85",
                "Apply SelectKBest to keep top 10 features",
            ],
        },
    }

    # Initialize LogicApplier
    logic_applier = LogicApplier(df, target_column="loan_approved")

    # Instead of calling generate_and_get_llm_analysis, we'll set the mock result directly
    logic_applier.llm_result = mock_llm_result

    # Set column types that would normally be extracted by the MetaGenerator
    logic_applier.column_types = {
        "numeric": [
            "age",
            "income",
            "education_years",
            "credit_score",
            "loan_amount",
            "employment_length",
            "num_dependents",
            "debt_to_income",
        ],
        "categorical": [
            "gender",
            "education_level",
            "loan_purpose",
            "employment_status",
            "marital_status",
        ],
        "datetime": ["registration_date"],
        "text": ["loan_description"],
    }

    # Apply the recommendations
    transformed_df = logic_applier.apply_recommendations()

    # Print transformation log
    print("\nTransformations applied:")
    for i, transformation in enumerate(logic_applier.get_transformations_log(), 1):
        print(f"{i}. {transformation}")

    # Visualize transformed data
    plt.figure(figsize=(12, 5))
    # Find log-transformed income column if it exists
    income_col = "income"
    if "income_log" in transformed_df.columns:
        income_col = "income_log"

    plt.subplot(1, 2, 1)
    sns.histplot(transformed_df[income_col].to_numpy(), kde=True)
    plt.title(f"{income_col} Distribution (Transformed)")
    plt.xlabel(income_col)

    plt.subplot(1, 2, 2)
    # sns.boxplot(x="loan_approved", y=income_col, data=transformed_df)
    plt.title(f"{income_col} by Loan Approval (Transformed)")
    plt.savefig("output/transformed_data_visualization.png")
    print("Saved visualization of transformed data distribution")

    # Train model on transformed data
    print("\nTraining model on LogicApplier-transformed data...")

    # Prepare features (exclude target and text column)
    X_transformed = transformed_df[
        transformed_df.columns.difference(["loan_approved", "loan_description"])
    ]
    y_transformed = transformed_df["loan_approved"]

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_transformed, y_transformed, test_size=0.2, random_state=42
    )

    transformed_accuracy, transformed_importances = evaluate_model(
        X_train_t, X_test_t, y_train_t, y_test_t
    )

    print(f"\nTransformed data model accuracy: {transformed_accuracy:.4f}")
    print(f"Improvement: {(transformed_accuracy - baseline_accuracy) * 100:.2f}%")

    # Show top features after transformation
    print("\nTop 10 features by importance after transformation:")
    print(transformed_importances.head(10))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = transformed_importances.head(15)
    sns.barplot(x="Importance", y="Feature", data=top_features)
    plt.title("Top 15 Features by Importance")
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")
    print("Saved feature importance visualization")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
