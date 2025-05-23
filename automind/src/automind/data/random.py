import numpy as np
import pandas as pd


def create_random_synthetic_dataset(rows=1000):
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
