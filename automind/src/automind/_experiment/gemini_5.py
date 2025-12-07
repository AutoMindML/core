import numpy as np
import pandas as pd

from automind.data.dataset import AvailableDataset, load_data
from automind.data_utils import dataframe_to_csv_with_mkdir

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
df_patients = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
)

df_encounters = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_encounters"]
)

df_conditions = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
)

# ---------------------------------------------------------
# 2. Preprocessing & Feature Engineering Methods
# ---------------------------------------------------------


def process_patients(df):
    # Calculate Age
    df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"])
    now = pd.to_datetime("now")
    df["AGE"] = (now - df["BIRTHDATE"]).dt.days // 365

    # Drop leakage or irrelevant columns
    # Dropping expenses here to avoid leakage, will extract target later
    cols_to_drop = [
        "BIRTHDATE",
        "DEATHDATE",
        "PREFIX",
        "FIRST",
        "LAST",
        "MAIDEN",
        "BIRTHPLACE",
        "ADDRESS",
        "LAT",
        "LON",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


def process_encounters(df):
    # Aggregate encounter metrics per patient
    df["START"] = pd.to_datetime(df["START"])
    df["STOP"] = pd.to_datetime(df["STOP"])
    df["DURATION_MINS"] = (df["STOP"] - df["START"]).dt.total_seconds() / 60.0

    # Aggregations
    agg_funcs = {
        "Id": "count",  # Total encounters
        "TOTAL_CLAIM_COST": ["sum", "mean"],
        "PAYER_COVERAGE": "mean",
        "DURATION_MINS": "mean",
    }

    enc_agg = df.groupby("PATIENT").agg(agg_funcs)
    enc_agg.columns = ["_".join(col).strip() for col in enc_agg.columns.values]
    enc_agg = enc_agg.rename(columns={"Id_count": "ENC_COUNT"})

    return enc_agg


def process_conditions(df):
    # Feature Engineering: Count conditions and specific flags
    # Simple count
    cond_count = df.groupby("PATIENT").size().rename("CONDITION_COUNT")

    # Vectorize common high-risk keywords (Simple Bag of Words approach for top risks)
    # Checking for chronic implications
    df["is_chronic"] = (
        df["DESCRIPTION"].astype(str).str.lower().str.contains("chronic")
    )
    df["is_diabetes"] = (
        df["DESCRIPTION"].astype(str).str.lower().str.contains("diabetes")
    )
    df["is_cardiac"] = (
        df["DESCRIPTION"].astype(str).str.lower().str.contains("heart|cardiac")
    )

    risk_agg = df.groupby("PATIENT")[
        ["is_chronic", "is_diabetes", "is_cardiac"]
    ].sum()

    return pd.concat([cond_count, risk_agg], axis=1)


# ---------------------------------------------------------
# 3. Execution Pipeline
# ---------------------------------------------------------

# A. Process Sub-tables
df_enc_features = process_encounters(df_encounters)
df_cond_features = process_conditions(df_conditions)

# B. Process Main Patient Table
df_clean = process_patients(df_patients)

# C. Merge Data Fusion
# Left join to keep all patients
df_merged = df_clean.merge(
    df_enc_features, left_on="Id", right_index=True, how="left"
)
df_merged = df_merged.merge(
    df_cond_features, left_on="Id", right_index=True, how="left"
)

# D. Handle Missing Values (Post-Merge)
# Numeric fills with 0 (assuming no encounter/condition means 0 count/cost)
num_cols = df_merged.select_dtypes(include=[np.number]).columns
df_merged[num_cols] = df_merged[num_cols].fillna(0)

# Categorical fills
cat_cols = df_merged.select_dtypes(include=["object", "category"]).columns
# Exclude Id columns from filling if they exist
cat_cols = [c for c in cat_cols if c not in ["Id", "PATIENT"]]
df_merged[cat_cols] = df_merged[cat_cols].fillna("Unknown")

# E. Categorical Encoding (One-Hot)
# Exclude Id and Target source
cols_to_encode = [c for c in cat_cols if c != "Id"]
df_final = pd.get_dummies(df_merged, columns=cols_to_encode, drop_first=True)

# F. Target Creation (High Healthcare Expenses)
# Define "High" as > 75th percentile
target_col = "HEALTHCARE_EXPENSES"
threshold = df_final[target_col].quantile(0.75)
df_final["Target"] = (df_final[target_col] > threshold).astype(int)

# G. Final Cleanup
# Drop original ID and Expenses column (to prevent target leakage)
df_final = df_final.drop(columns=["Id", "HEALTHCARE_EXPENSES"])

# ---------------------------------------------------------
# 4. Output
# ---------------------------------------------------------
dataframe_to_csv_with_mkdir(
    df_final, "src/automind/_experiment/output", "gemini_5.csv"
)
