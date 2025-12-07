import numpy as np
import pandas as pd

from automind.data.dataset import AvailableDataset, load_data
from automind.data_utils import dataframe_to_csv_with_mkdir

# -------------------------------------------------------------------------
# 1. Data Loading
# -------------------------------------------------------------------------
df_patients = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
)

df_encounters = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_encounters"]
)

df_conditions = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
)

# -------------------------------------------------------------------------
# 2. Preprocessing & Feature Engineering Methods
# -------------------------------------------------------------------------


def process_patients(df):
    # Convert dates
    df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"])

    # Calculate Age (Assuming current reference year is 2020 for this dataset context)
    df["AGE"] = 2020 - df["BIRTHDATE"].dt.year

    # Create Target: High Healthcare Expenses (Top 25%)
    # Using strict inequality > quantile(0.75)
    threshold = df["HEALTHCARE_EXPENSES"].quantile(0.75)
    df["Target"] = (df["HEALTHCARE_EXPENSES"] > threshold).astype(int)

    # Drop leakage columns (Expenses is the target source, Coverage is highly correlated)
    # Drop identifiers and raw dates
    drop_cols = [
        "Id",
        "BIRTHDATE",
        "DEATHDATE",
        "FIRST",
        "LAST",
        "MAIDEN",
        "PASSPORT",
        "DRIVERS",
        "SSN",
        "ADDRESS",
        "LAT",
        "LON",
        "HEALTHCARE_EXPENSES",
        "HEALTHCARE_COVERAGE",
    ]

    # Keep 'Id' temporarily for merging, drop later
    cols_to_keep = [c for c in df.columns if c not in drop_cols]
    return df[cols_to_keep], df[
        [
            "Id",
            "Target",
            "AGE",
            "MARITAL",
            "RACE",
            "ETHNICITY",
            "GENDER",
            "CITY",
            "COUNTY",
        ]
    ]


def feature_engineer_encounters(df_enc, df_pat_ids):
    # Aggregation by PATIENT
    # 1. Count of encounters
    # 2. Average base encounter cost
    # 3. Time since first encounter (tenure) - optional, skipping for simplicity

    enc_agg = (
        df_enc.groupby("PATIENT")
        .agg(
            {
                "Id": "count",
                "BASE_ENCOUNTER_COST": "mean",
                "TOTAL_CLAIM_COST": "sum",
            }
        )
        .reset_index()
    )

    enc_agg.rename(
        columns={
            "PATIENT": "Id",
            "Id": "ENC_COUNT",
            "BASE_ENCOUNTER_COST": "AVG_ENC_COST",
            "TOTAL_CLAIM_COST": "TOTAL_HISTORICAL_COST",
        },
        inplace=True,
    )

    return enc_agg


def feature_engineer_conditions(df_con):
    # Aggregation by PATIENT
    # 1. Total count of conditions
    # 2. Specific flags for chronic/high-cost conditions

    # Create flags
    df_con["IS_DIABETES"] = (
        df_con["DESCRIPTION"]
        .str.contains("Diabetes|diabetic", case=False, na=False)
        .astype(int)
    )
    df_con["IS_HYPERTENSION"] = (
        df_con["DESCRIPTION"]
        .str.contains("Hypertension", case=False, na=False)
        .astype(int)
    )
    df_con["IS_COVID"] = (
        df_con["DESCRIPTION"]
        .str.contains("COVID-19", case=False, na=False)
        .astype(int)
    )
    df_con["IS_OBESITY"] = (
        df_con["DESCRIPTION"]
        .str.contains("Obesity|Body mass index 30+", case=False, na=False)
        .astype(int)
    )

    con_agg = (
        df_con.groupby("PATIENT")
        .agg(
            {
                "DESCRIPTION": "count",
                "IS_DIABETES": "max",
                "IS_HYPERTENSION": "max",
                "IS_COVID": "max",
                "IS_OBESITY": "max",
            }
        )
        .reset_index()
    )

    con_agg.rename(
        columns={"PATIENT": "Id", "DESCRIPTION": "CONDITION_COUNT"},
        inplace=True,
    )

    return con_agg


def clean_and_impute(df):
    # Fill numeric NaNs with 0 (implies no history in joined tables)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Fill categorical NaNs with 'Unknown' or Mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(
            df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        )

    return df


def encode_features(df):
    # One-Hot Encoding for categorical variables
    # Drop Id as it's no longer needed for merging
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    df = pd.get_dummies(df, drop_first=True)
    return df


# -------------------------------------------------------------------------
# 3. Execution Pipeline
# -------------------------------------------------------------------------

# A. Process Patients (Base Table)
_, df_base = process_patients(df_patients)

# B. Process Encounters
df_enc_feats = feature_engineer_encounters(df_encounters, df_base[["Id"]])

# C. Process Conditions
df_con_feats = feature_engineer_conditions(df_conditions)

# D. Merge All
# Left join to keep all patients, even if no encounters/conditions found
df_merged = df_base.merge(df_enc_feats, on="Id", how="left")
df_merged = df_merged.merge(df_con_feats, on="Id", how="left")

# E. Clean & Impute
df_clean = clean_and_impute(df_merged)

# F. Final Encoding
df_final = encode_features(df_clean)

# Ensure Target is the last column or explicitly accessible, though not strictly required by XGBoost usually
# but good for readability. Here we just ensure it exists.

# -------------------------------------------------------------------------
# 4. Output
# -------------------------------------------------------------------------
dataframe_to_csv_with_mkdir(
    df_final, "src/automind/_experiment/output", "gemini_2.csv"
)
