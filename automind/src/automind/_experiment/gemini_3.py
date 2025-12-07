from datetime import datetime

import pandas as pd

from automind.data.dataset import AvailableDataset, load_data
from automind.data_utils import dataframe_to_csv_with_mkdir

# 1. Load Data
df_patients = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
)

df_encounters = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_encounters"]
)

df_conditions = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
)

# 2. Preprocessing & Feature Engineering

# --- Patients Level ---
# Calculate Age
df_patients["BIRTHDATE"] = pd.to_datetime(df_patients["BIRTHDATE"])
current_year = datetime.now().year
df_patients["AGE"] = current_year - df_patients["BIRTHDATE"].dt.year

# Handle Deceased (Feature: IS_DECEASED)
df_patients["IS_DECEASED"] = df_patients["DEATHDATE"].notnull().astype(int)

# Select base columns
base_cols = [
    "Id",
    "MARITAL",
    "RACE",
    "GENDER",
    "AGE",
    "IS_DECEASED",
    "HEALTHCARE_EXPENSES",
]
df_main = df_patients[base_cols].copy()

# --- Encounters Aggregation ---
# Count encounters per patient
encounter_counts = (
    df_encounters.groupby("PATIENT").size().reset_index(name="ENCOUNTER_COUNT")
)

# Calculate average encounter duration (in minutes)
df_encounters["START"] = pd.to_datetime(df_encounters["START"])
df_encounters["STOP"] = pd.to_datetime(df_encounters["STOP"])
df_encounters["DURATION_MINS"] = (
    df_encounters["STOP"] - df_encounters["START"]
).dt.total_seconds() / 60
encounter_duration = (
    df_encounters.groupby("PATIENT")["DURATION_MINS"]
    .mean()
    .reset_index(name="AVG_ENCOUNTER_DURATION")
)

# Merge encounter features
df_main = df_main.merge(
    encounter_counts, left_on="Id", right_on="PATIENT", how="left"
)
df_main = df_main.merge(
    encounter_duration, left_on="Id", right_on="PATIENT", how="left"
)

# --- Conditions Aggregation ---
# Count unique conditions per patient
condition_counts = (
    df_conditions.groupby("PATIENT")["DESCRIPTION"]
    .nunique()
    .reset_index(name="CONDITION_COUNT")
)

# Merge condition features
df_main = df_main.merge(
    condition_counts, left_on="Id", right_on="PATIENT", how="left"
)

# 3. Data Cleaning

# Fill Missing Values (Numerical -> 0, Categorical -> 'Unknown' or Mode)
df_main["ENCOUNTER_COUNT"] = df_main["ENCOUNTER_COUNT"].fillna(0)
df_main["AVG_ENCOUNTER_DURATION"] = df_main["AVG_ENCOUNTER_DURATION"].fillna(0)
df_main["CONDITION_COUNT"] = df_main["CONDITION_COUNT"].fillna(0)
df_main["MARITAL"] = df_main["MARITAL"].fillna("Unknown")
df_main["RACE"] = df_main["RACE"].fillna(df_main["RACE"].mode()[0])

# 4. Target Generation
# Define High Healthcare Expenses as top 25% (75th percentile)
threshold = df_main["HEALTHCARE_EXPENSES"].quantile(0.75)
df_main["Target"] = (df_main["HEALTHCARE_EXPENSES"] >= threshold).astype(int)

# Remove Leakage columns (The raw expense amount) and identifiers
cols_to_drop = [
    "Id",
    "PATIENT",
    "PATIENT_x",
    "PATIENT_y",
    "HEALTHCARE_EXPENSES",
]
# Note: Cleaning up merge artifacts if any
cols_to_drop = [c for c in cols_to_drop if c in df_main.columns]
df_main = df_main.drop(columns=cols_to_drop)

# 5. Encoding
# One-Hot Encoding for categorical variables
df_final = pd.get_dummies(
    df_main, columns=["MARITAL", "RACE", "GENDER"], drop_first=True
)

# Ensure Target is the last column for consistency
target_col = df_final.pop("Target")
df_final["Target"] = target_col

# 6. Output
dataframe_to_csv_with_mkdir(
    df_final, "src/automind/_experiment/output", "gemini_3.csv"
)
