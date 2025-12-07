import pandas as pd

from automind.data.dataset import AvailableDataset, load_data
from automind.data_utils import dataframe_to_csv_with_mkdir

# input
df_patients = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
)

df_encounters = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_encounters"]
)

df_conditions = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
)

# --- Gemini Method Start ---

# 1. Feature Engineering: Encounters
# Aggregate encounter data by patient to get usage patterns
enc_stats = (
    df_encounters.groupby("PATIENT")
    .agg({"Id": "count", "TOTAL_CLAIM_COST": "mean", "PAYER_COVERAGE": "mean"})
    .reset_index()
)
enc_stats.columns = [
    "Id",
    "encounter_count",
    "avg_encounter_cost",
    "avg_payer_coverage",
]

# 2. Feature Engineering: Conditions
# Count number of conditions per patient as a proxy for comorbidity
cond_stats = (
    df_conditions.groupby("PATIENT")["DESCRIPTION"].nunique().reset_index()
)
cond_stats.columns = ["PATIENT", "condition_count"]

# 3. Data Fusion
# Merge aggregations back to the main patient table
df = df_patients.merge(enc_stats, left_on="Id", right_on="Id", how="left")
df = df.merge(cond_stats, left_on="Id", right_on="PATIENT", how="left")

# 4. Target Generation (High Healthcare Expenses Risk)
# Define High Risk as top 25% of expenses (75th percentile)
threshold = df["HEALTHCARE_EXPENSES"].quantile(0.75)
df["Target"] = (df["HEALTHCARE_EXPENSES"] > threshold).astype(int)

# Drop the original label to prevent leakage
df = df.drop(columns=["HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"])

# 5. Cleaning & Transformation
# Calculate Age
df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"], errors="coerce")
df["age"] = (pd.Timestamp.now() - df["BIRTHDATE"]).dt.days // 365

# Impute missing numerical values with median
num_cols = [
    "age",
    "encounter_count",
    "avg_encounter_cost",
    "avg_payer_coverage",
    "condition_count",
]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle Categorical Columns (Simple Label Encoding for tree-based models)
cat_cols = ["MARITAL", "RACE", "ETHNICITY", "GENDER", "COUNTY"]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
    df[col] = df[col].astype("category").cat.codes

# Drop unused or identifier columns
drop_cols = [
    "Id",
    "BIRTHDATE",
    "DEATHDATE",
    "PREFIX",
    "BIRTHPLACE",
    "CITY",
    "PATIENT",
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Final check to ensure Target is the last column (optional but good for consistency)
cols = [c for c in df.columns if c != "Target"] + ["Target"]
df = df[cols]

# --- Gemini Method End ---

# output
dataframe_to_csv_with_mkdir(
    df, "src/automind/_experiment/output", "gemini_4.csv"
)
