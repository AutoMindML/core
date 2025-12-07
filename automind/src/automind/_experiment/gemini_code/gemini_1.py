from datetime import datetime

import pandas as pd

# automind specific imports
from automind.data.dataset import AvailableDataset, load_data
from automind.data_utils import dataframe_to_csv_with_mkdir

# Input Loading
df_patients = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
)

df_encounters = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_encounters"]
)

df_conditions = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
)

# --- Gemini Method Implementation ---


def process_data(patients, encounters, conditions):
    # 1. Feature Engineering: Demographics (from Patients)
    # Calculate Age
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
    current_year = datetime.now().year
    patients["AGE"] = current_year - patients["BIRTHDATE"].dt.year

    # Handle Missing Values (Categorical)
    patients["MARITAL"] = patients["MARITAL"].fillna("Unknown")

    # Encode Binary/Categorical Variables
    # Gender: M=0, F=1
    patients["GENDER_ENC"] = patients["GENDER"].map({"M": 0, "F": 1}).fillna(-1)

    # Simple encoding for Race/Ethnicity (Top categories)
    patients["IS_WHITE"] = (patients["RACE"] == "white").astype(int)

    # 2. Feature Engineering: Medical History Aggregation (from Conditions)
    # Count number of conditions per patient
    conditions_count = (
        conditions.groupby("PATIENT").size().reset_index(name="CONDITION_COUNT")
    )

    # Check for specific high-impact keywords (Simplified Bag-of-Words approach for medical history)
    # Identifying chronic flags based on description could be useful
    conditions["IS_CHRONIC"] = (
        conditions["DESCRIPTION"]
        .str.contains("Chronic|Diabetes|Hypertension", case=False, na=False)
        .astype(int)
    )
    chronic_count = (
        conditions.groupby("PATIENT")["IS_CHRONIC"]
        .sum()
        .reset_index(name="CHRONIC_CONDITION_COUNT")
    )

    # 3. Feature Engineering: Utilization Aggregation (from Encounters)
    # Count number of encounters
    encounters_count = (
        encounters.groupby("PATIENT").size().reset_index(name="ENCOUNTER_COUNT")
    )

    # Average base cost per encounter (Proxy for intensity of care)
    encounters["BASE_ENCOUNTER_COST"] = pd.to_numeric(
        encounters["BASE_ENCOUNTER_COST"], errors="coerce"
    ).fillna(0)
    avg_encounter_cost = (
        encounters.groupby("PATIENT")["BASE_ENCOUNTER_COST"]
        .mean()
        .reset_index(name="AVG_ENCOUNTER_COST")
    )

    # 4. Data Fusion
    df_final = patients.merge(
        conditions_count, left_on="Id", right_on="PATIENT", how="left"
    )
    df_final = df_final.merge(
        chronic_count, left_on="Id", right_on="PATIENT", how="left"
    )
    df_final = df_final.merge(
        encounters_count, left_on="Id", right_on="PATIENT", how="left"
    )

    # FIX
    df_final = pd.DataFrame(df_final)
    df_final.drop(columns=["PATIENT_x"], inplace=True)
    df_final = df_final.merge(
        avg_encounter_cost, left_on="Id", right_on="PATIENT", how="left"
    )

    # Fill NaN for features resulting from merge (implies 0 count or cost)
    fill_cols = [
        "CONDITION_COUNT",
        "CHRONIC_CONDITION_COUNT",
        "ENCOUNTER_COUNT",
        "AVG_ENCOUNTER_COST",
    ]
    df_final[fill_cols] = df_final[fill_cols].fillna(0)

    # 5. Target Creation: High Healthcare Expenses
    # Define High Expense as > 75th percentile
    threshold = df_final["HEALTHCARE_EXPENSES"].quantile(0.75)
    df_final["Target"] = (df_final["HEALTHCARE_EXPENSES"] > threshold).astype(
        int
    )

    # 6. Final Cleanup
    # Select feature columns and Target
    feature_cols = [
        "AGE",
        "GENDER_ENC",
        "IS_WHITE",
        "CONDITION_COUNT",
        "CHRONIC_CONDITION_COUNT",
        "ENCOUNTER_COUNT",
        "AVG_ENCOUNTER_COST",
        "Target",
    ]

    return df_final[feature_cols]


# Execute Logic
df_processed = process_data(df_patients, df_encounters, df_conditions)

# Output
dataframe_to_csv_with_mkdir(
    df_processed, "src/automind/_experiment/output", "gemini_1.csv"
)
