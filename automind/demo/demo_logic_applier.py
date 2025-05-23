from automind.data.random import create_random_synthetic_dataset
from automind.data_utils.logic_applier import LogicApplier


def main():
    print("=== LogicApplier Demo ===")
    print(
        "This demo showcases how LogicApplier applies LLM recommendations for data preparation"
    )

    # Create synthetic dataset with issues
    print("\nCreating synthetic loan approval dataset with data quality issues...")
    df = create_random_synthetic_dataset()
    print(f"Dataset shape: {df.shape}")

    # Display dataset summary
    print("\nOriginal dataset summary:")
    print(df.info())
    print("\nMissing values by column:")
    print(df.isna().sum())

    # Apply LogicApplier
    print("\nApplying LogicApplier to clean data and engineer features...")

    # Initialize LogicApplier
    logic_applier = LogicApplier(df, target_column="loan_approved")

    logic_applier.apply_pipeline()
    transformed_df = logic_applier.get_transformed_df()

    print("\nTransformations applied:")
    for i, transformation in enumerate(logic_applier.get_transformations_log(), 1):
        print(f"{i}. {transformation}")

    print("\n", transformed_df)


if __name__ == "__main__":
    main()
