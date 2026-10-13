
import pandas as pd
from pathlib import Path

file_path = Path("data/synthetic_fraud_dataset_v2.csv")

print("=" * 55)
print("   FRAUDGUARD AI - NEW DATASET INSPECTION")
print("=" * 55)

if not file_path.exists():
    print("ERROR: Dataset not found!")
    print("Check that the CSV is inside the data folder.")
else:
    df = pd.read_csv(file_path)

    print("\n1. DATASET SIZE")
    print("Rows and columns:", df.shape)

    print("\n2. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n3. MISSING VALUES")
    print(df.isnull().sum())

    print("\n4. DUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\n5. FRAUD LABEL DISTRIBUTION")
    if "Fraud_Label" in df.columns:
        print(df["Fraud_Label"].value_counts())
        print("\nFraud percentages:")
        print((df["Fraud_Label"].value_counts(normalize=True) * 100).round(2))

    print("\n6. FIRST FIVE ROWS")
    print(df.head().to_string())

    print("\n7. DATA TYPES")
    print(df.dtypes)

    print("\nINSPECTION COMPLETED")
