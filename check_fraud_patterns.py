
import pandas as pd

df = pd.read_csv("data/synthetic_fraud_dataset_v2.csv")

print("=" * 55)
print("FRAUDGUARD AI - FRAUD PATTERN ANALYSIS")
print("=" * 55)

# 1. Fraud and legitimate transaction averages
print("\n1. Average values by fraud label")
print(
    df.groupby("Fraud_Label")[
        [
            "Transaction_Amount",
            "Account_Balance",
            "Txn_to_Balance_Pct",
            "Amount_Zscore_User",
            "Daily_Transaction_Count",
            "Card_Age",
        ]
    ].mean().round(4)
)

# 2. Fraud rate across feature groups
print("\n2. Fraud rate by previous fraudulent activity")
print(
    df.groupby("Previous_Fraudulent_Activity")["Fraud_Label"]
    .mean().mul(100).round(2)
)

# 3. Numeric feature correlations with fraud label
print("\n3. Numeric correlations with Fraud_Label")
numeric = df.select_dtypes(include="number")
print(
    numeric.corr()["Fraud_Label"]
    .drop("Fraud_Label")
    .sort_values(key=abs, ascending=False)
    .round(4)
)

# 4. Check date/time range
print("\n4. Date range")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print("Start:", df["Date"].min())
print("End:  ", df["Date"].max())
print("Invalid dates:", df["Date"].isna().sum())

print("\nANALYSIS COMPLETED")
