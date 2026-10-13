import pandas as pd

# Load the financial transaction dataset
file_path = "data/financial_transactions.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET OVERVIEW ==========")

# Number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe(include="all"))

print("\n========== INSPECTION COMPLETE ==========")
print("\n========== FRAUD DISTRIBUTION ==========")

fraud_counts = df["Fraud_Label"].value_counts()

print("\nFraud Label Counts:")
print(fraud_counts)

print("\nFraud Label Percentages:")
print(df["Fraud_Label"].value_counts(normalize=True) * 100)

print("\n0 = Legitimate Transaction")
print("1 = Fraudulent Transaction")

print("\n========== FRAUD DISTRIBUTION COMPLETE ==========")
print("\n========== FRAUD PATTERN ANALYSIS ==========")

# 1. Average transaction amount by fraud status
print("\nAverage Transaction Amount:")
print(df.groupby("Fraud_Label")["Transaction_Amount"].mean())

# 2. Average account balance by fraud status
print("\nAverage Account Balance:")
print(df.groupby("Fraud_Label")["Account_Balance"].mean())

# 3. Average daily transaction count by fraud status
print("\nAverage Daily Transaction Count:")
print(df.groupby("Fraud_Label")["Daily_Transaction_Count"].mean())

# 4. Average card age by fraud status
print("\nAverage Card Age:")
print(df.groupby("Fraud_Label")["Card_Age"].mean())

# 5. Previous fraudulent activity
print("\nPrevious Fraudulent Activity:")
print(pd.crosstab(
    df["Previous_Fraudulent_Activity"],
    df["Fraud_Label"],
    normalize="index"
) * 100)

# 6. Fraud rate by transaction type
print("\nFraud Rate by Transaction Type:")
print(
    df.groupby("Transaction_Type")["Fraud_Label"]
    .mean()
    .sort_values(ascending=False) * 100
)

# 7. Fraud rate by device type
print("\nFraud Rate by Device Type:")
print(
    df.groupby("Device_Type")["Fraud_Label"]
    .mean()
    .sort_values(ascending=False) * 100
)

# 8. Fraud rate by card type
print("\nFraud Rate by Card Type:")
print(
    df.groupby("Card_Type")["Fraud_Label"]
    .mean()
    .sort_values(ascending=False) * 100
)

print("\n========== FRAUD PATTERN ANALYSIS COMPLETE ==========")
print("\n========== FEATURE CORRELATION ANALYSIS ==========")

# Select numerical columns
numeric_columns = [
    "Transaction_Amount",
    "Account_Balance",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Age",
    "Fraud_Label"
]

correlation = df[numeric_columns].corr()["Fraud_Label"].sort_values(
    ascending=False
)

print("\nCorrelation with Fraud_Label:")
print(correlation)

print("\n========== FEATURE CORRELATION ANALYSIS COMPLETE ==========")