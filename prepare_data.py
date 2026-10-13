import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# FRAUDGUARD AI - DATA PREPARATION
# ==========================================

# Load dataset
file_path = "data/financial_transactions.csv"
df = pd.read_csv(file_path)

print("\n========== DATA PREPARATION ==========")

# ------------------------------------------
# 1. Remove identifier columns
# ------------------------------------------

# Transaction_ID and User_ID are identifiers,
# not useful direct predictive features.
df = df.drop(columns=["Transaction_ID", "User_ID"])

# ------------------------------------------
# 2. Convert Date column
# ------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

# Extract useful date-based features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Remove original date column
df = df.drop(columns=["Date"])

# ------------------------------------------
# 3. Separate features and target
# ------------------------------------------

X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print("Fraud_Label")

# ------------------------------------------
# 4. Identify numerical and categorical data
# ------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# ------------------------------------------
# 5. Train/Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Set:")
print(X_train.shape)

print("\nTesting Set:")
print(X_test.shape)

print("\nTraining Fraud Distribution:")
print(y_train.value_counts())

print("\nTesting Fraud Distribution:")
print(y_test.value_counts())

print("\n========== DATA PREPARATION COMPLETE ==========")