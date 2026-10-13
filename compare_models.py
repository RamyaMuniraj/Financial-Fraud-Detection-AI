import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# FRAUDGUARD AI - RANDOM FOREST
# ==========================================

print("\n========== FRAUDGUARD AI - RANDOM FOREST ==========")

# Load dataset
df = pd.read_csv("data/financial_transactions.csv")

# Remove identifiers
df = df.drop(columns=["Transaction_ID", "User_ID"])

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

df = df.drop(columns=["Date"])

# Separate features and target
X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]

# Identify feature types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Random Forest
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Train
print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Random Forest training completed.")

# Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\n========== RANDOM FOREST COMPLETE ==========")
print("\n========== CATEGORICAL FRAUD ANALYSIS ==========")

categorical_check = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type"
]

for column in categorical_check:

    print(f"\n--- {column} ---")

    result = (
        df.groupby(column)["Fraud_Label"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    result["fraud_rate_percent"] = result["mean"] * 100

    print(result)

print("\n========== CATEGORICAL FRAUD ANALYSIS COMPLETE ==========")
print("\n========== FEATURE IMPORTANCE ANALYSIS ==========")

# Get the trained Random Forest
rf_model = model.named_steps["classifier"]

# Get transformed feature names
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Create feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 20 Features:")
print(importance_df.head(20).to_string(index=False))

print("\n========== FEATURE IMPORTANCE COMPLETE ==========")
print("\n========== NUMERICAL FRAUD COMPARISON ==========")

numerical_check = [
    "Transaction_Amount",
    "Account_Balance",
    "Daily_Transaction_Count",
    "Card_Age"
]

for column in numerical_check:

    legitimate = df[df["Fraud_Label"] == 0][column]
    fraudulent = df[df["Fraud_Label"] == 1][column]

    print(f"\n--- {column} ---")

    print(f"Legitimate Mean : {legitimate.mean():.4f}")
    print(f"Fraud Mean      : {fraudulent.mean():.4f}")

    print(f"Legitimate Median : {legitimate.median():.4f}")
    print(f"Fraud Median      : {fraudulent.median():.4f}")

    print(f"Legitimate Std : {legitimate.std():.4f}")
    print(f"Fraud Std      : {fraudulent.std():.4f}")

print("\n========== NUMERICAL FRAUD COMPARISON COMPLETE ==========")
print("\n========== BEHAVIORAL FEATURE ENGINEERING ==========")

# Amount relative to available account balance
df["Amount_Balance_Ratio"] = (
    df["Transaction_Amount"] /
    (df["Account_Balance"] + 1)
)

# Transaction activity intensity
df["Transaction_Activity_Score"] = (
    df["Daily_Transaction_Count"] *
    df["Transaction_Amount"]
)

# Combined behavioral risk
df["Behavioral_Risk_Score"] = (
    df["Amount_Balance_Ratio"] *
    (df["Daily_Transaction_Count"] + 1)
)

# Interaction between previous fraud and transaction activity
df["Fraud_History_Activity"] = (
    df["Previous_Fraudulent_Activity"] *
    df["Daily_Transaction_Count"]
)

# Transaction amount relative to card age
df["Amount_CardAge_Ratio"] = (
    df["Transaction_Amount"] /
    (df["Card_Age"] + 1)
)

print("\nNew Behavioral Features:")

new_features = [
    "Amount_Balance_Ratio",
    "Transaction_Activity_Score",
    "Behavioral_Risk_Score",
    "Fraud_History_Activity",
    "Amount_CardAge_Ratio"
]

print(new_features)

print("\nBehavioral Feature Statistics:")
print(
    df[
        new_features
    ].describe()
)

print("\n========== BEHAVIORAL FEATURE ENGINEERING COMPLETE ==========")
print("\n========== BEHAVIORAL FEATURE MODEL TEST ==========")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Features including behavioral features
behavioral_features = [
    "Transaction_Amount",
    "Account_Balance",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Age",
    "Amount_Balance_Ratio",
    "Transaction_Activity_Score",
    "Behavioral_Risk_Score",
    "Fraud_History_Activity",
    "Amount_CardAge_Ratio",
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type",
    "Year",
    "Month",
    "Day",
    "DayOfWeek"
]

X_behavioral = df[behavioral_features]
y_behavioral = df["Fraud_Label"]

numerical_features = [
    "Transaction_Amount",
    "Account_Balance",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Age",
    "Amount_Balance_Ratio",
    "Transaction_Activity_Score",
    "Behavioral_Risk_Score",
    "Fraud_History_Activity",
    "Amount_CardAge_Ratio",
    "Year",
    "Month",
    "Day",
    "DayOfWeek"
]

categorical_features = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type"
]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_behavioral,
    y_behavioral,
    test_size=0.20,
    random_state=42,
    stratify=y_behavioral
)

preprocessor_b = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

behavioral_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_b),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

behavioral_model.fit(X_train_b, y_train_b)

y_pred_b = behavioral_model.predict(X_test_b)
y_prob_b = behavioral_model.predict_proba(X_test_b)[:, 1]

print("\nBehavioral Feature Model Results:")
print("Accuracy :", round(accuracy_score(y_test_b, y_pred_b), 4))
print("Precision:", round(precision_score(y_test_b, y_pred_b, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test_b, y_pred_b, zero_division=0), 4))
print("F1 Score :", round(f1_score(y_test_b, y_pred_b, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test_b, y_prob_b), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_b, y_pred_b))

print("\n========== BEHAVIORAL FEATURE MODEL TEST COMPLETE ==========")
print("\n========== FRAUD LABEL DEPENDENCY CHECK ==========")

# Correlation of engineered numerical features with fraud
check_features = [
    "Transaction_Amount",
    "Account_Balance",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Age",
    "Amount_Balance_Ratio",
    "Transaction_Activity_Score",
    "Behavioral_Risk_Score",
    "Fraud_History_Activity",
    "Amount_CardAge_Ratio"
]

print("\nCorrelation with Fraud_Label:")

correlations = (
    df[check_features + ["Fraud_Label"]]
    .corr()["Fraud_Label"]
    .drop("Fraud_Label")
    .sort_values(key=abs, ascending=False)
)

print(correlations)

print("\nFraud rate by transaction amount quartile:")

df["Amount_Quartile"] = pd.qcut(
    df["Transaction_Amount"],
    q=4,
    duplicates="drop"
)

print(
    df.groupby(
        "Amount_Quartile",
        observed=True
    )["Fraud_Label"].mean()
)

print("\nFraud rate by account balance quartile:")

df["Balance_Quartile"] = pd.qcut(
    df["Account_Balance"],
    q=4,
    duplicates="drop"
)

print(
    df.groupby(
        "Balance_Quartile",
        observed=True
    )["Fraud_Label"].mean()
)

print("\n========== FRAUD LABEL DEPENDENCY CHECK COMPLETE ==========")
print("\n========== USER HISTORY SIGNAL TEST ==========")

# Work on a copy and sort chronologically
history_df = pd.read_csv("data/financial_transactions.csv")

history_df["Date"] = pd.to_datetime(history_df["Date"])

history_df = history_df.sort_values(
    ["User_ID", "Date", "Transaction_ID"]
).reset_index(drop=True)

# Previous transactions for each user
history_df["Previous_User_Transactions"] = (
    history_df.groupby("User_ID")
    .cumcount()
)

# Previous fraud count for each user
history_df["Previous_User_Fraud_Count"] = (
    history_df.groupby("User_ID")["Fraud_Label"]
    .transform(lambda x: x.shift(1).fillna(0).cumsum())
)

# Historical fraud rate
history_df["Previous_User_Fraud_Rate"] = (
    history_df["Previous_User_Fraud_Count"] /
    history_df["Previous_User_Transactions"].replace(0, 1)
)

print("\nUser history feature statistics:")
print(
    history_df[
        [
            "Previous_User_Transactions",
            "Previous_User_Fraud_Count",
            "Previous_User_Fraud_Rate"
        ]
    ].describe()
)

print("\nCorrelation with Fraud_Label:")

print(
    history_df[
        [
            "Previous_User_Transactions",
            "Previous_User_Fraud_Count",
            "Previous_User_Fraud_Rate",
            "Fraud_Label"
        ]
    ]
    .corr()["Fraud_Label"]
    .drop("Fraud_Label")
    .sort_values(key=abs, ascending=False)
)

print("\nFraud rate by previous user fraud rate:")

history_df["History_Risk_Group"] = pd.cut(
    history_df["Previous_User_Fraud_Rate"],
    bins=[-0.001, 0, 0.25, 0.50, 0.75, 1.0],
    labels=[
        "No Previous Fraud",
        "Low History Risk",
        "Medium History Risk",
        "High History Risk",
        "Very High History Risk"
    ]
)

print(
    history_df.groupby(
        "History_Risk_Group",
        observed=True
    )["Fraud_Label"].agg(
        ["count", "mean"]
    )
)

print("\n========== USER HISTORY SIGNAL TEST COMPLETE ==========")