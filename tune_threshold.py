
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

# Create output folder if needed
Path("outputs").mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv("data/synthetic_fraud_dataset_v2.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Extract date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Hour"] = df["Date"].dt.hour

# Exclude identifiers, target, date, and potentially leaky feature
X = df.drop(
    columns=[
        "Transaction_ID",
        "User_ID",
        "Date",
        "Fraud_Label",
        "Amount_Zscore_User",
    ]
)
y = df["Fraud_Label"]

# Split: 60% training, 20% validation, 20% final testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42,
    )),
])

# Train only on the training split
model.fit(X_train, y_train)

# Choose threshold using validation data only
val_probabilities = model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in np.arange(0.10, 0.81, 0.05):
    predictions = (val_probabilities >= threshold).astype(int)

    threshold_results.append({
        "Threshold": round(float(threshold), 2),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "Flagged_Transactions": int(predictions.sum()),
    })

results_df = pd.DataFrame(threshold_results)
results_df.to_csv("outputs/threshold_comparison.csv", index=False)

print("\nVALIDATION THRESHOLD COMPARISON")
print(results_df.round(4).to_string(index=False))

# Select threshold using validation F1
best = results_df.loc[results_df["F1"].idxmax()]
best_threshold = float(best["Threshold"])

print("\nSelected threshold (validation F1):", best_threshold)

# Evaluate once on the untouched test set
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Test rows:", len(X_test))
print("Actual fraud cases:", int(y_test.sum()))
print("Selected threshold:", best_threshold)
print("Precision:", round(
    precision_score(y_test, test_predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, test_predictions, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, test_predictions, zero_division=0), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_test, test_probabilities), 4
))
print("PR-AUC:", round(
    average_precision_score(y_test, test_probabilities), 4
))
print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nSaved validation threshold results to outputs/threshold_comparison.csv")
