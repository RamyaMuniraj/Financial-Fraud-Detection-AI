
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Load dataset and saved model
# --------------------------------------------------
DATA_PATH = "data/synthetic_fraud_dataset_v2.csv"
MODEL_PATH = "models/fraud_detection_model.joblib"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Hour"] = df["Date"].dt.hour

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

# --------------------------------------------------
# 2. Recreate the exact saved-model data split
# --------------------------------------------------
X = df.drop(
    columns=[
        "Transaction_ID",
        "User_ID",
        "Date",
        "Fraud_Label",
        "Amount_Zscore_User",
    ],
    errors="ignore",
)

y = df["Fraud_Label"]

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

# Use the exact features and fitted pipeline stored in the artifact.
X_test = X_test[features]

# --------------------------------------------------
# 3. Evaluate the saved model on the reserved test set
# --------------------------------------------------
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "Test_Rows": len(y_test),
    "Actual_Fraud_Cases": int(y_test.sum()),
    "Threshold": float(threshold),
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "F1_Score": f1_score(
        y_test, predictions, zero_division=0
    ),
    "ROC_AUC": roc_auc_score(y_test, probabilities),
    "PR_AUC": average_precision_score(y_test, probabilities),
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_test_metrics.csv"),
    index=False,
)

# Save confusion matrix with readable labels.
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

cm_df = pd.DataFrame(
    cm,
    index=["Actual_Legitimate", "Actual_Fraud"],
    columns=["Predicted_Legitimate", "Predicted_Fraud"],
)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)

# Save the detailed classification report.
report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0,
)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8",
) as file:
    file.write("SAVED MODEL - FINAL TEST EVALUATION\n")
    file.write("=" * 45 + "\n")
    file.write(f"Selected threshold: {threshold:.4f}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Validation rows: {len(X_val)}\n")
    file.write(f"Reserved test rows: {len(X_test)}\n\n")
    file.write(report)
    file.write("\n\nConfusion Matrix:\n")
    file.write(str(cm))
    file.write("\n")

# --------------------------------------------------
# 4. Display results
# --------------------------------------------------
print("\n" + "=" * 55)
print("FINAL TEST EVALUATION - SAVED FRAUD DETECTION MODEL")
print("=" * 55)

for metric, value in metrics.items():
    if isinstance(value, float):
        print(f"{metric:25s}: {value:.4f}")
    else:
        print(f"{metric:25s}: {value}")

print("\nConfusion Matrix:")
print(cm_df)

print("\nClassification Report:")
print(report)

print("\nSaved evaluation files:")
print(" - outputs/final_test_metrics.csv")
print(" - outputs/confusion_matrix.csv")
print(" - outputs/classification_report.txt")

print("\nEvaluation completed successfully.")
