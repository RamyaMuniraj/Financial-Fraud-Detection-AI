
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/fraud_detection_model.joblib")
DATA_PATH = Path("data/synthetic_fraud_dataset_v2.csv")

# Check that the files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load saved model artifact
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

# Load a few transactions for testing
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Hour"] = df["Date"].dt.hour

# Use the same features and order as training
X_sample = df[features].head(10)

# Generate fraud probabilities and predictions
probabilities = model.predict_proba(X_sample)[:, 1]
predictions = (probabilities >= threshold).astype(int)

results = pd.DataFrame({
    "Transaction_Amount": df["Transaction_Amount"].head(10),
    "Fraud_Probability": probabilities.round(4),
    "Prediction": predictions,
})

results["Prediction"] = results["Prediction"].map({
    0: "Legitimate",
    1: "Potential Fraud",
})

print("MODEL VERIFICATION SUCCESSFUL")
print("Model file:", MODEL_PATH)
print("Threshold:", threshold)
print("\nSample predictions:")
print(results.to_string(index=False))
