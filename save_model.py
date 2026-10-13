
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Create the models folder
Path("models").mkdir(parents=True, exist_ok=True)

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

# Keep the same 60/20/20 split used in threshold tuning
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

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42,
    )),
])

# Train on training data only
pipeline.fit(X_train, y_train)

# Choose threshold from validation data
val_probabilities = pipeline.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.10, 0.81, 0.05)

threshold_results = []

from sklearn.metrics import f1_score

for threshold in thresholds:
    predictions = (val_probabilities >= threshold).astype(int)
    threshold_results.append(
        (float(threshold), f1_score(y_val, predictions, zero_division=0))
    )

best_threshold, best_validation_f1 = max(
    threshold_results, key=lambda item: item[1]
)

# Save fitted pipeline and selected threshold
artifact = {
    "model": pipeline,
    "threshold": best_threshold,
    "features": X.columns.tolist(),
    "excluded_feature": "Amount_Zscore_User",
    "validation_f1": best_validation_f1,
}

model_path = Path("models/fraud_detection_model.joblib")
joblib.dump(artifact, model_path)

print("MODEL SAVED SUCCESSFULLY")
print("Model path:", model_path)
print("Selected threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_validation_f1, 4))
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows reserved:", len(X_test))
print("\nThe final test set was not used to select the threshold.")
