import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ==========================================
# FRAUDGUARD AI - MODEL TRAINING
# ==========================================

print("\n========== FRAUDGUARD AI MODEL TRAINING ==========")

# ------------------------------------------
# 1. Load dataset
# ------------------------------------------

file_path = "data/financial_transactions.csv"
df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# ------------------------------------------
# 2. Remove identifier columns
# ------------------------------------------

df = df.drop(columns=["Transaction_ID", "User_ID"])


# ------------------------------------------
# 3. Convert Date into useful features
# ------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

df = df.drop(columns=["Date"])


# ------------------------------------------
# 4. Separate X and y
# ------------------------------------------

X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]


# ------------------------------------------
# 5. Identify feature types
# ------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------
# 6. Train/Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------
# 7. Preprocessing
# ------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ------------------------------------------
# 8. Create Logistic Regression pipeline
# ------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------
# 9. Train model
# ------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed successfully.")


# ------------------------------------------
# 10. Generate predictions
# ------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully.")

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Fraud Probabilities:")
print(y_probability[:10])


print("\n========== MODEL TRAINING COMPLETE ==========")
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("\n========== MODEL EVALUATION ==========")

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n========== MODEL EVALUATION COMPLETE ==========")