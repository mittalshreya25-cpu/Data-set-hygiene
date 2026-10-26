import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 1. Load data
df = pd.read_csv("evals/synthetic_fraud_data.csv")

# 2. DROP IDENTIFIER, TEMPORAL, AND LEAKY TARGET PROXY COLUMNS
drop_cols = ["transaction_id", "timestamp", "chargeback_approved_flag"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# 3. Separate features and target
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

numeric_features = ["amount", "income", "age"]

# 4. STRATIFIED SPLIT FIRST (No data leakage into test fold)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. DEFINE LEAK-FREE COLUMN TRANSFORMER
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features)
])

# 6. ENCAPSULATE PREPROCESSING & BALANCED ESTIMATOR IN PIPELINE
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 7. FIT ONLY ON TRAINING SET
pipeline.fit(X_train, y_train)

# 8. ROBUST IMBALANCED EVALUATION (Avoid raw accuracy trap)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("=" * 45)
print("Remediated Leak-Free Model Evaluation")
print("=" * 45)
print(f"ROC-AUC Score:          {roc_auc:.4f}")
print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
print(f"Macro F1-Score:         {macro_f1:.4f}")
