#!/usr/bin/env python3
"""
Benchmark Generator for dataset-hygiene evaluation suite.
Creates:
1. evals/synthetic_fraud_data.csv
2. evals/bad_baseline_train.py
3. evals/fixed_pipeline_train.py
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_data(output_path: str):
    np.random.seed(42)
    n_rows = 1000

    # 96% class 0, 4% class 1
    n_fraud = 40
    n_legit = n_rows - n_fraud
    fraud_labels = np.zeros(n_rows, dtype=int)
    fraud_indices = np.random.choice(n_rows, size=n_fraud, replace=False)
    fraud_labels[fraud_indices] = 1

    # 1. Unique ID column
    transaction_ids = [f"TXN_{i:06d}" for i in range(1, n_rows + 1)]

    # 2. Timestamp column
    base_time = pd.Timestamp("2026-01-01 08:00:00")
    timestamps = [base_time + pd.Timedelta(minutes=i * 15) for i in range(n_rows)]

    # 3. Numeric features
    amounts = np.random.exponential(scale=100.0, size=n_rows) + 5.0
    amounts[fraud_indices] += np.random.normal(300.0, 50.0, size=n_fraud)
    amounts = np.round(np.clip(amounts, 1.0, 5000.0), 2)

    income = np.random.normal(loc=65000, scale=20000, size=n_rows)
    income = np.round(np.clip(income, 15000, 250000), 2)
    # Inject missing values in income (50 rows = 5%)
    missing_indices = np.random.choice(n_rows, size=50, replace=False)
    income[missing_indices] = np.nan

    age = np.random.randint(18, 75, size=n_rows)

    # 4. Leaky post-event target proxy (correlation ~0.98 with is_fraud)
    # Chargeback flag is identical to fraud with only ~1% flipped noise
    chargeback_flag = fraud_labels.copy().astype(float)
    flip_indices = np.random.choice(n_rows, size=1, replace=False)  # flip 1 value
    for idx in flip_indices:
        chargeback_flag[idx] = 1.0 - chargeback_flag[idx]

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "timestamp": [str(t) for t in timestamps],
        "amount": amounts,
        "income": income,
        "age": age,
        "chargeback_approved_flag": chargeback_flag,
        "is_fraud": fraud_labels
    })

    corr = df["chargeback_approved_flag"].corr(df["is_fraud"])
    print(f"Generated {len(df)} rows. Chargeback correlation with is_fraud: {corr:.4f}")

    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")


def generate_bad_baseline_script(output_path: str):
    code = '''import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("evals/synthetic_fraud_data.csv")

# Leaky practice: Retaining ID and leaky target proxy
# Fill missing income naively
df["income"] = df["income"].fillna(df["income"].mean())

# Prepare features (keeping leaky chargeback_approved_flag and converting ID)
df["txn_num"] = df["transaction_id"].apply(lambda x: int(x.split("_")[1]))
feature_cols = ["txn_num", "amount", "income", "age", "chargeback_approved_flag"]
X = df[feature_cols]
y = df["is_fraud"]

# CRITICAL LEAK: Calling fit_transform BEFORE train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SNOOPING & UNBALANCED: train_test_split without stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ACCURACY TRAP: Evaluating only with accuracy_score on 96:4 imbalanced data
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Suspiciously Perfect Test Accuracy: {acc:.4f}")
'''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Saved bad baseline script to {output_path}")


def generate_fixed_pipeline_script(output_path: str):
    code = '''import pandas as pd
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
'''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Saved fixed pipeline script to {output_path}")


if __name__ == "__main__":
    os.makedirs("evals", exist_ok=True)
    generate_synthetic_data("evals/synthetic_fraud_data.csv")
    generate_bad_baseline_script("evals/bad_baseline_train.py")
    generate_fixed_pipeline_script("evals/fixed_pipeline_train.py")
