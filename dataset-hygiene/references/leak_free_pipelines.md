# Leak-Free Machine Learning Pipelines Reference Guide

This reference provides production-ready, zero-leakage Scikit-learn patterns.

---

## 1. Standard Leak-Free Tabular Classification Pipeline

Encapsulates preprocessing and estimator in a single `Pipeline` to prevent train-test contamination. Preprocessing statistics (`mean`, `median`, `categories`) are fitted **only** on the training fold during cross-validation or split.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load and prune identifier / proxy leakage columns
df = pd.read_csv("dataset.csv")
drop_cols = ["id", "user_id", "transaction_id", "target_proxy"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# 2. Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# 3. Identify feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# 4. Stratified Split FIRST (Prevent data snooping)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Define Transformers (Never fit on full dataset!)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 6. Assemble leak-free Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 7. Fit ONLY on training data; evaluate on test
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

---

## 2. Imbalanced Dataset Evaluation Block

Avoid the raw accuracy trap on skewed datasets. Use PR-AUC, ROC-AUC, and Macro-F1.

```python
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score
)

def evaluate_imbalanced_model(y_true, y_pred, y_prob):
    """Evaluates classification performance avoiding accuracy traps."""
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    print("=" * 45)
    print("Zero-Leakage Imbalanced Evaluation Metrics")
    print("=" * 45)
    print(f"ROC-AUC Score:                 {roc_auc:.4f}")
    print(f"PR-AUC (Avg Precision):        {pr_auc:.4f}")
    print(f"Macro F1-Score:                {macro_f1:.4f}")
    print("-" * 45)
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "macro_f1": macro_f1}
```

---

## 3. Chronological / Temporal Split Pattern

When data contains time, timestamp, or sequential records, random shuffling leaks future events into past training data. Use chronological cutoff or `TimeSeriesSplit`.

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Sort chronologically
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Option A: Chronological Cutoff (Train on past, Test on future)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train, y_train = train_df.drop(columns=["target", "timestamp"]), train_df["target"]
X_test, y_test = test_df.drop(columns=["target", "timestamp"]), test_df["target"]

# Option B: Rolling TimeSeriesSplit for Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # Fit pipeline on (X_tr, y_tr), evaluate on (X_val, y_val)
```
