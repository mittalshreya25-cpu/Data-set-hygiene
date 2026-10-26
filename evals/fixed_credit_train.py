import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 1. Load German Credit Risk dataset
df = pd.read_csv("evals/german_credit.csv")

# 2. Separate features and target (1 = bad credit risk, 0 = good)
X = df.drop(columns=["class"])
y = df["class"].map({"good": 0, "bad": 1})

numeric_features = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents"
]
categorical_features = [c for c in X.columns if c not in numeric_features]

# 3. STRATIFIED SPLIT FIRST (Zero data leakage into test fold)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. DEFINE LEAK-FREE COLUMN TRANSFORMER
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

# 5. ENCAPSULATE PREPROCESSING & BALANCED ESTIMATOR IN PIPELINE
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 6. FIT ONLY ON TRAINING SET
pipeline.fit(X_train, y_train)

# 7. ROBUST EVALUATION FOR CREDIT RISK
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("=" * 45)
print("Remediated German Credit Pipeline Evaluation")
print("=" * 45)
print(f"ROC-AUC Score:          {roc_auc:.4f}")
print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
print(f"Macro F1-Score:         {macro_f1:.4f}")
