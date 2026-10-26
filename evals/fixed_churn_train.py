import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 1. Load data
df = pd.read_csv("evals/telco_customer_churn.csv")

# 2. DROP IDENTIFIER COLUMN (Prevent memorization)
df = df.drop(columns=["customerID"], errors="ignore")

# 3. Clean dirty numeric column TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 4. Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in X.columns if col not in numeric_features]

# 5. STRATIFIED SPLIT FIRST (No data leakage into test fold)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. DEFINE LEAK-FREE COLUMN TRANSFORMER
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

# 7. ENCAPSULATE PREPROCESSING & BALANCED ESTIMATOR IN PIPELINE
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 8. FIT ONLY ON TRAINING SET
pipeline.fit(X_train, y_train)

# 9. ROBUST EVALUATION
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("=" * 45)
print("Remediated Telco Churn Pipeline Evaluation")
print("=" * 45)
print(f"ROC-AUC Score:          {roc_auc:.4f}")
print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
print(f"Macro F1-Score:         {macro_f1:.4f}")
