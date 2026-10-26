import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 1. Load Kaggle Titanic dataset
df = pd.read_csv("evals/kaggle_titanic.csv")

# 2. DROP HIGH-CARDINALITY IDENTIFIERS & UNSTRUCTURED COLUMNS
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# 3. Separate features and target
X = df.drop(columns=["Survived"])
y = df["Survived"]

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

# 4. STRATIFIED SPLIT FIRST (No data snooping into test fold)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. DEFINE LEAK-FREE COLUMN TRANSFORMER
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

# 6. ENCAPSULATE PREPROCESSING & BALANCED ESTIMATOR IN PIPELINE
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 7. FIT ONLY ON TRAINING SET
pipeline.fit(X_train, y_train)

# 8. MULTI-METRIC EVALUATION
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("=" * 45)
print("Remediated Kaggle Titanic Pipeline Evaluation")
print("=" * 45)
print(f"ROC-AUC Score:          {roc_auc:.4f}")
print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
print(f"Macro F1-Score:         {macro_f1:.4f}")
