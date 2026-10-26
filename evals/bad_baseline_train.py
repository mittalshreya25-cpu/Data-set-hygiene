import pandas as pd
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
