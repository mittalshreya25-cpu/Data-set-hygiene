import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Telco Churn dataset
df = pd.read_csv("evals/telco_customer_churn.csv")

# Flaw 1: Retains customerID (unique identifier)
# Naively convert TotalCharges to numeric with fillna(0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Select numeric features but keep customer ID encoded
df["cust_id_num"] = df["customerID"].astype("category").cat.codes
feature_cols = ["cust_id_num", "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
X = df[feature_cols]
y = df["Churn"].map({"No": 0, "Yes": 1})

# Flaw 2: Calling fit_transform BEFORE train_test_split (Pre-split data leakage!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Flaw 3: train_test_split without stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Flaw 4: Accuracy trap - reporting raw accuracy on customer churn
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Telco Churn Flawed Model Accuracy: {acc:.4f}")
