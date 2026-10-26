import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load German Credit Risk dataset
df = pd.read_csv("evals/german_credit.csv")

# Select numeric features
numeric_cols = ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"]
X = df[numeric_cols]
# Target: 1 = bad credit risk (default), 0 = good
y = df["class"].map({"good": 0, "bad": 1})

# FLAW 1: Pre-split transformation on the whole dataset
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# FLAW 2: train_test_split without stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# Train estimator
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# FLAW 3: Evaluates only with raw accuracy on credit risk
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"German Credit Flawed Baseline Accuracy: {acc:.4f}")
