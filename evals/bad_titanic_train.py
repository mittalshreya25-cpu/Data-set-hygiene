import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Kaggle Titanic dataset
df = pd.read_csv("evals/kaggle_titanic.csv")

# FLAW 1: Keeping PassengerId (unique ID) in features
# Naive global imputation before split
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

feature_cols = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"]
X = df[feature_cols]
y = df["Survived"]

# FLAW 2: Pre-split scaling on entire dataset (Data Snooping / Leakage)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# FLAW 3: train_test_split without stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# FLAW 4: Accuracy trap - reporting raw accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Kaggle Titanic Flawed Baseline Accuracy: {acc:.4f}")
