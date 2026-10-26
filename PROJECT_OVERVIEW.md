# Dataset Hygiene: Complete Project Guide & Testing Overview

Welcome! This document provides a complete, beginner-friendly walkthrough of the **`dataset-hygiene`** project, explaining what problem it solves, how it works under the hood, and every single test we ran.

---

## 1. What is `dataset-hygiene`?

When AI coding assistants (like ChatGPT, Claude, Cursor, or Antigravity) write Machine Learning code, they frequently make **silent statistical mistakes**:
- The code runs without crashing.
- It prints a high accuracy number (e.g. 95%+).
- But in real-world production, the model completely fails!

**`dataset-hygiene`** is an AI Agent Skill built to prevent and fix these silent mistakes automatically. It acts as an automated "pre-flight safety inspector" for tabular data and Python ML training code.

---

## 2. The 4 Silent Mistakes It Catches

```
┌─────────────────────────────────┐
│     Raw Tabular CSV Dataset     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Identifier Memorization (e.g. `transaction_id`, `Name`)  │
│ 2. Post-Event Cheat Columns (e.g. `chargeback_flag` > 0.95) │
│ 3. Pre-Split Data Snooping (`fit_transform` before split)   │
│ 4. Imbalanced Accuracy Trap (96:4 split evaluated by acc)   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Automated AST & Stat Auditor   │ ──► Blocks code if leaks exist
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Zero-Leakage Pipeline Enforced│ ──► Verified Safe for Production
└─────────────────────────────────┘
```

1. **Pre-Split Data Leakage (Data Snooping):**
   - *The Mistake:* Calling `scaler.fit_transform(X)` on the entire dataset *before* `train_test_split()`.
   - *Why it's bad:* The scaler calculates the mean and standard deviation of your **future test data**. The test data has now contaminated your training fold.
2. **Identifier Memorization:**
   - *The Mistake:* Keeping columns like `customerID`, `transaction_id`, or `PassengerId`.
   - *Why it's bad:* These columns are 100% unique primary keys. Decision trees memorize individual rows instead of learning general patterns.
3. **Post-Event Proxy Target Leakage:**
   - *The Mistake:* Including features collected *after* the outcome occurs (e.g., `chargeback_approved_flag` predicting fraud).
   - *Why it's bad:* It has a ~0.98 correlation with the target. In real life, chargebacks only happen weeks after a transaction.
4. **The Imbalanced Accuracy Trap:**
   - *The Mistake:* Using raw `accuracy_score` on heavily skewed data (e.g., 96% legit vs 4% fraud) without stratified splitting.
   - *Why it's bad:* A useless model that predicts "No Fraud" for everyone gets **96% accuracy** while catching **0% of fraudsters**!

---

## 3. How the Skill Works (The 3-Phase Workflow)

### Phase 1: Zero-Token Pre-Flight Audit
Instead of dumping thousands of raw CSV rows into the AI's prompt (wasting context tokens), the agent runs:
```bash
python dataset-hygiene/scripts/audit_dataset.py --data <dataset.csv> --target <target_col>
```
It extracts key data hygiene metrics:
- 100% unique identifier columns
- Constant zero-variance columns
- Temporal / date columns (warning against random splits)
- Severe class imbalance (<15% minority ratio)
- Missing values and dirty whitespace blanks
- High-correlation proxy leakage (>0.95 Pearson correlation)

### Phase 2: Abstract Syntax Tree (AST) Code Inspection
When an ML Python script is provided, the script parses the code's **Abstract Syntax Tree (AST)**:
- Checks the exact line numbers of `train_test_split` vs `.fit_transform()` and `.fit_resample()`.
- Flags `PRE_SPLIT_DATA_LEAKAGE` if transformations occur before splitting.
- Flags `UNSAFE_STANDALONE_TRANSFORMER` if scalers/imputers are used outside of `Pipeline` or `ColumnTransformer`.
- Flags `MISSING_STRATIFICATION` if `stratify=y` is missing.
- Flags `ACCURACY_TRAP_RISK` if only `accuracy_score` is used without PR-AUC or ROC-AUC.

### Phase 3: Closed-Loop Self-Verification
The agent writes the remediated code and re-runs:
```bash
python dataset-hygiene/scripts/audit_dataset.py --script <fixed_script.py>
```
The agent is forbidden from finishing the task until:
`"verification_passed": true` and `"code_leakage_issues": []`.

---

## 4. The 4 Datasets & Live Tests We Ran

We validated this skill across 4 distinct domains:

### Test 1: Synthetic Fraud Benchmark (`evals/synthetic_fraud_data.csv`)
- **Characteristics:** 1,000 rows, 96% legit vs 4% fraud (severe imbalance), unique `transaction_id`, sequential `timestamp`, missing values in `income`, and leaky proxy `chargeback_approved_flag` (0.9872 correlation).
- **Flawed Baseline (`bad_baseline_train.py`):** Scaled data before splitting, no stratification, reported only accuracy. Result: All 40 fraud cases randomly fell into the test set; model trained with 0 fraud cases and got 80% accuracy with 0% recall.
- **Auditor Result:** Caught all 4 flaws; reported `verification_passed: false`.
- **Remediated Pipeline (`fixed_pipeline_train.py`):** Dropped ID and proxy, stratified 96:4 across folds, wrapped in `ColumnTransformer` + `Pipeline`.
- **Verified Result:** **ROC-AUC: 0.9720**, **PR-AUC: 0.5577**, **Verification: PASSED (`true`)**.

---

### Test 2: IBM Telco Customer Churn (`evals/telco_customer_churn.csv`)
- **Characteristics:** Real-world enterprise dataset with 7,043 customers, 21 columns, predicting customer churn (`Yes` / `No`).
- **Flaws Caught:**
  - `customerID`: 7,043 unique alphanumeric strings (100% unique primary key).
  - Hidden Dirty Blanks: 11 rows in `TotalCharges` contained whitespace `" "` instead of numbers, causing standard pandas `isna()` to report 0 missing values while crashing standard ML scalers!
- **Auditor Result:** Flagged `customerID` as high-cardinality ID and flagged `TotalCharges` as a dirty numeric column needing imputation.
- **Remediated Pipeline (`fixed_churn_train.py`):** Dropped `customerID`, cleaned `TotalCharges`, applied `OneHotEncoder` on categories and `StandardScaler` on numerics inside a `Pipeline`.
- **Verified Result:** **ROC-AUC: 0.8206**, **PR-AUC: 0.6039**, **Verification: PASSED (`true`)**.

---

### Test 3: German Credit Risk (`evals/german_credit.csv`)
- **Characteristics:** Real-world financial risk dataset with 1,000 loan applicants, 21 financial attributes, predicting `good` vs `bad` credit risk.
- **Flaws Caught:**
  - `MinMaxScaler().fit_transform()` executed before `train_test_split()`.
  - Unstratified splitting of financial defaults.
  - Relying solely on raw accuracy on credit evaluation.
- **Remediated Pipeline (`fixed_credit_train.py`):** Stratified split first, encapsulated numeric and categorical transformers into `ColumnTransformer` inside `Pipeline`.
- **Verified Result:** **ROC-AUC: 0.7978**, **PR-AUC: 0.5983**, **Verification: PASSED (`true`)**.

---

### Test 4: Kaggle Titanic Competition (`evals/kaggle_titanic.csv`)
- **Characteristics:** Iconic Kaggle dataset with 891 passenger records, 12 features, predicting `Survived`.
- **Flaws Caught:**
  - `PassengerId` and `Name` (both 891 unique values, 100% primary keys).
  - 866 missing cells across `Age` (19.9%), `Cabin` (77.1%), and `Embarked` (0.22%).
  - Pre-split scaling on `Age` and `Fare`.
- **Remediated Pipeline (`fixed_titanic_train.py`):** Dropped identifiers, imputed missing values fold-by-fold inside `ColumnTransformer`, stratified survival classes.
- **Verified Result:** **ROC-AUC: 0.8287**, **PR-AUC: 0.7715**, **Verification: PASSED (`true`)**.

---

## 5. What We Improved During Testing

During live testing across these datasets, we improved the skill scanner:
1. **Pandas 3.0 Compatibility:** Python 3.12 with Pandas 3.0 uses `StringDtype` (`str`), which failed legacy `df[col].dtype == object` checks. We upgraded the scanner to use `pd.api.types.is_string_dtype` to ensure universal compatibility.
2. **Whitespace "Dirty Blanks" Detection:** Standard `df.isna()` ignores spaces like `" "`. We added whitespace-aware missing cell detection to catch real-world data traps like Telco `TotalCharges`.
3. **Dirty Numeric Column Detection:** Added automatic detection for string columns where >90% of values are numbers with a few invalid blanks, recommending `pd.to_numeric(errors='coerce')`.
4. **Binary String Target Mapping:** Enhanced proxy correlation calculation to automatically encode string targets (like `"Yes"`/`"No"`, `"good"`/`"bad"`) to `0`/`1` so target proxy leakage is detected across all categorical targets.

---

## 6. Project Directory Summary

- [`dataset-hygiene/SKILL.md`](dataset-hygiene/SKILL.md): Spec-compliant skill definition for LLM SkillHub and AgentSkills.
- [`dataset-hygiene/scripts/audit_dataset.py`](dataset-hygiene/scripts/audit_dataset.py): The core deterministic pre-flight scanner and AST code auditor.
- [`dataset-hygiene/references/leak_free_pipelines.md`](dataset-hygiene/references/leak_free_pipelines.md): Copy-pasteable zero-leakage Scikit-learn templates.
- [`dataset-hygiene.zip`](dataset-hygiene.zip): The packaged distribution file ready for LLM SkillHub upload.
- [`evals/`](evals/): All benchmark scripts, synthetic datasets, and real-world test cases.
