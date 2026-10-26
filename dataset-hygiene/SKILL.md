---
name: dataset-hygiene
description: Audits tabular machine learning datasets (CSV/Parquet) and Python training scripts (Scikit-learn, XGBoost, LightGBM) to detect and fix data leakage, pre-split scaling or oversampling contamination, identifier memorization, temporal leakage, and imbalanced accuracy traps. Activate when the user asks to train a model on a CSV, debug suspiciously high validation accuracy, clean a dataset, or review an ML pipeline.
license: MIT
compatibility: Works with Claude Code, Cursor, Gemini CLI, Google Antigravity, Codex, and OpenCode. Requires Python 3.8+, pandas, numpy, scikit-learn.
metadata:
  author: shreya-mittal
  version: "1.1.0"
  category: data-science-ml
---

# Dataset Hygiene: Zero-Leakage ML Pre-Flight Auditor & Pipeline Enforcer

## 1. Executive Summary & Problem Statement

### The Problem
When AI coding agents (Claude Code, Cursor, Antigravity) and developers build machine learning workflows on tabular data (`.csv`/`.parquet`), they frequently introduce silent statistical flaws:
1. **Pre-Split Data Leakage:** Calling `.fit_transform()` (e.g., `StandardScaler`, `SimpleImputer`, `OneHotEncoder`) or `.fit_resample()` (`SMOTE`) on the full dataset *before* `train_test_split()`, contaminating training data with test distribution parameters.
2. **Identifier Memorization:** Feeding 100%-unique primary keys (`customerID`, `transaction_id`, `PassengerId`) into tree estimators, which memorize row identities instead of learning generalizable patterns.
3. **Post-Event Proxy Target Leakage:** Including features collected after the outcome occurs (>0.95 correlation with target).
4. **Imbalanced Accuracy Traps:** Reporting raw `accuracy_score` on skewed distributions without stratified splitting, producing models that show 96% accuracy while catching 0% of minority cases.

Because the code executes without syntax errors and prints high validation scores, these flaws slip silently into production where models fail catastrophically.

### Who Has It
Data scientists, machine learning engineers, and autonomous AI coding agents building predictive models on tabular business, financial, healthcare, or e-commerce datasets.

### How This Skill Solves It
`dataset-hygiene` enforces a deterministic, zero-token pre-flight workflow:
1. **Deterministic Dataset Scanner (`scripts/audit_dataset.py`):** Audits CSVs without dumping raw rows into context tokens, catching 100%-unique IDs, temporal columns, class skew (<15%), and target proxies (>0.95 Pearson correlation).
2. **AST Code Scanner:** Uses Python's Abstract Syntax Tree (`ast`) to inspect training scripts, flagging pre-split `.fit_transform()` calls, standalone transformers outside of pipelines, unstratified splits, and missing PR-AUC/ROC-AUC metrics.
3. **Closed-Loop Verification:** Enforces refactoring into `ColumnTransformer` + `Pipeline` and blocks execution until `verification_passed: true` and 0 leakage issues remain.

### What Was Improved After Testing
Through multi-domain testing on synthetic fraud, IBM Telco Churn, German Credit Risk, and Kaggle Titanic datasets, three critical production enhancements were integrated:
1. **Hidden Whitespace Trap:** Discovered real-world datasets (e.g., Telco `TotalCharges`) use whitespace `" "` instead of NaNs, causing standard `isna()` to miss them while crashing downstream scalers. Added whitespace missing-value and dirty numeric column detection.
2. **Pandas 3.0 Compatibility:** Replaced legacy `object` checks with `pd.api.types.is_string_dtype` to support Pandas 3.0's native `StringDtype`.
3. **Categorical Target Correlation:** Added automatic binary mapping (`Yes`/`No`, `good`/`bad`) so proxy leakage detection works across both numeric and string classification targets.

---

## 2. When to Use This Skill
Activate this skill when the user asks to:
- Train, build, or optimize a classification or regression model on tabular data (`.csv`, `.parquet`).
- Audit or debug an existing Python ML script or Jupyter workflow.
- Investigate why a model has unrealistic validation accuracy (e.g., 99%+ accuracy that fails in production).
- Build feature engineering, imputation, scaling, or class-balancing pipelines.

## When NOT to Use This Skill
Do NOT activate this skill for:
- Frontend UI styling, web deployment configs, or non-ML CRUD API endpoints.
- Pure LLM prompt templates that do not involve tabular data or statistical model training.

---

## 3. Mandatory Execution Workflow

### Phase 1: Deterministic Zero-Token Pre-Flight Audit
**RULE:** Never print raw CSV files into context and never write model training code before running the bundled deterministic scanner.

Execute the bundled script from the skill root:
```bash
python scripts/audit_dataset.py --data <path_to_csv> --target <target_column_name> --script <optional_existing_py_script>
```
Parse the returned JSON output carefully. It contains two top-level keys: `dataset_audit` and `script_audit`.

### Phase 2: Triage & Apply Decision Guardrails
Consult `references/leak_free_pipelines.md` only for the specific patterns flagged in the JSON report:

1. **IF `HIGH_CARDINALITY_ID_COLUMN` or `SUSPECTED_IDENTIFIER_COLUMN` is flagged:**
   - Explicitly drop those columns (`df.drop(columns=[...])`) prior to splitting or modeling.
2. **IF `SUSPECTED_TARGET_PROXY_LEAKAGE` is flagged (abs correlation > 0.95):**
   - Warn the user immediately that the feature may be a post-event proxy or deterministic transformation of the target. Drop or isolate it for verification.
3. **IF `DIRTY_NUMERIC_COLUMNS_DETECTED` is flagged:**
   - Convert dirty string columns to numeric using `pd.to_numeric(df[col], errors="coerce")` and impute missing values.
4. **IF `TEMPORAL_COLUMN_DETECTED` is flagged:**
   - Warn against random `train_test_split` if forecasting future events; recommend `TimeSeriesSplit` or chronological cutoff splitting.
5. **IF `SEVERE_CLASS_IMBALANCE` is flagged (minority class ratio < 0.15):**
   - Enforce `train_test_split(..., stratify=y, random_state=42)`.
   - Forbid standalone `accuracy_score`. Require `roc_auc_score`, `average_precision_score` (PR-AUC), and `f1_score(average="macro")`.
   - Use `class_weight="balanced"` on the estimator or wrap `SMOTE` strictly inside `imblearn.pipeline.Pipeline` (never before splitting).
6. **IF `PRE_SPLIT_DATA_LEAKAGE` or `UNSAFE_STANDALONE_TRANSFORMER` is flagged in Python code:**
   - Refactor all preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) inside a `sklearn.compose.ColumnTransformer` wrapped in a `sklearn.pipeline.Pipeline`.

### Phase 3: Closed-Loop Self-Verification
After generating or refactoring the Python training script, you MUST verify your own code by running:
```bash
python scripts/audit_dataset.py --script <path_to_new_script.py>
```
Do not mark the task complete until `code_leakage_issues` in the JSON output is an empty list (`[]`) and `verification_passed` is `true`.

### Phase 4: Structured Output Summary
Always conclude your response with a concise **Hygiene Audit Summary Table**:
| Audit Check | Status (Before) | Remediation Applied | Verified Status (After) |
| :--- | :--- | :--- | :--- |

---

## 4. Multi-Domain Empirical Validation Results

The skill has been benchmarked and verified across 4 diverse datasets:

| Dataset | Domain | Flaws Detected by Auditor | Verified Remediated Result |
| :--- | :--- | :--- | :--- |
| **Synthetic Fraud** | Financial / Fraud | 96:4 severe imbalance, 100% unique ID, 0.98 proxy target leak | **PASSED** (PR-AUC: 0.5577, ROC-AUC: 0.9720) |
| **IBM Telco Churn** | Enterprise Telecom | `customerID` primary key, hidden whitespace blanks in `TotalCharges` | **PASSED** (PR-AUC: 0.6039, ROC-AUC: 0.8206) |
| **German Credit** | Banking / Loan Default | Pre-split `MinMaxScaler` data snooping, unstratified credit split | **PASSED** (PR-AUC: 0.5983, ROC-AUC: 0.7978) |
| **Kaggle Titanic** | Competitive ML | `PassengerId` & `Name` memorization, 866 missing cells, pre-split scaling | **PASSED** (PR-AUC: 0.7715, ROC-AUC: 0.8287) |
