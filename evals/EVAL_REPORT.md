# Dataset Hygiene Skill: Verification & Benchmark Report

**Skill Version:** `1.1.0`  
**Specification:** LLM SkillHub (`llmskillhub.com`) & AgentSkills (`agentskills.io`)  
**Target:** Tabular Machine Learning Pipelines (`Scikit-Learn`, `XGBoost`, `LightGBM`)  
**Evaluation Date:** October 2026  

---

## 1. Executive Summary

AI coding assistants frequently generate machine learning code that achieves deceptively high validation accuracy while suffering from fatal architectural flaws:
1. **Pre-split data snooping**: Calling `fit_transform()` on full datasets before `train_test_split()`.
2. **Identifier memorization**: Fitting tree or linear estimators on high-cardinality primary keys (`transaction_id`).
3. **Post-event proxy target leakage**: Including features collected after the event occurs (`chargeback_approved_flag`).
4. **Unstratified imbalanced sampling**: Shuffling highly imbalanced datasets (e.g., 96:4) without `stratify=y`, causing complete omission of minority instances in training folds.
5. **Raw accuracy traps**: Measuring performance on imbalanced targets with `accuracy_score` instead of PR-AUC, ROC-AUC, and Macro-F1.

The **`dataset-hygiene`** skill prevents and fixes these issues using a deterministic, zero-token AST scanner and pre-flight dataset auditor. All 3 benchmark test cases passed with 100% compliance.

---

## 2. Benchmark Evaluation Test Cases

### TC-001: Imbalanced Fraud Leakage Audit (`evals/synthetic_fraud_data.csv`)
- **Trigger Scenario:** User requests: *"Train a fraud detection classifier on synthetic_fraud_data.csv with is_fraud as the target."*
- **Dataset Properties:**
  - Rows: 1,000
  - Columns: 7
  - Missing Values: 50 cells in `income` (5.0%)
  - Target: `is_fraud` (960 legit / 40 fraud = 96% : 4% severe imbalance)
  - Identifier: `transaction_id` (100% unique strings)
  - Temporal: `timestamp` (sequential 15-minute intervals)
  - Leaky Proxy: `chargeback_approved_flag` (Pearson correlation = 0.9872 with `is_fraud`)
- **Audit Findings:**
  - `HIGH_CARDINALITY_ID_COLUMN`: Flagged `transaction_id` (1000 unique values).
  - `TEMPORAL_COLUMN_DETECTED`: Flagged `timestamp` column.
  - `SEVERE_CLASS_IMBALANCE`: Minority class ratio is 0.04 (< 0.15 threshold).
  - `SUSPECTED_TARGET_PROXY_LEAKAGE`: `chargeback_approved_flag` flagged with correlation 0.9872.
  - `MISSING_VALUE_COLUMNS`: `income` flagged with 50 missing values.
- **Verdict:** **PASSED**

---

### TC-002: Pre-Split Scaling Refactor (`evals/bad_baseline_train.py` vs `evals/fixed_pipeline_train.py`)
- **Trigger Scenario:** User requests: *"Audit and fix evals/bad_baseline_train.py to ensure zero data leakage in production."*
- **AST Code Analysis Comparison:**

| Diagnostic Metric | Without Skill (`bad_baseline_train.py`) | With Skill (`fixed_pipeline_train.py`) |
| :--- | :--- | :--- |
| **Pre-Split Data Leakage** | `CRITICAL`: `fit_transform()` on line 23 before split on line 26 | **NONE**: Split on line 25, no pre-split transform |
| **Standalone Transformer** | `CRITICAL`: `StandardScaler` fitted outside pipeline | **NONE**: Clean `ColumnTransformer` + `Pipeline` |
| **Stratification Flag** | `WARNING`: Missing `stratify=` in `train_test_split` | **VERIFIED**: `stratify=y` explicitly provided |
| **Metric Trap Risk** | `WARNING`: Only `accuracy_score` imported/used | **VERIFIED**: `roc_auc_score`, `pr_auc`, `f1_score` |
| **Verification Passed** | **`false`** | **`true`** |
| **Code Leakage Issues** | `["PRE_SPLIT_DATA_LEAKAGE", "UNSAFE_STANDALONE_TRANSFORMER"]` | `[]` (0 issues) |

- **Empirical Model Evaluation Comparison:**

| Metric | Without Skill (Flawed Baseline) | With Skill (Remediated Pipeline) | Real-World Impact |
| :--- | :--- | :--- | :--- |
| **Reported Metric** | `Accuracy = 0.8000` | `PR-AUC = 0.5577`<br>`ROC-AUC = 0.9720`<br>`Macro F1 = 0.7235` | Baseline had 0% recall on fraud because unstratified split put all 40 fraud cases into test fold |
| **Data Snooping** | Test set statistics leaked into scaler | Completely isolated test set | True generalization in production |
| **Feature Set** | Contaminated with ID & leaky proxy | Clean features (`amount`, `income`, `age`) | Avoids catastrophic production memorization |
- **Verdict:** **PASSED**

---

### TC-003: Negative Trigger Non-ML Workflow Rejection
- **Trigger Scenario:** User requests: *"Create a responsive React navigation bar with dark mode toggle using Tailwind CSS."*
- **Skill Decision:** Evaluate activation criteria in `SKILL.md` (When NOT to Use This Skill). Non-ML front-end task rejected.
- **Context Overhead Saved:** 100% token overhead saved; no unnecessary AST or CSV scanner execution.
- **Verdict:** **PASSED**

---

## 3. Hygiene Audit Summary Table

| Audit Check | Status (Before) | Remediation Applied | Verified Status (After) |
| :--- | :--- | :--- | :--- |
| **Pre-Split Transformer Leakage** | `CRITICAL`: `StandardScaler().fit_transform()` executed before train-test split | Moved all transformations into `ColumnTransformer` inside `sklearn.pipeline.Pipeline` | **PASSED** (0 leakage issues) |
| **Identifier Memorization** | `HIGH_CARDINALITY_ID_COLUMN`: `transaction_id` retained in features | Explicitly pruned `transaction_id` before model fitting | **PASSED** (Pruned) |
| **Post-Event Proxy Target Leakage** | `SUSPECTED_TARGET_PROXY_LEAKAGE`: `chargeback_approved_flag` corr = 0.9872 | Dropped target proxy from feature matrix | **PASSED** (Pruned) |
| **Class Imbalance Stratification** | `MISSING_STRATIFICATION`: `train_test_split` unstratified on 96:4 data | Added `stratify=y` to preserve 96:4 ratio across folds | **PASSED** (Stratified) |
| **Imbalanced Accuracy Metric Trap** | `ACCURACY_TRAP_RISK`: Only `accuracy_score` reported | Replaced with `roc_auc_score`, `average_precision_score` (PR-AUC), and `f1_score` | **PASSED** (Multi-metric verified) |
| **Automated AST Self-Verification** | `verification_passed: false` | Verified via `audit_dataset.py --script` | **PASSED** (`verification_passed: true`) |

---

## 4. Package Artifacts Verification

- `dataset-hygiene/SKILL.md`: Valid YAML frontmatter, compliant with AgentSkills specification.
- `dataset-hygiene/scripts/audit_dataset.py`: Fully deterministic AST and dataset auditor.
- `dataset-hygiene/references/leak_free_pipelines.md`: Under 150 lines, contains copy-pasteable zero-leakage templates.
- Clean isolation: No `__pycache__`, no test logs, no eval data inside `./dataset-hygiene/`.
- Archive packaging: `dataset-hygiene.zip` with directory root prefix.
