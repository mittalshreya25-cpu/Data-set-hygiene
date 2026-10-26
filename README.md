# Dataset Hygiene: Zero-Leakage ML Pre-Flight Auditor & Pipeline Enforcer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Spec: LLM SkillHub](https://img.shields.io/badge/Spec-LLM%20SkillHub-purple.svg)](https://llmskillhub.com)
[![Spec: AgentSkills](https://img.shields.io/badge/Spec-AgentSkills.io-green.svg)](https://agentskills.io)

> A production-grade AI Agent Skill that audits tabular machine learning datasets (CSV/Parquet) and Python training scripts (Scikit-Learn, XGBoost, LightGBM) to detect and fix data leakage, pre-split scaling or oversampling contamination, identifier memorization, temporal leakage, and imbalanced accuracy traps.

---

## 1. Overview & Problem Statement

Standard AI coding assistants frequently introduce silent statistical flaws when generating machine learning code:
1. **Pre-Split Data Leakage:** Calling `.fit_transform()` (e.g., `StandardScaler`, `SimpleImputer`, `OneHotEncoder`) or `.fit_resample()` (`SMOTE`) on the full dataset *before* calling `train_test_split()`, leaking test distribution parameters into training.
2. **Identifier Memorization:** Feeding 100%-unique primary keys (`customerID`, `transaction_id`, `PassengerId`) into tree estimators, causing models to memorize row identities instead of generalizable patterns.
3. **Post-Event Proxy Target Leakage:** Including features collected after the outcome occurs (>0.95 Pearson correlation with target).
4. **Imbalanced Accuracy Traps:** Reporting raw `accuracy_score` on skewed distributions without stratified splitting, producing models that show 96% accuracy while catching 0% of minority cases.

Because the code runs without syntax errors and prints high validation scores, these flaws slip silently into production where models fail catastrophically.

`dataset-hygiene` enforces a **deterministic, zero-leakage workflow** before and after writing any ML training code.

---

## 2. Repository Architecture

```text
├── dataset-hygiene/                  # Production Skill Package (LLM SkillHub compliant)
│   ├── SKILL.md                      # Complete agent skill definition & triage guardrails
│   ├── scripts/
│   │   └── audit_dataset.py          # Deterministic AST scanner & pandas dataset auditor
│   └── references/
│       └── leak_free_pipelines.md    # Leak-free Scikit-learn templates & recipes
│
├── evals/                            # Comprehensive Evaluation & Benchmark Suite
│   ├── synthetic_fraud_data.csv      # Synthetic benchmark dataset (96:4 imbalance, proxy cheat)
│   ├── bad_baseline_train.py         # Flawed "Without-Skill" baseline script
│   ├── fixed_pipeline_train.py       # Remediated "With-Skill" leak-free pipeline
│   ├── telco_customer_churn.csv      # IBM Telco Customer Churn dataset (7,043 rows)
│   ├── bad_churn_train.py            # Flawed churn script (customerID retained, pre-split scaling)
│   ├── fixed_churn_train.py          # Remediated churn pipeline (OneHotEncoder + StandardScaler)
│   ├── german_credit.csv             # German Credit Risk financial dataset (1,000 rows)
│   ├── bad_credit_train.py           # Flawed credit script (pre-split MinMaxScaler)
│   ├── fixed_credit_train.py         # Remediated credit pipeline (stratified + ColumnTransformer)
│   ├── kaggle_titanic.csv            # Kaggle Titanic competition dataset (891 rows)
│   ├── bad_titanic_train.py          # Flawed Titanic script (PassengerId retained, pre-split scaling)
│   ├── fixed_titanic_train.py        # Remediated Titanic pipeline (stratified + Pipeline)
│   ├── eval_cases.json               # Formal benchmark test cases schema
│   └── EVAL_REPORT.md                # Full evaluation & benchmark report
│
├── eval-package/                     # OpenEval Hub package
│   ├── manifest.yaml                 # Eval manifest matching evalpkg schema
│   └── dataset.jsonl                 # Benchmark samples
│
├── dataset-hygiene.zip               # Upload-ready skill package for LLM SkillHub
├── dataset-hygiene-eval.zip          # Upload-ready eval package for LLM SkillHub
├── PROJECT_OVERVIEW.md               # In-depth project overview & test guide
└── README.md                         # Repository documentation
```

---

## 3. Quickstart & CLI Usage

### Prerequisites
- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`

```bash
pip install pandas numpy scikit-learn
```

### Running the Pre-Flight Scanner

Audit both a dataset and an existing Python training script:
```bash
python dataset-hygiene/scripts/audit_dataset.py --data <path_to_csv> --target <target_column> --script <path_to_script.py>
```

#### Example Output:
```json
{
  "dataset_audit": {
    "rows": 1000,
    "columns": 7,
    "missing_cells_total": 50,
    "flags": {
      "HIGH_CARDINALITY_ID_COLUMN": [{"column": "transaction_id", "reason": "100% unique values"}],
      "TEMPORAL_COLUMN_DETECTED": ["timestamp"],
      "SEVERE_CLASS_IMBALANCE": {"minority_class_ratio": 0.04, "distribution": {"0": 0.96, "1": 0.04}},
      "SUSPECTED_TARGET_PROXY_LEAKAGE": [{"column": "chargeback_approved_flag", "correlation": 0.9872}]
    }
  },
  "script_audit": {
    "critical_issues": [
      {
        "type": "PRE_SPLIT_DATA_LEAKAGE",
        "line": 23,
        "description": ".fit_transform() called on line 23 before train_test_split() on line 26."
      },
      {
        "type": "UNSAFE_STANDALONE_TRANSFORMER",
        "line": 23,
        "description": "Standalone transformer 'scaler' called .fit_transform() outside Pipeline."
      }
    ],
    "warnings": [
      {
        "type": "MISSING_STRATIFICATION",
        "line": 26,
        "description": "train_test_split() called without 'stratify=' parameter."
      },
      {
        "type": "ACCURACY_TRAP_RISK",
        "description": "accuracy_score is used without robust imbalanced metrics (roc_auc_score, average_precision_score, f1_score)."
      }
    ],
    "verification_passed": false
  }
}
```

### Self-Verification of Remediated Scripts
```bash
python dataset-hygiene/scripts/audit_dataset.py --script evals/fixed_pipeline_train.py
```
Outputs `"verification_passed": true` and `code_leakage_issues: []`.

---

## 4. Multi-Domain Validation Benchmarks

The skill has been empirically validated across 4 diverse datasets:

| Benchmark Dataset | Domain | Flaws Caught by Auditor | Remediated Metrics | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Synthetic Fraud** | Financial / Fraud | 96:4 severe imbalance, 100% unique ID, 0.98 proxy target cheat | **ROC-AUC: 0.9720**<br>**PR-AUC: 0.5577** | **PASSED** (`true`) |
| **IBM Telco Churn** | Enterprise Telecom | `customerID` primary key, hidden whitespace blanks in `TotalCharges` | **ROC-AUC: 0.8206**<br>**PR-AUC: 0.6039** | **PASSED** (`true`) |
| **German Credit** | Banking / Credit Risk | Pre-split `MinMaxScaler` data snooping, unstratified credit split | **ROC-AUC: 0.7978**<br>**PR-AUC: 0.5983** | **PASSED** (`true`) |
| **Kaggle Titanic** | Competitive ML | `PassengerId` & `Name` memorization, 866 missing cells, pre-split scaling | **ROC-AUC: 0.8287**<br>**PR-AUC: 0.7715** | **PASSED** (`true`) |

---

## 5. Live Testing Improvements

During live multi-domain testing, three critical enhancements were implemented into `audit_dataset.py`:
1. **Pandas 3.0 Compatibility:** Replaced legacy `object` type checks with `pd.api.types.is_string_dtype` to support Pandas 3.0's native `StringDtype`.
2. **Hidden Whitespace Trap:** Added detection for whitespace `" "` strings that standard `isna()` misses, preventing runtime scaler crashes on dirty columns like Telco `TotalCharges`.
3. **Binary String Target Mapping:** Enhanced correlation detection to automatically map string categories (`Yes`/`No`, `good`/`bad`) to `0`/`1` for proxy target leakage analysis.

---

## 6. Author & License

- **Author:** Shreya Mittal
- **Email:** [mittalshreya2517@gmail.com](mailto:mittalshreya2517@gmail.com)
- **GitHub Repository:** [https://github.com/mittalshreya25-cpu/Data-set-hygiene](https://github.com/mittalshreya25-cpu/Data-set-hygiene)
- **License:** [MIT License](LICENSE)
