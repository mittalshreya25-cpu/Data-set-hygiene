#!/usr/bin/env python3
"""
dataset-hygiene: Zero-Leakage Tabular ML Pre-Flight Auditor & Script Scanner.
Audits CSV datasets and Python ML training scripts for data leakage, identifier memorization,
temporal leakage, pre-split transformations, and imbalanced accuracy traps.
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def audit_csv(csv_path: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Audits a CSV dataset for hygiene and leakage risks:
    - High cardinality identifier columns
    - Constant zero-variance columns
    - Temporal / timestamp columns
    - Severe class imbalance
    - Suspected target proxy leakage (>0.95 Pearson correlation)
    - Missing value summary
    """
    if not os.path.exists(csv_path):
        return {"error": f"Dataset file not found: {csv_path}"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Failed to read CSV: {str(e)}"}

    total_rows = len(df)
    total_cols = len(df.columns)
    missing_cells = int(df.isna().sum().sum())

    id_regex = re.compile(r"(^id$|_id$|uuid|transaction_id|user_id)", re.IGNORECASE)
    temporal_regex = re.compile(r"(date|time|timestamp|created_at)", re.IGNORECASE)

    high_cardinality_id_cols = []
    zero_variance_cols = []
    temporal_cols = []
    missing_val_cols = []
    dirty_numeric_cols = []

    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)
        n_unique_dropped = df[col].nunique(dropna=True)

        # 1. Identifier / High Cardinality
        is_id_name = bool(id_regex.search(str(col)))
        is_100_pct_unique = (n_unique_dropped == total_rows and total_rows > 1)
        if is_id_name or is_100_pct_unique:
            high_cardinality_id_cols.append({
                "column": col,
                "unique_values": int(n_unique_dropped),
                "total_rows": total_rows,
                "reason": "100% unique values" if is_100_pct_unique else "Column name matches identifier pattern"
            })

        # 2. Constant / Zero-Variance
        if n_unique <= 1:
            zero_variance_cols.append({
                "column": col,
                "unique_values": int(n_unique)
            })

        # 3. Temporal columns
        if temporal_regex.search(str(col)):
            temporal_cols.append(col)

        # 4. Missing values (including whitespace-only strings)
        isna_mask = df[col].isna()
        is_str_col = pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object
        if is_str_col:
            blank_mask = df[col].astype(str).str.strip().isin(["", "NA", "N/A", "null", "?", "nan", "NaN"])
            missing_mask = isna_mask | blank_mask
        else:
            missing_mask = isna_mask

        missing_count = int(missing_mask.sum())
        if missing_count > 0:
            missing_val_cols.append({
                "column": col,
                "missing_count": missing_count,
                "missing_percentage": round((missing_count / total_rows) * 100, 2)
            })

        # Check for dirty numeric columns stored as object/string with blanks
        if is_str_col and not is_100_pct_unique:
            converted = pd.to_numeric(df[col], errors="coerce")
            valid_numeric = converted.notna().sum()
            non_empty = (df[col].astype(str).str.strip() != "").sum()
            if non_empty > 10 and (valid_numeric / non_empty) > 0.90 and valid_numeric < total_rows:
                dirty_numeric_cols.append({
                    "column": col,
                    "valid_numeric_count": int(valid_numeric),
                    "blank_or_invalid_count": int(total_rows - valid_numeric),
                    "suggestion": "Convert with pd.to_numeric(..., errors='coerce') and impute missing values"
                })

    missing_cells = sum(m["missing_count"] for m in missing_val_cols)

    # 5. Target analysis
    severe_class_imbalance: Optional[Dict[str, Any]] = None
    target_proxies = []

    if target_col and target_col in df.columns:
        target_series = df[target_col].dropna()
        n_target_unique = target_series.nunique()

        # Class imbalance check for categorical / discrete targets
        if n_target_unique <= 10:
            val_counts = target_series.value_counts(normalize=True).to_dict()
            min_ratio = float(min(val_counts.values()))
            if min_ratio < 0.15:
                severe_class_imbalance = {
                    "is_severe": True,
                    "target_column": target_col,
                    "minority_class_ratio": round(min_ratio, 4),
                    "class_distribution": {str(k): round(float(v), 4) for k, v in val_counts.items()},
                    "threshold": 0.15
                }

        # Target proxy leakage check (Pearson correlation > 0.95)
        try:
            target_numeric = pd.to_numeric(df[target_col], errors="coerce")
            # If target is binary string categories (e.g. 'Yes'/'No', 'churn'/'no churn')
            if target_numeric.isna().sum() > total_rows * 0.5 and n_target_unique == 2:
                u_vals = list(target_series.unique())
                target_numeric = df[target_col].map({u_vals[0]: 0, u_vals[1]: 1})

            if target_numeric.notna().sum() > 1:
                for col in df.columns:
                    if col == target_col:
                        continue
                    try:
                        col_numeric = pd.to_numeric(df[col], errors="coerce")
                        if col_numeric.notna().sum() > 1:
                            valid_mask = target_numeric.notna() & col_numeric.notna()
                            if valid_mask.sum() > 2:
                                corr = float(target_numeric[valid_mask].corr(col_numeric[valid_mask]))
                                if not np.isnan(corr) and abs(corr) > 0.95:
                                    target_proxies.append({
                                        "column": col,
                                        "correlation": round(corr, 4),
                                        "warning": "Absolute Pearson correlation > 0.95 with target"
                                    })
                    except Exception:
                        continue
        except Exception:
            pass

    return {
        "rows": total_rows,
        "columns": total_cols,
        "missing_cells_total": missing_cells,
        "flags": {
            "HIGH_CARDINALITY_ID_COLUMN": high_cardinality_id_cols,
            "CONSTANT_ZERO_VARIANCE_COLUMN": zero_variance_cols,
            "TEMPORAL_COLUMN_DETECTED": temporal_cols,
            "SEVERE_CLASS_IMBALANCE": severe_class_imbalance,
            "SUSPECTED_TARGET_PROXY_LEAKAGE": target_proxies,
            "MISSING_VALUE_COLUMNS": missing_val_cols,
            "DIRTY_NUMERIC_COLUMNS_DETECTED": dirty_numeric_cols
        }
    }


class MLScriptVisitor(ast.NodeVisitor):
    """AST Visitor to inspect machine learning scripts for data leakage and anti-patterns."""

    TRANSFORMER_CLASSES = {
        "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
        "SimpleImputer", "KNNImputer", "IterativeImputer",
        "OneHotEncoder", "OrdinalEncoder", "TargetEncoder",
        "SMOTE", "RandomOverSampler", "RandomUnderSampler", "ADASYN"
    }

    PIPELINE_CLASSES = {"Pipeline", "ColumnTransformer", "make_pipeline", "make_column_transformer"}

    METRIC_NAMES = {
        "accuracy_score": "accuracy",
        "roc_auc_score": "roc_auc",
        "average_precision_score": "pr_auc",
        "f1_score": "f1"
    }

    def __init__(self):
        self.train_test_split_calls: List[Dict[str, Any]] = []
        self.fit_transform_calls: List[Dict[str, Any]] = []
        self.fit_resample_calls: List[Dict[str, Any]] = []
        self.fit_calls: List[Dict[str, Any]] = []
        self.transformer_instances: Dict[str, str] = {}  # var_name -> class_name
        self.pipeline_instances: Dict[str, str] = {}     # var_name -> class_name
        self.imported_metrics: set = set()
        self.used_metrics: set = set()

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            name = alias.name
            if name in self.METRIC_NAMES:
                self.imported_metrics.add(name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            for m in self.METRIC_NAMES:
                if m in alias.name:
                    self.imported_metrics.add(m)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Detect variable assignment to transformer or pipeline instances
        if isinstance(node.value, ast.Call):
            func_name = self._get_callable_name(node.value.func)
            if func_name in self.TRANSFORMER_CLASSES:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.transformer_instances[target.id] = func_name
            elif func_name in self.PIPELINE_CLASSES:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.pipeline_instances[target.id] = func_name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = self._get_callable_name(node.func)

        # 1. train_test_split call
        if func_name == "train_test_split" or func_name.endswith(".train_test_split"):
            has_stratify = any(kw.arg == "stratify" for kw in node.keywords)
            self.train_test_split_calls.append({
                "line": node.lineno,
                "has_stratify": has_stratify
            })

        # 2. fit_transform call
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit_transform":
            caller_name = self._get_callable_name(node.func.value)
            self.fit_transform_calls.append({
                "line": node.lineno,
                "caller": caller_name
            })

        # 3. fit_resample call (SMOTE, etc.)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit_resample":
            caller_name = self._get_callable_name(node.func.value)
            self.fit_resample_calls.append({
                "line": node.lineno,
                "caller": caller_name
            })

        # 4. fit call
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            caller_name = self._get_callable_name(node.func.value)
            self.fit_calls.append({
                "line": node.lineno,
                "caller": caller_name
            })

        # 5. Metric calls
        for m in self.METRIC_NAMES:
            if func_name == m or func_name.endswith(f".{m}"):
                self.used_metrics.add(m)

        self.generic_visit(node)

    def _get_callable_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val_name = self._get_callable_name(node.value)
            return f"{val_name}.{node.attr}" if val_name else node.attr
        elif isinstance(node, ast.Call):
            # E.g. StandardScaler().fit_transform()
            return self._get_callable_name(node.func)
        return ""


def audit_ml_script(script_path: str) -> Dict[str, Any]:
    """
    Audits a Python machine learning script using AST and regex analysis:
    - PRE_SPLIT_DATA_LEAKAGE (Critical): fit_transform / fit_resample before train_test_split or missing split
    - UNSAFE_STANDALONE_TRANSFORMER: Standalone scalers/imputers outside Pipeline/ColumnTransformer
    - MISSING_STRATIFICATION: train_test_split called without stratify parameter
    - ACCURACY_TRAP_RISK: accuracy_score used without ROC-AUC, PR-AUC, or F1
    """
    if not os.path.exists(script_path):
        return {
            "error": f"Script not found: {script_path}",
            "code_leakage_issues": ["SCRIPT_NOT_FOUND"],
            "verification_passed": False
        }

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code, filename=script_path)
    except Exception as e:
        return {
            "error": f"Failed to parse script AST: {str(e)}",
            "code_leakage_issues": ["SYNTAX_ERROR"],
            "verification_passed": False
        }

    visitor = MLScriptVisitor()
    visitor.visit(tree)

    critical_issues: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    split_lines = [c["line"] for c in visitor.train_test_split_calls]
    earliest_split_line = min(split_lines) if split_lines else None

    # Check 1: PRE_SPLIT_DATA_LEAKAGE
    # Check .fit_transform() and .fit_resample()
    all_transform_calls = []
    for c in visitor.fit_transform_calls:
        all_transform_calls.append({"type": ".fit_transform()", "line": c["line"], "caller": c["caller"]})
    for c in visitor.fit_resample_calls:
        all_transform_calls.append({"type": ".fit_resample()", "line": c["line"], "caller": c["caller"]})

    for call in all_transform_calls:
        call_line = call["line"]
        call_type = call["type"]
        caller = call["caller"]

        if earliest_split_line is None:
            # Transformer used without any train_test_split
            critical_issues.append({
                "type": "PRE_SPLIT_DATA_LEAKAGE",
                "severity": "CRITICAL",
                "line": call_line,
                "description": f"{call_type} called on line {call_line} with no train_test_split() found in script."
            })
        elif call_line < earliest_split_line:
            # Transformer fit before train_test_split
            critical_issues.append({
                "type": "PRE_SPLIT_DATA_LEAKAGE",
                "severity": "CRITICAL",
                "line": call_line,
                "description": f"{call_type} called on line {call_line} before train_test_split() on line {earliest_split_line}."
            })

    # Check 2: UNSAFE_STANDALONE_TRANSFORMER
    # Calling .fit_transform() on standalone scalers/imputers rather than inside Pipeline/ColumnTransformer
    for c in visitor.fit_transform_calls:
        caller = c["caller"]
        line = c["line"]
        # If caller is known transformer class or instance of transformer
        is_transformer_inst = caller in visitor.transformer_instances
        is_direct_class = any(cls in caller for cls in visitor.TRANSFORMER_CLASSES)
        is_pipeline_inst = caller in visitor.pipeline_instances or any(p in caller for p in visitor.PIPELINE_CLASSES)

        if (is_transformer_inst or is_direct_class) and not is_pipeline_inst:
            critical_issues.append({
                "type": "UNSAFE_STANDALONE_TRANSFORMER",
                "severity": "CRITICAL",
                "line": line,
                "description": f"Standalone transformer '{caller}' called .fit_transform() on line {line} outside of Pipeline or ColumnTransformer."
            })

    # Check 3: MISSING_STRATIFICATION
    for split_call in visitor.train_test_split_calls:
        if not split_call["has_stratify"]:
            warnings.append({
                "type": "MISSING_STRATIFICATION",
                "severity": "WARNING",
                "line": split_call["line"],
                "description": f"train_test_split() at line {split_call['line']} called without 'stratify=' parameter."
            })

    # Check 4: ACCURACY_TRAP_RISK
    # If accuracy_score is imported or used, but no roc_auc_score, average_precision_score, or f1_score
    accuracy_present = ("accuracy_score" in visitor.imported_metrics) or ("accuracy_score" in visitor.used_metrics)
    robust_metrics = {"roc_auc_score", "average_precision_score", "f1_score"}
    has_robust_metric = bool(visitor.imported_metrics.intersection(robust_metrics) or visitor.used_metrics.intersection(robust_metrics))

    if accuracy_present and not has_robust_metric:
        warnings.append({
            "type": "ACCURACY_TRAP_RISK",
            "severity": "WARNING",
            "line": None,
            "description": "accuracy_score is imported/used without robust imbalanced metrics (roc_auc_score, average_precision_score, f1_score)."
        })

    code_leakage_issues = [issue["type"] for issue in critical_issues]
    verification_passed = (len(critical_issues) == 0)

    return {
        "script_path": script_path,
        "line_detections": {
            "train_test_split_lines": split_lines,
            "fit_transform_lines": [c["line"] for c in visitor.fit_transform_calls],
            "fit_resample_lines": [c["line"] for c in visitor.fit_resample_calls],
            "fit_lines": [c["line"] for c in visitor.fit_calls],
            "transformers_detected": list(visitor.transformer_instances.values()),
            "pipelines_detected": list(visitor.pipeline_instances.values())
        },
        "critical_issues": critical_issues,
        "warnings": warnings,
        "code_leakage_issues": code_leakage_issues,
        "verification_passed": verification_passed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Hygiene: Tabular ML Pre-Flight Auditor & Script Scanner"
    )
    parser.add_argument("--data", type=str, default=None, help="Path to tabular CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--script", type=str, default=None, help="Path to Python ML training script")

    args = parser.parse_args()

    if not args.data and not args.script:
        parser.print_help()
        sys.exit(1)

    output: Dict[str, Any] = {}

    if args.data:
        output["dataset_audit"] = audit_csv(args.data, args.target)

    if args.script:
        script_res = audit_ml_script(args.script)
        output["script_audit"] = script_res
        output["code_leakage_issues"] = script_res.get("code_leakage_issues", [])
        output["verification_passed"] = script_res.get("verification_passed", False)
    elif args.data:
        output["script_audit"] = None

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
