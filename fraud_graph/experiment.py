"""Chronological PaySim transaction-fraud experiments with Logistic Regression and XGBoost."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import TARGET, build_features, feature_sets


TYPE_LEVELS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
ALERT_BUDGETS = {"0.1%": 0.001, "0.5%": 0.005, "1.0%": 0.01}


def _logistic_model(columns: list[str]) -> Pipeline:
    categorical = [column for column in columns if column == "type"]
    numeric = [column for column in columns if column not in categorical]
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ], remainder="drop")
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(class_weight="balanced", max_iter=300, solver="lbfgs")),
    ])


def _xgb_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Create a numeric float32 matrix with a fixed one-hot transaction-type schema."""
    matrix = frame[columns].copy()
    if "type" in matrix:
        types = pd.Categorical(matrix.pop("type"), categories=TYPE_LEVELS)
        dummies = pd.get_dummies(types, prefix="type", dtype=np.uint8)
        matrix = pd.concat([matrix, dummies.set_index(matrix.index)], axis=1)
    for column in matrix:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype("float32")
    return matrix


def _top_k(y: np.ndarray, scores: np.ndarray, fraction: float) -> dict:
    n_alerts = max(1, int(np.ceil(len(y) * fraction)))
    order = np.argsort(-scores, kind="stable")
    selected = order[:n_alerts]
    threshold = float(scores[order[n_alerts - 1]])
    inclusive_count = int(np.count_nonzero(scores >= threshold))
    tn, fp, fn, tp = confusion_matrix(y, np.isin(np.arange(len(y)), selected), labels=[0, 1]).ravel()
    return {
        "requested_alerts": n_alerts,
        "actual_alerts": int(len(selected)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "score_cutoff": threshold,
        "rows_at_or_above_cutoff": inclusive_count,
        "cutoff_tie_overflow": int(inclusive_count - n_alerts),
        "tie_break": "Stable test-row order selects exactly the requested alert count.",
    }


def _metrics(y: np.ndarray, scores: np.ndarray, primary_fraction: float) -> dict:
    fractions = dict(ALERT_BUDGETS)
    primary_name = f"{100 * primary_fraction:g}%"
    fractions[primary_name] = primary_fraction
    budget_results = {name: _top_k(y, scores, fraction) for name, fraction in fractions.items()}
    return {
        "test_rows": int(len(y)),
        "test_positive_rows": int(y.sum()),
        "test_prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores)),
        "fixed_alert_budgets": budget_results,
        "primary_budget": primary_name,
    }


def _plot_metrics(results: dict, output: Path) -> None:
    names = list(results)
    aps = [results[name]["average_precision"] for name in names]
    p01 = [results[name]["fixed_alert_budgets"]["0.1%"]["precision"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(names, aps, color="#2878B5")
    axes[0].set_title("Test average precision")
    axes[0].set_ylabel("Average precision")
    axes[1].bar(names, p01, color="#59A14F")
    axes[1].set_title("Precision at exact 0.1% alert budget")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(
    input_path: Path,
    outdir: Path,
    *,
    budget_fraction: float = 0.001,
    n_estimators: int = 600,
    early_stopping_rounds: int = 30,
    n_jobs: int = 8,
) -> dict:
    started = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {input_path}", flush=True)
    raw = pd.read_csv(input_path)
    print(f"Building strictly prior-step, pre-transaction features for {len(raw):,} transactions", flush=True)
    fraud_mask = raw[TARGET].eq(1)
    balance_match = raw["amount"].eq(raw["oldbalanceOrg"])
    balance_shortcut_audit = {
        "predicate": "amount == oldbalanceOrg",
        "fraud_rows": int(fraud_mask.sum()),
        "fraud_rows_matching": int((fraud_mask & balance_match).sum()),
        "fraud_fraction_matching": float((fraud_mask & balance_match).sum() / max(1, fraud_mask.sum())),
        "nonfraud_rows": int((~fraud_mask).sum()),
        "nonfraud_rows_matching": int((~fraud_mask & balance_match).sum()),
        "interpretation": "Pre-transaction fields make this a simulator-specific predictive shortcut, not post-transaction leakage; test separately with the no-balance ablation.",
    }
    features = build_features(raw)
    del raw

    train_mask = features.step <= 520
    valid_mask = features.step.between(521, 631)
    test_mask = features.step >= 632
    split_masks = {"train": train_mask, "validation": valid_mask, "test": test_mask}
    split_counts = {
        name: {
            "rows": int(mask.sum()),
            "fraud": int(features.loc[mask, TARGET].sum()),
            "prevalence": float(features.loc[mask, TARGET].mean()),
        }
        for name, mask in split_masks.items()
    }
    if not all(split_counts[name]["fraud"] > 0 for name in split_counts):
        raise ValueError(f"A chronological split has no positive examples: {split_counts}")
    y_train = features.loc[train_mask, TARGET].to_numpy(dtype=np.uint8)
    y_valid = features.loc[valid_mask, TARGET].to_numpy(dtype=np.uint8)
    y_test = features.loc[test_mask, TARGET].to_numpy(dtype=np.uint8)
    y_all = {"train": y_train, "validation": y_valid, "test": y_test}
    test_steps = features.loc[test_mask, "step"].to_numpy(dtype=np.int16)
    model_results: dict[str, dict] = {}
    test_scores: dict[str, np.ndarray] = {}
    importances: dict[str, list[dict]] = {}

    # Transparent non-ML baseline, ranked without any test labels.
    test_type = features.loc[test_mask, "type"].astype(str).to_numpy()
    test_amount = features.loc[test_mask, "amount"].to_numpy(dtype=np.float32)
    rule_scores = np.where(np.isin(test_type, ["TRANSFER", "CASH_OUT"]), np.log1p(test_amount), 0.0)
    model_results["rule_transfer_cashout_amount"] = _metrics(y_test, rule_scores, budget_fraction)
    test_scores["rule_transfer_cashout_amount"] = rule_scores.astype(np.float32)

    # Logistic Regression is retained as a conventional transaction baseline.
    feature_map = feature_sets()
    lr_columns = feature_map["A_pretransaction"]
    print("Fitting Logistic Regression transaction baseline", flush=True)
    lr_model = _logistic_model(lr_columns)
    lr_started = time.perf_counter()
    lr_model.fit(features.loc[train_mask, lr_columns], y_train)
    lr_valid = lr_model.predict_proba(features.loc[valid_mask, lr_columns])[:, 1]
    lr_scores = lr_model.predict_proba(features.loc[test_mask, lr_columns])[:, 1].astype(np.float32)
    model_results["logistic_A_pretransaction"] = _metrics(y_test, lr_scores, budget_fraction)
    model_results["logistic_A_pretransaction"]["fit_seconds"] = round(time.perf_counter() - lr_started, 2)
    model_results["logistic_A_pretransaction"]["validation_average_precision"] = float(average_precision_score(y_valid, lr_valid))
    test_scores["logistic_A_pretransaction"] = lr_scores
    del lr_model, lr_scores

    positive_weight = float((len(y_train) - int(y_train.sum())) / max(1, int(y_train.sum())))
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_estimators": n_estimators,
        "learning_rate": 0.08,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "max_bin": 256,
        "scale_pos_weight": positive_weight,
        "early_stopping_rounds": early_stopping_rounds,
        "n_jobs": n_jobs,
        "random_state": 42,
        "verbosity": 0,
    }
    for name, columns in feature_map.items():
        print(f"Preparing XGBoost matrix for {name}", flush=True)
        x_train = _xgb_matrix(features.loc[train_mask], columns)
        x_valid = _xgb_matrix(features.loc[valid_mask], columns)
        x_test = _xgb_matrix(features.loc[test_mask], columns)
        model = XGBClassifier(**xgb_params)
        fit_started = time.perf_counter()
        print(f"Fitting XGBoost {name}", flush=True)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        valid_scores = model.predict_proba(x_valid)[:, 1]
        scores = model.predict_proba(x_test)[:, 1].astype(np.float32)
        result = _metrics(y_test, scores, budget_fraction)
        result.update({
            "fit_seconds": round(time.perf_counter() - fit_started, 2),
            "validation_average_precision": float(average_precision_score(y_valid, valid_scores)),
            "best_iteration": int(model.best_iteration) if model.best_iteration is not None else None,
            "training_scale_pos_weight": positive_weight,
            "feature_count": int(x_train.shape[1]),
        })
        model_name = f"xgb_{name}"
        model_results[model_name] = result
        test_scores[model_name] = scores
        importance = pd.DataFrame({"feature": x_train.columns, "gain_weight": model.feature_importances_})
        importances[model_name] = importance.sort_values("gain_weight", ascending=False).head(30).to_dict("records")
        checkpoint = {
            "split_definition": {"train": "step <= 520", "validation": "521 <= step <= 631", "test": "step >= 632"},
            "split_counts": split_counts,
            "models_completed": list(model_results),
            "models": model_results,
        }
        (outdir / "metrics.partial.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
        del x_train, x_valid, x_test, model, scores

    (outdir / "feature_importance.json").write_text(json.dumps(importances, indent=2) + "\n")
    np.savez_compressed(
        outdir / "test_predictions.npz",
        step=test_steps,
        y_true=y_test,
        **{f"score__{name}": scores for name, scores in test_scores.items()},
    )
    result = {
        "input": str(input_path),
        "dataset_rows": int(len(features)),
        "split_definition": {"train": "step <= 520", "validation": "521 <= step <= 631", "test": "step >= 632"},
        "split_counts": split_counts,
        "target": "isFraud transaction label; never used in group generation or account features.",
        "feature_availability": "Only current transaction type, amount, pre-transaction balances, step, and strictly prior-step account/graph aggregates are model inputs. Post-transaction balances, isFlaggedFraud, and raw account IDs are excluded.",
        "balance_shortcut_audit": balance_shortcut_audit,
        "alert_budget_fraction": budget_fraction,
        "tie_break_policy": "Stable test-row order selects exactly k rows at each fixed alert budget; inclusive cutoff counts are reported to expose ties.",
        "xgboost_parameters": xgb_params,
        "xgboost_version": xgboost.__version__,
        "models": model_results,
        "artifacts": {"test_predictions": "test_predictions.npz", "feature_importance": "feature_importance.json"},
        "total_runtime_seconds": round(time.perf_counter() - started, 2),
        "peak_process_max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (outdir / "metrics.partial.json").unlink(missing_ok=True)
    _plot_metrics(model_results, outdir / "model_comparison.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--alert-budget", type=float, default=0.001)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, budget_fraction=args.alert_budget,
                         n_estimators=args.n_estimators,
                         early_stopping_rounds=args.early_stopping_rounds,
                         n_jobs=args.n_jobs), indent=2))


if __name__ == "__main__":
    main()
