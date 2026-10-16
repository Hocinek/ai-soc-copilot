"""
Evaluates both detectors against the ground-truth labels the generator kept
aside. This is the only script allowed to touch is_anomaly/anomaly_type --
neither detector sees them during fit/score.

Usage:
    python evaluate.py --logs ../results/auth_logs.csv --out ../results/eval_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from detection.features import build_features
from detection.model import IsolationForestDetector, RuleBasedDetector


def evaluate(df: pd.DataFrame, predicted_col: str = "predicted_anomaly") -> dict:
    y_true = df["is_anomaly"]
    y_pred = df[predicted_col]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "alerts_raised": int(y_pred.sum()),
    }


def per_anomaly_type_recall(df: pd.DataFrame, predicted_col: str) -> dict:
    out = {}
    for atype, grp in df[df["is_anomaly"] == 1].groupby("anomaly_type"):
        out[atype] = round(grp[predicted_col].mean(), 3)  # fraction caught
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="../results/auth_logs.csv")
    ap.add_argument("--out", default="../results/eval_report.json")
    args = ap.parse_args()

    raw = pd.read_csv(args.logs)
    feat_df, feature_cols = build_features(raw)

    iso = IsolationForestDetector(contamination=0.1).fit(feat_df, feature_cols)
    iso_scored = iso.score(feat_df)

    rules = RuleBasedDetector()
    rules_scored = rules.score(feat_df)

    hybrid_pred = ((iso_scored["predicted_anomaly"] == 1) | (rules_scored["predicted_anomaly"] == 1)).astype(int)
    hybrid_df = pd.concat([feat_df[["is_anomaly", "anomaly_type"]], hybrid_pred.rename("predicted_anomaly")], axis=1)

    report = {
        "n_events": len(feat_df),
        "n_labeled_anomalies": int(feat_df["is_anomaly"].sum()) if "is_anomaly" in feat_df else None,
        "isolation_forest": evaluate(pd.concat([feat_df[["is_anomaly", "anomaly_type"]],
                                                  iso_scored["predicted_anomaly"]], axis=1)),
        "rule_based_baseline": evaluate(pd.concat([feat_df[["is_anomaly", "anomaly_type"]],
                                                     rules_scored["predicted_anomaly"]], axis=1)),
        "isolation_forest_recall_by_type": per_anomaly_type_recall(
            pd.concat([feat_df[["is_anomaly", "anomaly_type"]], iso_scored["predicted_anomaly"]], axis=1),
            "predicted_anomaly"),
        "rule_based_recall_by_type": per_anomaly_type_recall(
            pd.concat([feat_df[["is_anomaly", "anomaly_type"]], rules_scored["predicted_anomaly"]], axis=1),
            "predicted_anomaly"),
        "hybrid_union": evaluate(hybrid_df),
        "hybrid_recall_by_type": per_anomaly_type_recall(hybrid_df, "predicted_anomaly"),
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
