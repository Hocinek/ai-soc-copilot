"""
End-to-end pipeline: raw auth logs -> features -> anomaly detection ->
MITRE mapping -> LLM triage -> ranked alert list.

Usage:
    python pipeline.py --logs ../results/auth_logs.csv --out ../results/alerts.json --top 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from detection.features import build_features
from detection.model import IsolationForestDetector
from triage.mitre_mapping import map_to_mitre
from triage.llm_triage import generate_triage


def run_pipeline(logs_path: str, top_n: int = 20) -> list[dict]:
    df = pd.read_csv(logs_path)
    feat_df, feature_cols = build_features(df)

    detector = IsolationForestDetector(contamination=0.1).fit(feat_df, feature_cols)
    scored = detector.score(feat_df)

    flagged = scored[scored["predicted_anomaly"] == 1].sort_values(
        "anomaly_score", ascending=False
    ).head(top_n)

    alerts = []
    for _, row in flagged.iterrows():
        event = row.to_dict()
        mitre_matches = map_to_mitre(event)
        triage = generate_triage(event, mitre_matches)
        alerts.append({
            "event_id": event["event_id"],
            "timestamp": str(event["timestamp"]),
            "src_user": event["src_user"],
            "src_ip": event["src_ip"],
            "dst_host": event["dst_host"],
            "anomaly_score": round(float(event["anomaly_score"]), 4),
            "mitre": mitre_matches,
            "triage": triage,
            # ground truth kept for demo/eval transparency, never fed to the model
            "ground_truth_is_anomaly": int(event.get("is_anomaly", -1)),
            "ground_truth_anomaly_type": event.get("anomaly_type", "unknown"),
        })
    return alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="../results/auth_logs.csv")
    ap.add_argument("--out", default="../results/alerts.json")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    alerts = run_pipeline(args.logs, top_n=args.top)
    Path(args.out).write_text(json.dumps(alerts, indent=2, default=str))
    print(f"Wrote {len(alerts)} triaged alerts to {args.out}")
    for a in alerts[:5]:
        print(f"  [{a['triage']['severity'].upper():8s}] {a['dst_host']:10s} "
              f"{a['src_user']:12s} {a['mitre'][0]['technique_name']}")


if __name__ == "__main__":
    main()
