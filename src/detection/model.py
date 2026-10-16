"""
Two detectors, so results can be compared honestly:

  - IsolationForestDetector: unsupervised ML, learns "normal" from the
    feature distribution and scores how isolated each point is.
  - RuleBasedDetector: fixed-threshold baseline, the kind of logic a
    traditional SIEM correlation rule would encode.

Neither detector is trained on the ground-truth anomaly labels -- those are
reserved for evaluate.py only.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self.feature_cols: list[str] | None = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "IsolationForestDetector":
        self.feature_cols = feature_cols
        self.model.fit(df[feature_cols])
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # decision_function: higher = more normal. Flip sign so higher = more anomalous.
        df["anomaly_score"] = -self.model.decision_function(df[self.feature_cols])
        df["predicted_anomaly"] = (self.model.predict(df[self.feature_cols]) == -1).astype(int)
        return df


class RuleBasedDetector:
    """Fixed thresholds, no learning -- the baseline the ML model has to beat."""

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        flagged = (
            (df["failed_attempts_10min_by_ip"] >= 10)
            | (df["distinct_hosts_60min_by_user"] >= 7)
            | (df["distinct_countries_180min_by_user"] >= 2)
            | ((df["is_off_hours"] == 1) & (df["unusual_host_for_user"] == 1))
        )
        df["predicted_anomaly"] = flagged.astype(int)
        df["anomaly_score"] = flagged.astype(float)
        return df
