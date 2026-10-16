from datetime import datetime, timedelta

import pandas as pd

from detection.features import build_features, _rolling_nunique
from detection.model import RuleBasedDetector
from triage.mitre_mapping import map_to_mitre


def test_rolling_nunique_basic():
    ts = pd.Series([datetime(2026, 1, 1, 10, 0, i) for i in range(5)])
    vals = pd.Series(["a", "a", "b", "c", "a"])
    result = _rolling_nunique(ts, vals, "10min")
    # all within the same 10-min window, so distinct count grows monotonically
    # until it plateaus at 3 unique values (a, b, c)
    assert list(result) == [1, 1, 2, 3, 3]


def test_rolling_nunique_respects_window():
    ts = pd.Series([
        datetime(2026, 1, 1, 10, 0, 0),
        datetime(2026, 1, 1, 10, 0, 30),
        datetime(2026, 1, 1, 10, 20, 0),  # 20 min later -- outside a 10-min window
    ])
    vals = pd.Series(["a", "b", "c"])
    result = _rolling_nunique(ts, vals, "10min")
    assert result[2] == 1  # only "c" itself should be in window by the 3rd event


def test_build_features_flags_brute_force():
    base = datetime(2026, 1, 1, 9, 0, 0)
    rows = []
    for i in range(15):
        rows.append({
            "event_id": f"e{i}", "timestamp": base + timedelta(seconds=i * 5),
            "src_user": "victim", "src_ip": "203.0.113.9", "src_country": "XX",
            "dst_host": "host-01", "auth_type": "network", "auth_result": "failure",
        })
    df = pd.DataFrame(rows)
    feat_df, feature_cols = build_features(df)
    assert feat_df["failed_attempts_10min_by_ip"].iloc[-1] >= 10

    baseline = RuleBasedDetector().score(feat_df)
    assert baseline["predicted_anomaly"].iloc[-1] == 1


def test_mitre_mapping_covers_brute_force():
    event = {"failed_attempts_10min_by_ip": 12, "src_ip": "1.2.3.4",
              "distinct_countries_180min_by_user": 0, "distinct_hosts_60min_by_user": 1,
              "is_off_hours": 0, "unusual_host_for_user": 0}
    matches = map_to_mitre(event)
    ids = [m["technique_id"] for m in matches]
    assert "T1110" in ids


def test_mitre_mapping_falls_back_when_nothing_matches():
    event = {"failed_attempts_10min_by_ip": 0, "src_ip": "1.2.3.4",
              "distinct_countries_180min_by_user": 0, "distinct_hosts_60min_by_user": 1,
              "is_off_hours": 0, "unusual_host_for_user": 0}
    matches = map_to_mitre(event)
    assert matches[0]["technique_id"] is None
