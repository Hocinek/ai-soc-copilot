"""
Feature engineering for auth-log anomaly detection.

Turns each raw event into a feature vector describing the *context* around
it -- what's unusual is rarely the event itself, it's the event relative to
recent history for that user/source. All features are computed from a
rolling window so the detector never needs the ground-truth label.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pandas as pd


WINDOW_SHORT = "10min"   # brute force horizon
WINDOW_MED = "60min"     # lateral movement horizon
WINDOW_TRAVEL = "180min"  # impossible travel horizon


def _rolling_nunique(timestamps: pd.Series, values: pd.Series, window: str) -> np.ndarray:
    """
    Distinct-value count in a trailing time window ending at each row.
    pandas' rolling().apply() can't aggregate non-numeric dtypes (categorical
    columns like hostnames/country codes), so this does the sliding window
    by hand -- fine at this data volume (a few hundred rows per group).
    """
    window_td = pd.Timedelta(window)
    ts = timestamps.to_numpy()
    vals = values.to_numpy()
    result = np.empty(len(vals), dtype=float)
    counts: Counter = Counter()
    window_items: deque = deque()  # (timestamp, value)

    for i in range(len(vals)):
        window_items.append((ts[i], vals[i]))
        counts[vals[i]] += 1
        while ts[i] - window_items[0][0] > np.timedelta64(window_td):
            old_t, old_v = window_items.popleft()
            counts[old_v] -= 1
            if counts[old_v] == 0:
                del counts[old_v]
        result[i] = len(counts)
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: raw events with columns
        event_id, timestamp, src_user, src_ip, src_country, dst_host,
        auth_type, auth_result
    returns: df with added feature columns, indexed identically to input
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["is_off_hours"] = ((df["hour"] < 6) | (df["hour"] > 21)).astype(int)
    df["is_failure"] = (df["auth_result"] == "failure").astype(int)

    # Keep a unique positional key -- timestamps can collide, so we can't
    # reindex on them directly when stitching per-group rolling results back.
    df["_orig_idx"] = df.index

    fail_counts = []
    lateral_counts = []
    travel_flags = []
    unusual_host_flags = []

    # group by src_ip for brute-force-style counting (rolling window needs a
    # DatetimeIndex, so we set it *inside* each group, then key results by
    # the group's own _orig_idx rather than by timestamp)
    for ip, grp in df.groupby("src_ip"):
        grp_ts = grp.set_index("timestamp")
        roll = grp_ts["is_failure"].rolling(WINDOW_SHORT).sum()
        fail_counts.append(pd.Series(roll.values, index=grp["_orig_idx"].values))

    for user, grp in df.groupby("src_user"):
        distinct_hosts = _rolling_nunique(grp["timestamp"], grp["dst_host"], WINDOW_MED)
        lateral_counts.append(pd.Series(distinct_hosts, index=grp["_orig_idx"].values))

        distinct_countries = _rolling_nunique(grp["timestamp"], grp["src_country"], WINDOW_TRAVEL)
        travel_flags.append(pd.Series(distinct_countries, index=grp["_orig_idx"].values))

        # a host this user hasn't touched in their first 20% of activity
        seen_hosts = set()
        flags = []
        baseline_cutoff = max(1, int(len(grp) * 0.2))
        for i, host in enumerate(grp["dst_host"]):
            flags.append(1 if (i >= baseline_cutoff and host not in seen_hosts) else 0)
            seen_hosts.add(host)
        unusual_host_flags.append(pd.Series(flags, index=grp["_orig_idx"].values))

    df = df.set_index("_orig_idx")
    df["failed_attempts_10min_by_ip"] = pd.concat(fail_counts)
    df["distinct_hosts_60min_by_user"] = pd.concat(lateral_counts)
    df["distinct_countries_180min_by_user"] = pd.concat(travel_flags)
    df["unusual_host_for_user"] = pd.concat(unusual_host_flags).astype(int)
    df = df.reset_index(drop=True)

    feature_cols = [
        "is_off_hours",
        "is_failure",
        "failed_attempts_10min_by_ip",
        "distinct_hosts_60min_by_user",
        "distinct_countries_180min_by_user",
        "unusual_host_for_user",
    ]
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols
