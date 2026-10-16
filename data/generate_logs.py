"""
Synthetic authentication log generator.

Simulates a small org's auth events (think Windows 4624/4625 or SSH auth logs)
over N days, with realistic per-user baselines, then injects four labeled
anomaly patterns on top:

  - brute_force        many failed logins from one source in a short window
  - impossible_travel   same user authenticating from two distant countries
                        within a time gap too short to be physically possible
  - off_hours_access    a legit user succeeding at an unusual hour, to a host
                        they don't normally touch
  - lateral_movement    one user authenticating to an unusually large number
                        of distinct hosts in a short window

Ground-truth labels (is_anomaly, anomaly_type) are kept in a separate column
so the *unsupervised* detector never sees them during training -- they exist
only for evaluation (see evaluate.py).

Usage:
    python generate_logs.py --out ../results/auth_logs.csv --users 40 --days 14
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --- small fixed "world" so results are inspectable and reproducible ---

USERNAMES = [
    "j.martin", "a.dubois", "l.bernard", "c.thomas", "m.robert", "e.richard",
    "s.petit", "n.durand", "t.leroy", "v.moreau", "p.simon", "o.laurent",
    "d.lefebvre", "g.michel", "f.garcia", "b.roux", "h.vincent", "k.muller",
    "r.girard", "y.andre", "w.mercier", "q.dupont", "x.bonnet", "z.francois",
]

COUNTRIES = {
    # country: (label, is_far_from_home -- used to build implausible pairs)
    "FR": "France", "DE": "Germany", "BE": "Belgium", "CH": "Switzerland",
}
FAR_COUNTRIES = ["RU", "CN", "BR", "NG", "VN", "KP"]  # attacker-side geos

HOSTS = [f"host-{i:02d}" for i in range(1, 26)]
SENSITIVE_HOSTS = ["host-01", "host-02", "host-03"]  # domain controllers etc.

AUTH_TYPES = ["interactive", "network", "remote_interactive", "service"]


def rand_ip(country_hint: str | None = None) -> str:
    if country_hint in FAR_COUNTRIES:
        return f"185.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    return f"10.{random.randint(0,20)}.{random.randint(0,255)}.{random.randint(1,254)}"


class UserProfile:
    """A user's normal-life baseline: where, when, and what they touch."""

    def __init__(self, username: str):
        self.username = username
        self.home_country = random.choice(list(COUNTRIES.keys()))
        self.home_ip = rand_ip(self.home_country)
        self.work_start = random.choice([7, 8, 8, 9])
        self.work_end = self.work_start + random.choice([8, 9, 9, 10])
        self.usual_hosts = random.sample(HOSTS, k=random.randint(2, 4))


def gen_normal_events(users: list[UserProfile], days: int, start: datetime) -> list[dict]:
    events = []
    for day in range(days):
        day_start = start + timedelta(days=day)
        for user in users:
            # 3-12 normal events per working day, weekends lighter
            is_weekend = (day_start.weekday() >= 5)
            n_events = random.randint(0, 3) if is_weekend else random.randint(3, 12)
            for _ in range(n_events):
                hour = random.randint(user.work_start, min(user.work_end, 23))
                ts = day_start.replace(
                    hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
                )
                result = "success" if random.random() > 0.03 else "failure"  # occasional typo
                events.append({
                    "timestamp": ts,
                    "src_user": user.username,
                    "src_ip": user.home_ip,
                    "src_country": user.home_country,
                    "dst_host": random.choice(user.usual_hosts),
                    "auth_type": random.choice(AUTH_TYPES),
                    "auth_result": result,
                    "is_anomaly": 0,
                    "anomaly_type": "none",
                })
    return events


def inject_brute_force(events: list[dict], users: list[UserProfile], start: datetime, days: int, n_incidents: int):
    for _ in range(n_incidents):
        target = random.choice(users)
        attacker_ip = rand_ip(random.choice(FAR_COUNTRIES))
        incident_start = start + timedelta(
            days=random.randint(0, days - 1), hours=random.randint(0, 23), minutes=random.randint(0, 59)
        )
        n_attempts = random.randint(20, 45)
        for i in range(n_attempts):
            ts = incident_start + timedelta(seconds=i * random.randint(3, 15))
            success = (i == n_attempts - 1) and random.random() < 0.3  # sometimes it lands
            events.append({
                "timestamp": ts,
                "src_user": target.username,
                "src_ip": attacker_ip,
                "src_country": "XX",
                "dst_host": random.choice(SENSITIVE_HOSTS),
                "auth_type": "network",
                "auth_result": "success" if success else "failure",
                "is_anomaly": 1,
                "anomaly_type": "brute_force",
            })


def inject_impossible_travel(events: list[dict], users: list[UserProfile], start: datetime, days: int, n_incidents: int):
    for _ in range(n_incidents):
        user = random.choice(users)
        day = start + timedelta(days=random.randint(0, days - 1))
        t1 = day.replace(hour=random.randint(8, 14), minute=random.randint(0, 59))
        gap_minutes = random.randint(5, 45)  # far too short to fly between continents
        t2 = t1 + timedelta(minutes=gap_minutes)
        far_country = random.choice(FAR_COUNTRIES)
        for ts, country in [(t1, user.home_country), (t2, far_country)]:
            events.append({
                "timestamp": ts,
                "src_user": user.username,
                "src_ip": rand_ip(country),
                "src_country": country,
                "dst_host": random.choice(user.usual_hosts),
                "auth_type": "remote_interactive",
                "auth_result": "success",
                "is_anomaly": 1,
                "anomaly_type": "impossible_travel",
            })


def inject_off_hours(events: list[dict], users: list[UserProfile], start: datetime, days: int, n_incidents: int):
    for _ in range(n_incidents):
        user = random.choice(users)
        day = start + timedelta(days=random.randint(0, days - 1))
        ts = day.replace(hour=random.choice([1, 2, 3, 4]), minute=random.randint(0, 59))
        events.append({
            "timestamp": ts,
            "src_user": user.username,
            "src_ip": user.home_ip,
            "src_country": user.home_country,
            "dst_host": random.choice(SENSITIVE_HOSTS),
            "auth_type": "interactive",
            "auth_result": "success",
            "is_anomaly": 1,
            "anomaly_type": "off_hours_access",
        })


def inject_lateral_movement(events: list[dict], users: list[UserProfile], start: datetime, days: int, n_incidents: int):
    for _ in range(n_incidents):
        user = random.choice(users)
        incident_start = start + timedelta(
            days=random.randint(0, days - 1), hours=random.randint(8, 20), minutes=random.randint(0, 59)
        )
        targets = random.sample(HOSTS, k=random.randint(8, 15))
        for i, host in enumerate(targets):
            ts = incident_start + timedelta(minutes=i * random.randint(1, 4))
            events.append({
                "timestamp": ts,
                "src_user": user.username,
                "src_ip": user.home_ip,
                "src_country": user.home_country,
                "dst_host": host,
                "auth_type": "network",
                "auth_result": "success",
                "is_anomaly": 1,
                "anomaly_type": "lateral_movement",
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/auth_logs.csv")
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    users = [UserProfile(u) for u in USERNAMES[: args.users]]
    start = datetime(2026, 7, 1)

    events = gen_normal_events(users, args.days, start)
    inject_brute_force(events, users, start, args.days, n_incidents=6)
    inject_impossible_travel(events, users, start, args.days, n_incidents=5)
    inject_off_hours(events, users, start, args.days, n_incidents=8)
    inject_lateral_movement(events, users, start, args.days, n_incidents=4)

    df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    df["event_id"] = df.index.map(lambda i: f"evt-{i:06d}")
    df = df[["event_id", "timestamp", "src_user", "src_ip", "src_country",
             "dst_host", "auth_type", "auth_result", "is_anomaly", "anomaly_type"]]

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} events ({df['is_anomaly'].sum()} labeled anomalous) to {args.out}")
    print(df["anomaly_type"].value_counts())


if __name__ == "__main__":
    main()
