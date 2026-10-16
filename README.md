# AI SOC Copilot

A small end-to-end pipeline that detects anomalous authentication behavior
with unsupervised ML, maps it to MITRE ATT&CK deterministically, and uses an
LLM to turn each flagged event into an analyst-readable triage note —
severity, plain-language summary, recommended action.

Built as a companion piece to offensive-side work (bug bounty research on
parser/deserialization vulnerabilities): this project deliberately looks at
the defensive side of the same skill set — applying ML/NLP to detection and
triage instead of exploitation.

## Why this design

**Detection and explanation are kept in separate stages, on purpose.**
The anomaly detector (Isolation Forest) never sees the LLM, and the LLM
never picks a MITRE technique — a deterministic rules module
(`mitre_mapping.py`) does that from the same features the detector used.
The LLM's only job is turning structured signal into something a human can
skim in a queue of 200 alerts. This avoids the two failure modes that make
"AI SOC tools" untrustworthy in practice: a black-box model nobody can
audit, and an LLM inventing technique IDs or severities with no ground in
the data.

**The evaluation is honest, not flattering.** `evaluate.py` compares the ML
detector against a fixed-threshold rule-based baseline on the same labeled
data. On the generated dataset, the rule baseline actually *beats* Isolation
Forest on raw precision/recall — but the two catch different anomaly types
well (rules: brute force; ML: lateral movement), so a hybrid (union of both)
outperforms either alone:

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Rule-based baseline | 0.93 | 0.82 | 0.87 |
| Isolation Forest | 0.89 | 0.70 | 0.78 |
| **Hybrid (union)** | **0.92** | **0.95** | **0.93** |

Recall by anomaly type tells the more useful story:

| Anomaly type | Rules | Isolation Forest |
|---|---|---|
| Brute force | 0.93 | 0.67 |
| Lateral movement | 0.49 | 0.91 |
| Impossible travel | 0.50 | 0.50 |
| Off-hours access | 0.38 | 0.38 |

Rules catch what's a hard fixed threshold (many failures fast). ML catches
what's about *shape* rather than a single number (a user suddenly touching
many hosts). Neither should replace the other; that's the actual argument
for a hybrid pipeline, not a marketing claim that "AI beats rules."

## Architecture

```
data/generate_logs.py   synthetic auth logs + 4 labeled anomaly patterns
        │
src/detection/features.py   rolling-window behavioral features per event
src/detection/model.py      IsolationForestDetector + RuleBasedDetector
        │
src/triage/mitre_mapping.py deterministic feature -> MITRE ATT&CK technique
src/triage/llm_triage.py    LLM turns (event, mitre matches) -> triage note
        │
src/pipeline.py             orchestrates the above -> results/alerts.json
src/evaluate.py             ML vs rules vs hybrid, against held-out labels
        │
dashboard/app.py + static/index.html   alert queue UI (FastAPI)
```

## Anomaly patterns modeled

- **Brute force** — many failed auths from one source in a short window
- **Impossible travel** — same user, two distant countries, implausible gap
- **Off-hours access** — success at an unusual hour, to a host they don't normally touch
- **Lateral movement** — one user authenticating to an unusually large number of distinct hosts fast

The dataset is synthetic (`data/generate_logs.py`) rather than a public
capture — this keeps ground-truth labels exact for evaluation and avoids
licensing/PII issues with real auth data, at the cost of not testing against
real-world noise.

## Setup

```bash
pip install -r requirements.txt

# 1. generate the dataset
cd data && python generate_logs.py --out ../results/auth_logs.csv --users 40 --days 14

# 2. run the pipeline (detection -> MITRE -> triage)
cd ../src && python pipeline.py --logs ../results/auth_logs.csv --out ../results/alerts.json --top 20

# 3. evaluate detectors against ground truth
python evaluate.py --logs ../results/auth_logs.csv --out ../results/eval_report.json

# 4. view the dashboard
cd ../dashboard && uvicorn app:app --reload --port 8000
# open http://127.0.0.1:8000
```

LLM triage works without any API key (deterministic offline fallback in
`llm_triage.py`). To use real Claude-generated triage notes, copy
`.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

Run tests: `pytest tests/`

## Known limitations

- Synthetic data only — no validation against a real SOC environment's noise floor
- Isolation Forest is retrained on the full batch each run; a production version would need incremental/online scoring
- MITRE mapping covers 4 techniques as a proof of concept, not the full ATT&CK matrix
- No feedback loop yet from analyst triage decisions back into the detector

## Possible extensions

- Add a supervised layer once enough analyst-confirmed labels accumulate
- Swap Isolation Forest for a sequence model (e.g. an autoencoder over
  per-user event sequences) to catch slower, lower-and-slower attacks
- Feed the LLM layer real threat intel context (recent CVEs, known bad IPs) for richer triage
