"""
Minimal FastAPI backend for the SOC copilot dashboard.

Serves the pipeline's output (results/alerts.json, results/eval_report.json)
to a static frontend. Run pipeline.py and evaluate.py first to generate
those files, then:

    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
RESULTS = BASE.parent / "results"

app = FastAPI(title="AI SOC Copilot")


def _load(name: str):
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


@app.get("/api/alerts")
def get_alerts():
    data = _load("alerts.json")
    if data is None:
        return JSONResponse({"error": "no alerts.json yet -- run src/pipeline.py first"}, status_code=404)
    return data


@app.get("/api/eval")
def get_eval():
    data = _load("eval_report.json")
    if data is None:
        return JSONResponse({"error": "no eval_report.json yet -- run src/evaluate.py first"}, status_code=404)
    return data


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
