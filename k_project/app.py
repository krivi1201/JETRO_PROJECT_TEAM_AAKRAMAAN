"""
app.py — JETRO · SURPTech coaching dashboard (replaces the Streamlit app2.py)

Run:
    python app.py
then open http://127.0.0.1:8000 (the browser opens automatically).

A small FastAPI server reads the CSV/JSON files written by the pipeline
and serves a custom web interface from web/. Nothing here changes the
analysis; it only reads outputs and, on request, runs:
    * run_pipeline.py     (button "Run analysis")
    * ollama_feedback.py  (button "Generate feedback"), exactly as app2.py did:
      participant ID on stdin, answer read from outputs/ollama_feedback.txt,
      saved into outputs/saved_llm_answers.json without removing other
      participants' answers.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
WEB_DIR = BASE_DIR / "web"

ANSWERS_FILE = OUTPUT_DIR / "saved_llm_answers.json"
FEEDBACK_META_FILE = OUTPUT_DIR / "feedback_meta.json"
OLLAMA_SCRIPT = BASE_DIR / "ollama_feedback.py"
LATEST_FEEDBACK_FILE = OUTPUT_DIR / "ollama_feedback.txt"
LATEST_META_FILE = OUTPUT_DIR / "ollama_feedback_meta.json"
PIPELINE_SCRIPT = BASE_DIR / "run_pipeline.py"
PIPELINE_LOG = OUTPUT_DIR / "pipeline.log"

FILES = {
    "features": OUTPUT_DIR / "window_features.csv",
    "labels": OUTPUT_DIR / "overall_expertise.csv",
    "shap": OUTPUT_DIR / "explain_shap_oof_values.csv",
    "importance": OUTPUT_DIR / "explain_shap_feature_importance.csv",
    "predictions": OUTPUT_DIR / "explain_shap_predictions.csv",
    "windows": OUTPUT_DIR / "timing_window_metrics.csv",
    "reference": OUTPUT_DIR / "timing_reference.csv",
    "moments": OUTPUT_DIR / "timing_moments.csv",
    "timing_summary": OUTPUT_DIR / "timing_summary.json",
    "preprocessing": OUTPUT_DIR / "preprocessing_summary.csv",
    "classifiers": OUTPUT_DIR / "classifier_comparison.csv",
    "xgb_metrics": OUTPUT_DIR / "xgboost_metrics.csv",
}
REQUIRED_FOR_UI = ["features", "labels", "shap", "importance", "windows", "moments"]

PARTICIPANTS = [f"{i:02d}" for i in range(1, 22)]
SHAPES = ["box", "cylinder", "sphere", "wooden_corners"]
SHAPE_LABELS = {"box": "Box", "cylinder": "Cylinder", "sphere": "Sphere", "wooden_corners": "Wooden corners"}
EXPERIENCED = "Experienced (inferred)"
OVERALL_EXPERT = "Experienced across all four shapes (inferred)"

sys.path.insert(0, str(BASE_DIR))

app = FastAPI(title="JETRO SURPTech")
_feedback_lock = threading.Lock()
_pipeline = {"proc": None, "started": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(obj):
    """Make numpy/pandas values JSON-safe (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NaT:
        return None
    return obj


def norm_pid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(2)


_csv_cache: dict = {}


def read_csv(key: str) -> pd.DataFrame | None:
    path = FILES[key]
    if not path.exists() or path.stat().st_size == 0:
        return None
    mtime = path.stat().st_mtime
    hit = _csv_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if "participant_folder" in df.columns:
        df["participant_folder"] = norm_pid(df["participant_folder"])
    _csv_cache[key] = (mtime, df)
    return df


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Feedback (same behaviour as app2.py)
# ---------------------------------------------------------------------------
def load_answers():
    """Load saved participant answers from JSON."""
    if not ANSWERS_FILE.exists():
        return {}
    with ANSWERS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{ANSWERS_FILE.name} must contain a JSON object.")
    return {
        str(pid).strip().zfill(2): str(answer).strip()
        for pid, answer in data.items()
        if isinstance(answer, str) and answer.strip()
    }


def _atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_answer(participant_id, answer):
    """Save one answer without deleting other participant answers."""
    existing = read_json(ANSWERS_FILE, {})
    if not isinstance(existing, dict):
        raise ValueError(f"{ANSWERS_FILE.name} must contain a JSON object.")
    existing[participant_id] = answer.strip()
    _atomic_write_json(ANSWERS_FILE, existing)


def extract_answer(text):
    """Remove a leading participant label if the pipeline writes one."""
    answer = text.strip()
    if answer.lower().startswith("participant:"):
        parts = answer.split("\n\n", 1)
        if len(parts) == 2:
            answer = parts[1].strip()
    return answer


def run_existing_ollama_pipeline(participant_id):
    """Run ollama_feedback.py and read its final answer from the output file."""
    if not OLLAMA_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find {OLLAMA_SCRIPT.name}. Keep it beside app.py.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in (LATEST_FEEDBACK_FILE, LATEST_META_FILE):
        if f.exists():
            f.unlink()

    completed = subprocess.run(
        [sys.executable, str(OLLAMA_SCRIPT)],
        input=f"{participant_id}\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=str(BASE_DIR),
        timeout=300,
        # Windows would otherwise encode the child's output as cp1252 and
        # fail on characters such as "≥" in the model's answer.
        env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
    )
    print("\n" + "=" * 72)
    print(f"OLLAMA_FEEDBACK.PY — PARTICIPANT {participant_id}")
    print("=" * 72)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print("[stderr]")
        print(completed.stderr.rstrip())
    print("=" * 72, flush=True)

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"ollama_feedback.py exited with code {completed.returncode}.\n"
            f"{details[-1500:] or 'No error details were printed.'}"
        )
    if not LATEST_FEEDBACK_FILE.exists():
        raise RuntimeError("ollama_feedback.py finished but did not create outputs/ollama_feedback.txt.")

    answer = extract_answer(LATEST_FEEDBACK_FILE.read_text(encoding="utf-8"))
    if not answer:
        raise RuntimeError("outputs/ollama_feedback.txt is empty.")
    save_answer(participant_id, answer)

    meta = read_json(LATEST_META_FILE, {})
    all_meta = read_json(FEEDBACK_META_FILE, {})
    all_meta[participant_id] = meta or {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feedback_version": 1,
    }
    _atomic_write_json(FEEDBACK_META_FILE, all_meta)
    return answer, all_meta[participant_id]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def shape_status_table() -> dict:
    labels = read_csv("labels")
    out = {}
    if labels is None:
        return out
    for _, row in labels.iterrows():
        out[row["participant_folder"]] = {
            "overall": row.get("overall_inferred_status"),
            "shapes": {s: (row.get(f"{s}_status") if isinstance(row.get(f"{s}_status"), str) else None) for s in SHAPES},
        }
    return out


def predictions_table() -> dict:
    shap = read_csv("shap")
    if shap is None or shap.empty:
        return {}
    out = {}
    for _, row in shap.iterrows():
        out[row["participant_folder"]] = {
            "probability_expert": row.get("predicted_probability_expert"),
            "predicted_label": row.get("predicted_label"),
            "true_inferred_label": row.get("true_inferred_label"),
        }
    return out


def moments_for(pid: str | None = None) -> pd.DataFrame:
    moments = read_csv("moments")
    if moments is None or moments.empty:
        return pd.DataFrame()
    if pid is not None:
        moments = moments[moments["participant_folder"] == pid]
    return moments.sort_values("max_abs_z", ascending=False)


def moment_records(df: pd.DataFrame) -> list:
    cols = ["participant_folder", "object_type", "task", "recording_id", "metric", "metric_label", "direction",
            "start_s", "end_s", "peak_s", "start_label", "end_label", "peak_label", "value", "reference_median",
            "ratio_to_reference", "max_abs_z", "n_windows", "description", "tip", "reference_group",
            "recording_duration_s", "start_time_raw", "peak_time_raw",
            "pattern", "coverage", "worst_start_s", "worst_end_s", "worst_start_label", "worst_end_label"]
    if df.empty:
        return []
    return clean(df[[c for c in cols if c in df.columns]].to_dict(orient="records"))


@lru_cache(maxsize=24)
def _signals(path: str, mtime: float):
    from data_loader import load_pts
    from preprocessing import clean_recording
    from timing_analysis import compute_signals

    df = clean_recording(load_pts(path))
    t0 = float(df["Time[s]"].iloc[0])
    sig = compute_signals(df)
    sig["elapsed"] = sig["t"] - t0
    return sig


def decimate(x: np.ndarray, ys: dict, max_points: int = 4000):
    """Min/max decimation that keeps spikes visible."""
    n = len(x)
    if n <= max_points:
        idx = np.arange(n)
    else:
        bucket = int(math.ceil(n / (max_points / 2)))
        ref = ys.get("force")
        keep = []
        for start in range(0, n, bucket):
            stop = min(start + bucket, n)
            seg = ref[start:stop] if ref is not None else None
            if seg is None or not np.isfinite(seg).any():
                keep.extend([start])
                continue
            keep.extend(sorted({start + int(np.nanargmin(seg)), start + int(np.nanargmax(seg))}))
        idx = np.array(sorted(set(keep)))
    return idx


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/status")
def api_status():
    folders = [p for p in PARTICIPANTS if (DATA_DIR / p).is_dir()]
    recordings = sum(1 for p in folders for s in SHAPES if (DATA_DIR / p / f"{s}.pts").is_file())
    outputs = {k: FILES[k].exists() for k in FILES}
    proc = _pipeline["proc"]
    running = proc is not None and proc.poll() is None
    log_tail = ""
    if PIPELINE_LOG.exists():
        log_tail = "\n".join(PIPELINE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
    return clean({
        "data_dir": str(DATA_DIR),
        "participant_folders": len(folders),
        "recordings": recordings,
        "outputs": outputs,
        "ready": all(outputs[k] for k in REQUIRED_FOR_UI),
        "missing": [FILES[k].name for k in REQUIRED_FOR_UI if not outputs[k]],
        "pipeline": {
            "running": running,
            "returncode": None if proc is None or running else proc.returncode,
            "started": _pipeline["started"],
            "log": log_tail,
        },
        "timing_summary": read_json(FILES["timing_summary"], None),
    })


@app.post("/api/pipeline/run")
def api_pipeline_run():
    proc = _pipeline["proc"]
    if proc is not None and proc.poll() is None:
        return {"started": False, "message": "The analysis is already running."}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = PIPELINE_LOG.open("w", encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    _pipeline["proc"] = subprocess.Popen(
        [sys.executable, "-u", str(PIPELINE_SCRIPT)], cwd=str(BASE_DIR),
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )
    _pipeline["started"] = datetime.now().isoformat(timespec="seconds")
    _csv_cache.clear()
    _signals.cache_clear()
    return {"started": True}


@app.get("/api/participants")
def api_participants():
    status = shape_status_table()
    preds = predictions_table()
    answers = {}
    try:
        answers = load_answers()
    except Exception:
        pass
    meta = read_json(FEEDBACK_META_FILE, {})
    moments = moments_for()
    windows = read_csv("windows")
    out = []
    for pid in PARTICIPANTS:
        m = moments[moments["participant_folder"] == pid] if not moments.empty else moments
        st_ = status.get(pid, {})
        top = moment_records(m.head(1))
        out.append({
            "pid": pid,
            "has_data": windows is not None and not windows.empty and pid in set(windows["participant_folder"]),
            "overall": st_.get("overall"),
            "is_expert": st_.get("overall") == OVERALL_EXPERT,
            "shapes": st_.get("shapes", {}),
            "n_novice_shapes": sum(1 for v in st_.get("shapes", {}).values() if v and v != EXPERIENCED),
            "n_moments": int(len(m)),
            "top_moment": top[0] if top else None,
            "has_feedback": pid in answers,
            "feedback_has_timing": bool(meta.get(pid, {}).get("feedback_version", 1) >= 2),
            "prediction": preds.get(pid),
        })
    return clean(out)


@app.get("/api/participant/{pid}")
def api_participant(pid: str):
    pid = pid.zfill(2)
    if pid not in PARTICIPANTS:
        raise HTTPException(404, "Unknown participant")
    status = shape_status_table().get(pid, {})
    windows = read_csv("windows")
    shapes = []
    moments = moments_for(pid)
    for s in SHAPES:
        w = windows[(windows["participant_folder"] == pid) & (windows["object_type"] == s)] if windows is not None else pd.DataFrame()
        shapes.append({
            "shape": s, "label": SHAPE_LABELS[s],
            "status": status.get("shapes", {}).get(s),
            "duration_s": float(w["recording_duration_s"].iloc[0]) if len(w) else None,
            "n_windows": int(len(w)),
            "n_moments": int((moments["object_type"] == s).sum()) if not moments.empty else 0,
        })

    try:
        answer = load_answers().get(pid)
    except Exception as exc:
        answer, _ = None, exc
    meta = read_json(FEEDBACK_META_FILE, {}).get(pid)

    shap_rows = []
    shap = read_csv("shap")
    importance = read_csv("importance")
    features = read_csv("features")
    if shap is not None and not shap.empty and pid in set(shap["participant_folder"]):
        row = shap[shap["participant_folder"] == pid].iloc[0]
        cols = [c for c in shap.columns if c.startswith("SHAP_")]
        medians = {}
        if features is not None:
            pw = features[features["participant_folder"] == pid]
            medians = pw.median(numeric_only=True).to_dict()
        imp = dict(zip(importance["feature"], importance["mean_absolute_shap"])) if importance is not None else {}
        for c in cols:
            f = c[5:]
            shap_rows.append({"feature": f, "shap": row[c], "participant_value": medians.get(f), "global_importance": imp.get(f)})
        shap_rows.sort(key=lambda r: -abs(r["shap"]) if r["shap"] is not None and np.isfinite(r["shap"]) else 0)

    return clean({
        "pid": pid,
        "overall": status.get("overall"),
        "is_expert": status.get("overall") == OVERALL_EXPERT,
        "shapes": shapes,
        "prediction": predictions_table().get(pid),
        "moments": moment_records(moments),
        "feedback": {"text": answer, "meta": meta},
        "shap": shap_rows[:12],
    })


@app.get("/api/participant/{pid}/session/{shape}")
def api_session(pid: str, shape: str):
    pid = pid.zfill(2)
    if pid not in PARTICIPANTS or shape not in SHAPES:
        raise HTTPException(404, "Unknown participant or task")
    path = DATA_DIR / pid / f"{shape}.pts"
    signals = None
    error = None
    if path.exists():
        try:
            sig = _signals(str(path), path.stat().st_mtime)
            idx = decimate(sig["elapsed"].to_numpy(), {"force": sig["force"].to_numpy()})
            signals = {k: sig[k].to_numpy()[idx].tolist() for k in ("elapsed", "force", "torque", "speed", "angular_speed")}
        except Exception as exc:
            error = f"Could not read the recording: {exc}"
    else:
        error = f"Recording not found: data/{pid}/{shape}.pts"

    windows = read_csv("windows")
    w = windows[(windows["participant_folder"] == pid) & (windows["object_type"] == shape)].sort_values("start_s") if windows is not None else pd.DataFrame()
    reference = read_csv("reference")
    ref = {}
    if reference is not None and not reference.empty:
        r = reference[(reference["participant_folder"] == pid) & (reference["object_type"] == shape)]
        ref = {row["metric"]: {"median": row["reference_median"], "q25": row["reference_q25"], "q75": row["reference_q75"],
                               "group": row["reference_group"], "participants": row["reference_participants"]}
               for _, row in r.iterrows()}
    summary = read_json(FILES["timing_summary"], {}) or {}
    metric_meta = summary.get("metrics", {})
    moments = moments_for(pid)
    moments = moments[moments["object_type"] == shape] if not moments.empty else moments
    win = {}
    if len(w):
        win = {"start_s": w["start_s"].tolist(), "end_s": w["end_s"].tolist(),
               "mid_s": ((w["start_s"] + w["end_s"]) / 2).tolist()}
        for m in metric_meta:
            if m in w:
                win[m] = w[m].tolist()
                win[f"z_{m}"] = w[f"z_{m}"].tolist()
    return clean({
        "pid": pid, "shape": shape, "label": SHAPE_LABELS[shape],
        "signals": signals, "error": error,
        "windows": win, "reference": ref,
        "metrics": {k: {"label": v.get("label"), "unit": v.get("unit"), "direction": v.get("direction")} for k, v in metric_meta.items()},
        "moments": moment_records(moments),
        "duration_s": float(w["recording_duration_s"].iloc[0]) if len(w) else None,
        "z_threshold": summary.get("z_threshold"),
    })


@app.get("/api/cohort")
def api_cohort(shape: str = "box", metric: str = "force_variability"):
    windows = read_csv("windows")
    status = shape_status_table()
    summary = read_json(FILES["timing_summary"], {}) or {}
    rows = []
    if windows is not None and not windows.empty and metric in windows.columns:
        w = windows[windows["object_type"] == shape]
        grp = w.groupby("participant_folder")[metric]
        for pid, series in grp:
            v = series.dropna().to_numpy()
            if not len(v):
                continue
            q25, med, q75 = np.percentile(v, [25, 50, 75])
            rows.append({"pid": pid, "median": med, "q25": q25, "q75": q75, "n_windows": int(len(v)),
                         "status": status.get(pid, {}).get("shapes", {}).get(shape)})
        exp = w[w["participant_folder"].isin([r["pid"] for r in rows if r["status"] == EXPERIENCED])][metric].dropna()
        exp_median = float(np.median(exp)) if len(exp) else None
    else:
        exp_median = None
    moments = moments_for()
    counts = []
    for pid in PARTICIPANTS:
        m = moments[moments["participant_folder"] == pid] if not moments.empty else moments
        counts.append({"pid": pid, **{s: int((m["object_type"] == s).sum()) if len(m) else 0 for s in SHAPES}})
    importance = read_csv("importance")
    classifiers = read_csv("classifiers")
    return clean({
        "shape": shape, "metric": metric,
        "metrics": {k: {"label": v.get("label"), "unit": v.get("unit")} for k, v in summary.get("metrics", {}).items()},
        "participants": rows, "experienced_median": exp_median,
        "moment_counts": counts,
        "importance": importance.head(15).to_dict(orient="records") if importance is not None else [],
        "classifiers": classifiers.to_dict(orient="records") if classifiers is not None else [],
        "preprocessing": (read_csv("preprocessing").to_dict(orient="records") if read_csv("preprocessing") is not None else []),
    })


@app.post("/api/participant/{pid}/feedback")
def api_feedback(pid: str):
    pid = pid.zfill(2)
    if pid not in PARTICIPANTS:
        raise HTTPException(404, "Unknown participant")
    if not _feedback_lock.acquire(blocking=False):
        return JSONResponse({"ok": False, "error": "Another feedback request is still running. Try again in a moment."}, status_code=409)
    try:
        answer, meta = run_existing_ollama_pipeline(pid)
        return clean({"ok": True, "text": answer, "meta": meta})
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "The feedback process took longer than five minutes. Check that Ollama is running."}, status_code=504)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    finally:
        _feedback_lock.release()


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
@app.get("/vendor/plotly.min.js")
def plotly_js():
    import plotly
    path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not path.exists():
        raise HTTPException(404, "plotly.min.js not found; run pip install plotly")
    return FileResponse(path, media_type="application/javascript", headers={"Cache-Control": "max-age=86400"})


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    host, port = "127.0.0.1", int(os.environ.get("JETRO_PORT", "8000"))
    url = f"http://{host}:{port}"
    print(f"\nJETRO dashboard running at {url}  (Ctrl+C to stop)\n", flush=True)
    if os.environ.get("JETRO_NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
