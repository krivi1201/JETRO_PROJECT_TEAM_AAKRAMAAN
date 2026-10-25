
"""
ollama_feedback.py

Generates coaching feedback for one participant with a local Ollama model.

What changed: the feedback now says WHEN things happened. It reads the
timed moments produced by timing_analysis.py (outputs/timing_moments.csv)
and gives the model the exact task and time of each moment, e.g.
"Box task, 0:12.0-0:16.0 (sharpest at 0:13.4)". The model is told to copy
those times and never invent new ones, and every time it writes is
checked afterwards (see outputs/ollama_feedback_meta.json).

Usage:
    python ollama_feedback.py            (asks for a participant)
    python ollama_feedback.py 07         (participant on the command line)
"""

from pathlib import Path
import json
import os
import re
import sys
import urllib.request
import urllib.error

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES_FILE = OUTPUT_DIR / "window_features.csv"
SHAP_FILE = OUTPUT_DIR / "explain_shap_oof_values.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "explain_shap_feature_importance.csv"

MOMENTS_FILE = OUTPUT_DIR / "timing_moments.csv"

OUTPUT_FILE = OUTPUT_DIR / "ollama_feedback.txt"
FEEDBACK_CSV = OUTPUT_DIR / "ollama_feedback.csv"
META_FILE = OUTPUT_DIR / "ollama_feedback_meta.json"

# The URL can be overridden with the OLLAMA_URL environment variable.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = "gemma3:4b"

# How many timed moments are passed to the model.
MAX_PROMPT_MOMENTS = 5

# Set to a participant folder such as "01", "04", or "17".
# None means prompt for a participant interactively.
PARTICIPANT_ID = None

TOP_FEATURES = 10


# --------------------------------------------------
# OLLAMA REQUEST
# --------------------------------------------------
import numpy as np
import pandas as pd

SENSOR_CHANNELS = [
    "X", "Y", "Z",
    "RX", "RY", "RZ",
    "FX", "FY", "FZ",
    "TX", "TY", "TZ",
]


def detect_candidate_events(df, window_seconds=1.0, threshold=4.0):
    """Flag unusual sensor deviations for review, not confirmed mistakes."""
    time_col = "Elapsed (s)" if "Elapsed (s)" in df.columns else "Time[s]"

    if time_col not in df.columns or df.empty:
        return []

    times = pd.to_numeric(df[time_col], errors="coerce")
    valid_times = times.dropna()

    if len(valid_times) < 5:
        return []

    dt = valid_times.diff().median()
    if not np.isfinite(dt) or dt <= 0:
        return []

    window = max(5, int(round(window_seconds / dt)))
    events = []

    for channel in SENSOR_CHANNELS:
        if channel not in df.columns:
            continue

        values = pd.to_numeric(df[channel], errors="coerce")
        local_median = values.rolling(
            window, center=True, min_periods=max(3, window // 2)
        ).median()

        residual = values - local_median
        residual_median = residual.median()
        mad = (residual - residual_median).abs().median()

        # A zero MAD means this channel cannot be scored reliably
        # using this particular robust threshold.
        if not np.isfinite(mad) or mad <= 0:
            continue

        score = (residual - residual_median).abs() / (1.4826 * mad)
        flagged = score >= threshold

        # Group adjacent flagged samples into candidate events.
        groups = (flagged != flagged.shift(fill_value=False)).cumsum()

        for _, group in df[flagged].assign(_group=groups[flagged]).groupby("_group"):
            indices = group.index
            peak_idx = score.loc[indices].idxmax()

            events.append({
                "channel": channel,
                "start_s": round(float(times.loc[indices].min()), 3),
                "end_s": round(float(times.loc[indices].max()), 3),
                "peak_time_s": round(float(times.loc[peak_idx]), 3),
                "sensor_value": round(float(values.loc[peak_idx]), 5),
                "anomaly_score": round(float(score.loc[peak_idx]), 2),
                "interpretation": "Unusual sensor event; technique error unconfirmed",
            })

    return sorted(
        events,
        key=lambda event: event["anomaly_score"],
        reverse=True,
    )[:10]

def ask_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            # Raised from 220: the answer now also lists timed moments.
            "num_predict": 700,
            "num_ctx": 8192
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Check that Ollama is running "
            f"and that {MODEL_NAME} is installed.\nDetails: {exc}"
        ) from exc


# --------------------------------------------------
# LOAD EXISTING RESULTS
# --------------------------------------------------

def load_files():
    for file_path in [
        FEATURES_FILE,
        SHAP_FILE,
        IMPORTANCE_FILE
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}\n"
                "Run feature_extraction.py and explain.py first."
            )

    features = pd.read_csv(FEATURES_FILE)
    shap_values = pd.read_csv(SHAP_FILE)
    importance = pd.read_csv(IMPORTANCE_FILE)

    required = {
        "participant_folder",
        "predicted_probability_expert",
        "predicted_label"
    }

    if not required.issubset(shap_values.columns):
        raise ValueError(
            "Unexpected columns in explain_shap_oof_values.csv: "
            f"{shap_values.columns.tolist()}"
        )

    return features, shap_values, importance


# --------------------------------------------------
# TIMED MOMENTS (from timing_analysis.py)
# --------------------------------------------------

def load_moments(participant_id):
    """Timed moments for one participant, strongest first. [] if none."""
    if not MOMENTS_FILE.exists() or MOMENTS_FILE.stat().st_size == 0:
        return None
    try:
        moments = pd.read_csv(MOMENTS_FILE)
    except pd.errors.EmptyDataError:
        return []
    if moments.empty:
        return []
    moments["participant_folder"] = (
        moments["participant_folder"].astype(str).str.strip()
        .str.replace(r"\.0$", "", regex=True).str.zfill(2)
    )
    rows = moments[moments["participant_folder"] == participant_id]
    return rows.sort_values("max_abs_z", ascending=False).to_dict(orient="records")


def select_prompt_moments(moments, limit=MAX_PROMPT_MOMENTS):
    """Strongest moments, spread across tasks and behaviours."""
    def focus(moment):
        if moment.get("pattern") == "throughout" and has_distinct_worst(moment):
            return float(moment["worst_start_s"]), float(moment["worst_end_s"])
        if moment.get("pattern") == "throughout":
            peak = float(moment.get("peak_s") or moment["start_s"])
            return peak - 1.0, peak + 1.0
        return float(moment["start_s"]), float(moment["end_s"])

    def overlaps(moment, others):
        a0, a1 = focus(moment)
        return any(
            o["object_type"] == moment["object_type"]
            and a0 < focus(o)[1] and focus(o)[0] < a1
            for o in others
        )

    chosen, per_task, per_metric = [], {}, {}
    for moment in moments:
        task, metric = moment["object_type"], moment["metric"]
        if per_task.get(task, 0) >= 2 or per_metric.get(metric, 0) >= 2:
            continue
        # Prefer moments at different times over facets of the same event.
        if overlaps(moment, chosen):
            continue
        chosen.append(moment)
        per_task[task] = per_task.get(task, 0) + 1
        per_metric[metric] = per_metric.get(metric, 0) + 1
        if len(chosen) == limit:
            break
    # Fill remaining slots if the diversity rule was too strict.
    for moment in moments:
        if len(chosen) == limit:
            break
        if moment not in chosen:
            chosen.append(moment)
    return chosen


def has_distinct_worst(moment):
    """True when the worst stretch is clearly shorter than the whole pattern."""
    try:
        worst = float(moment["worst_end_s"]) - float(moment["worst_start_s"])
        whole = float(moment["end_s"]) - float(moment["start_s"])
    except (KeyError, TypeError, ValueError):
        return False
    return whole > 0 and worst < 0.6 * whole


def moment_line(moment):
    """One plain-language line with exact task and times."""
    peak = moment.get("peak_label")
    peak_text = f" (sharpest at {peak})" if isinstance(peak, str) and peak != "?" else ""
    if moment.get("pattern") == "throughout":
        when = f"throughout most of the recording ({moment['start_label']}-{moment['end_label']})"
        if has_distinct_worst(moment):
            when += f", worst at {moment['worst_start_label']}-{moment['worst_end_label']}"
        description = str(moment["description"]).replace("throughout most of the task, ", "")
    else:
        when = f"{moment['start_label']}-{moment['end_label']}"
        description = moment["description"]
    return (
        f"{moment['task']} task, {when}{peak_text}: {description}. "
        f"Try instead: {moment['tip']}"
    )


def allowed_time_labels(moments):
    labels = set()
    for moment in moments:
        for key in ("start_label", "end_label", "peak_label",
                    "worst_start_label", "worst_end_label"):
            value = moment.get(key)
            if isinstance(value, str) and value != "?":
                labels.add(value)
    return labels


TIME_PATTERN = re.compile(r"\b\d{1,3}:\d{2}(?:\.\d)?\b")


def check_times(feedback, moments):
    """Times written by the model that are not in the supplied moments."""
    allowed = allowed_time_labels(moments)
    allowed_loose = {label.split(".")[0] for label in allowed} | allowed
    found = TIME_PATTERN.findall(feedback)
    return sorted({t for t in found if t not in allowed and t not in allowed_loose})


# --------------------------------------------------
# BUILD PARTICIPANT PROFILE
# --------------------------------------------------

def make_profile(participant_id, features, shap_values, importance):
    # Match folder IDs as strings, preserving leading zeros.
    participant_id = str(participant_id).strip().zfill(2)

    shap_values = shap_values.copy()
    shap_values["participant_folder"] = (
        shap_values["participant_folder"]
        .astype(str)
        .str.strip()
        .str.zfill(2)
    )

    features = features.copy()
    features["participant_folder"] = (
        features["participant_folder"]
        .astype(str)
        .str.strip()
        .str.zfill(2)
    )

    rows = shap_values[
        shap_values["participant_folder"] == participant_id
    ]

    if rows.empty:
        available = sorted(
            shap_values["participant_folder"].unique().tolist()
        )
        raise ValueError(
            f"Participant {participant_id} not found. "
            f"Available participants: {available}"
        )

    prediction = rows.iloc[0]

    # SHAP columns are named SHAP_<feature>.
    shap_columns = [
        col for col in rows.columns
        if col.startswith("SHAP_")
    ]

    shap_records = []

    for col in shap_columns:
        feature_name = col[len("SHAP_"):]
        value = float(rows.iloc[0][col])

        shap_records.append({
            "feature": feature_name,
            "shap_value": value,
            "absolute_shap": abs(value)
        })

    shap_table = pd.DataFrame(shap_records)

    if shap_table.empty:
        raise ValueError("No SHAP feature values were found.")

    shap_table = shap_table.merge(
        importance[["feature", "mean_absolute_shap"]],
        on="feature",
        how="left"
    )

    # Participant's median sensor features.
    participant_windows = features[
        features["participant_folder"] == participant_id
    ]

    if participant_windows.empty:
        raise ValueError(
            f"No sensor windows found for participant {participant_id}."
        )

    numeric_columns = participant_windows.select_dtypes(
        include="number"
    ).columns.tolist()

    exclude = {
        "recording_id",
        "participant_folder",
        "window_id",
        "segment_id"
    }

    numeric_columns = [
        col for col in numeric_columns
        if col not in exclude
    ]

    participant_medians = (
        participant_windows[numeric_columns]
        .median(numeric_only=True)
        .to_dict()
    )

    # Join the actual sensor feature values to the SHAP explanations.
    shap_table["participant_value"] = shap_table["feature"].map(
        participant_medians
    )

    shap_table = shap_table.sort_values(
        "absolute_shap",
        ascending=False
    )

    profile = {
        "participant_id": participant_id,
        "predicted_label": int(prediction["predicted_label"]),
        "predicted_probability_expert": float(
            prediction["predicted_probability_expert"]
        ),
        "true_inferred_label": int(
            prediction["true_inferred_label"]
        ) if "true_inferred_label" in prediction.index else None,
        "top_features": shap_table.head(TOP_FEATURES).to_dict(
            orient="records"
        ),
        "sensor_feature_medians": participant_medians
    }

    all_moments = load_moments(participant_id)
    profile["timing_available"] = all_moments is not None
    profile["all_moments"] = all_moments or []
    profile["moments"] = select_prompt_moments(all_moments or [])

    return profile


# --------------------------------------------------
# GENERATE CAREFUL, ACTIONABLE FEEDBACK
# --------------------------------------------------



def build_prompt(profile):
    features = profile["top_features"]

    feature_lines = []
    for item in features[:6]:
        feature_lines.append(
            f"- {item['feature']}: "
            f"participant value={item['participant_value']}, "
            f"SHAP={item['shap_value']:+.4f}"
        )

    feature_text = "\n".join(feature_lines)

    predicted_class = (
        "inferred-expert class"
        if profile["predicted_label"] == 1
        else "not-expert-across-all-shapes class"
    )

    moments = profile.get("moments", [])
    if moments:
        moment_text = "\n".join(
            f"M{i}. {moment_line(m)}" for i, m in enumerate(moments, start=1)
        )
        timing_rules = """- The TIMED MOMENTS list gives the exact task and time of each moment.
  Times are minutes:seconds from the start of that task's recording.
- Copy times EXACTLY as written (for example 0:12.0-0:16.0). Never invent,
  round or change a time, and never mention a time that is not listed.
- A moment means the sensors differed strongly from experienced
  participants at that time. Say "around that time" or "the sensors show";
  do not claim the person definitely made an error."""
        when_section = """4. WHEN IT HAPPENED
List up to 4 moments from TIMED MOMENTS, most important first. For each
one write one bullet in this form:
- <Task> task, <time range> (sharpest at <peak>): what the sensors show,
  in everyday words, and the one thing to do differently next time.
  For a pattern "throughout most of the recording", say so and give its
  worst stretch and sharpest time.
Use only the listed moments and their exact times."""
    else:
        moment_text = (
            "None. No period stood out strongly from experienced participants."
            if profile.get("timing_available")
            else "Not available (timing_analysis.py has not been run)."
        )
        timing_rules = "- Do not invent times; no timed moments are available."
        when_section = """4. WHEN IT HAPPENED
Say in one sentence that no specific moment stood out strongly from
experienced participants, so the advice applies to the whole session.
Do not invent a time."""

    return f"""
You are a practical coach helping a beginner improve wood shaping,
sanding, and controlled object manipulation.

PARTICIPANT: {profile['participant_id']}
MODEL PREDICTION: {predicted_class}

The prediction is based on inferred labels, not verified expertise.

MOST INFLUENTIAL SENSOR MEASUREMENTS (whole session):
{feature_text}

TIMED MOMENTS (from the participant's own recordings):
{moment_text}

HOW TO INTERPRET THE DATA:
- Negative SHAP means a measurement pushed the model away from the
  inferred-expert class. It does not prove that the measurement is bad.
- Do not assume higher or lower force or torque is always better.
- Do not invent exact values or mistakes.
{timing_rules}
- Only recommend actions relevant to wood shaping and sanding.

Write ONLY these sections. Use simple everyday language.

1. WHERE TO FOCUS
Give up to 3 concise bullets. Name the finding and explain the possible
practical relevance. Where a timed moment supports it, mention the task
and time.

2. SMALL CHANGES TO TRY
Give up to 3 specific actions for the next wood-shaping or sanding trial,
connected to the findings above, for example:
- Apply steady, moderate pressure rather than pressing harder
  whenever progress seems slow.
- Keep the sanding stroke and direction consistent.
- Avoid sudden twisting or changing grip midway through a stroke.
Do not present these examples as confirmed mistakes.

3. HOW TO CHECK PROGRESS
Choose the two most relevant simple checks (even smoothness, fewer deep
scratches or uneven patches, pressure that feels the same from start to
end, repeated trials giving similar results) and say how to record them,
such as rating smoothness from 1 to 5.

{when_section}

Keep the entire response under 300 words. Be direct, supportive and
specific. Do not use unexplained technical terms such as RMS, skewness,
IQR, z-score or SHAP in the advice to the participant.
"""


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("=" * 60)
    print("OLLAMA SENSOR FEEDBACK")
    print("=" * 60)

    features, shap_values, importance = load_files()

    if len(sys.argv) > 1 and sys.argv[1].strip():
        participant_id = sys.argv[1].strip()
    elif PARTICIPANT_ID is None:
        participant_id = input(
            "Enter participant folder (01-21): "
        ).strip()
    else:
        participant_id = PARTICIPANT_ID

    profile = make_profile(
        participant_id,
        features,
        shap_values,
        importance
    )

    print(f"\nGenerating feedback for participant "
          f"{profile['participant_id']}...")
    print(f"Using Ollama model: {MODEL_NAME}")

    prompt = build_prompt(profile)
    feedback = ask_ollama(prompt)

    if not feedback:
        raise RuntimeError("Ollama returned an empty response.")

    # Check that every time the model wrote was one we supplied.
    unknown_times = check_times(feedback, profile["moments"])
    if unknown_times:
        print(
            "\nWARNING: the model wrote times that are not in the timed "
            f"moments list: {unknown_times}"
        )

    from datetime import datetime, timezone
    META_FILE.write_text(json.dumps({
        "participant_id": profile["participant_id"],
        "model": MODEL_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feedback_version": 2,
        "includes_timing": bool(profile["moments"]),
        "timing_available": profile["timing_available"],
        "moments_in_prompt": [
            {k: m.get(k) for k in (
                "task", "object_type", "metric", "metric_label", "direction",
                "start_s", "end_s", "peak_s", "start_label", "end_label",
                "peak_label", "pattern", "worst_start_label", "worst_end_label",
                "description", "tip")}
            for m in profile["moments"]
        ],
        "unknown_times": unknown_times,
    }, indent=2, default=str), encoding="utf-8")

    OUTPUT_FILE.write_text(
        f"Participant: {profile['participant_id']}\n"
        f"Model: {MODEL_NAME}\n\n"
        f"{feedback}\n",
        encoding="utf-8"
    )

    # Save one row per participant run.
    result_row = {
        "participant_folder": profile["participant_id"],
        "predicted_label": profile["predicted_label"],
        "predicted_probability_expert":
            profile["predicted_probability_expert"],
        "model": MODEL_NAME,
        "feedback": feedback
    }

    new_result = pd.DataFrame([result_row])

    if FEEDBACK_CSV.exists():
        old_results = pd.read_csv(FEEDBACK_CSV)
        new_result = pd.concat(
            [old_results, new_result],
            ignore_index=True
        )

    new_result.to_csv(FEEDBACK_CSV, index=False)

    print("\n" + "=" * 60)
    print("GENERATED FEEDBACK")
    print("=" * 60)
    print(feedback)

    print("\nSaved feedback to:")
    print(OUTPUT_FILE)
    print(FEEDBACK_CSV)


if __name__ == "__main__":
    main()