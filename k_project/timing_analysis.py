"""
timing_analysis.py

Answers "WHEN did it happen?" for every participant and every task
(box, cylinder, sphere, wooden_corners).

Method (all steps are deterministic; no LLM is involved):

1. Each recording is loaded with data_loader.py and cleaned / split /
   windowed with preprocessing.py, exactly as feature_extraction.py does
   (2-second windows, 1-second stride, no windows across timestamp gaps).

2. Seven interpretable behaviour metrics are computed per window from the
   raw sensor channels:
       force_level         mean |F|                    (how hard)
       force_variability   SD of |F|                   (how steady)
       force_abruptness    95th pct of |d|F|/dt|       (sudden jabs/slips)
       torque_variability  SD of |T|                   (uneven twisting)
       movement_speed      median tool speed            (rushing/hesitating)
       movement_jerk       95th pct of |d speed/dt|     (jerky motion)
       rotation_rate       median angular speed         (turning the piece)
   |F| = sqrt(FX²+FY²+FZ²), |T| = sqrt(TX²+TY²+TZ²), speed from X/Y/Z,
   angular speed from the RX/RY/RZ/RW quaternion. Rates are computed over
   a ~50 ms lag to reduce sample-to-sample sensor noise.

3. The reference for a participant is every window of the same task from
   participants labelled "Experienced (inferred)" on that task in
   outputs/overall_expertise.csv, always EXCLUDING the participant being
   analysed. If fewer than 3 such participants exist, all other
   participants are used and this is recorded.

4. Each window is scored with a robust z-score against that reference:
   z = (value - reference median) / (1.4826 × reference MAD).
   A window is flagged when |z| >= 3.5 in the direction that matters for
   that metric AND the value differs from the reference median by at least
   25 %.

5. Consecutive flagged windows are merged into one "moment" with a start
   time, an end time and a peak time (the single sample inside the worst
   window where the relevant signal was most extreme). Times are seconds
   since the start of that recording. A moment covering most of the
   recording is labelled "throughout the task" and also gets its worst
   stretch: the windows around the strongest one that score at least half
   as high.

Outputs (outputs/):
    timing_window_metrics.csv   every window, metric values and z-scores
    timing_reference.csv        reference statistics used for each case
    timing_moments.csv          the timed moments used for feedback
    timing_summary.json         settings and counts

IMPORTANT: a moment is a period where the sensor pattern differed
strongly from inferred-experienced participants doing the same task. It
is a strong hint about when a technique problem happened, not a
confirmed mistake.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
LABELS_FILE = OUTPUT_DIR / "overall_expertise.csv"
WINDOW_METRICS_FILE = OUTPUT_DIR / "timing_window_metrics.csv"
REFERENCE_FILE = OUTPUT_DIR / "timing_reference.csv"
MOMENTS_FILE = OUTPUT_DIR / "timing_moments.csv"
SUMMARY_FILE = OUTPUT_DIR / "timing_summary.json"

TIME_COLUMN = "Time[s]"
SHAPES = ["box", "cylinder", "sphere", "wooden_corners"]
SHAPE_LABELS = {
    "box": "Box",
    "cylinder": "Cylinder",
    "sphere": "Sphere",
    "wooden_corners": "Wooden corners",
}
EXPERIENCED = "Experienced (inferred)"

Z_THRESHOLD = 3.5
MIN_RELATIVE_DIFFERENCE = 0.25
MIN_REFERENCE_PARTICIPANTS = 3
RATE_LAG_SECONDS = 0.05
MAX_MOMENTS_PER_RECORDING = 5
# A moment covering at least this share of the recording is reported as a
# pattern "throughout the task", together with its worst stretch.
THROUGHOUT_COVERAGE = 0.6
# The worst stretch is the run of windows around the strongest window whose
# score is at least this fraction of the strongest score.
WORST_STRETCH_FRACTION = 0.5

# ------------------------------------------------------------------
# Metric definitions: plain-language text is used by the UI and the LLM.
# ------------------------------------------------------------------
METRICS = {
    "force_level": dict(
        label="Pressure level", signal="force", stat="mean", direction="both",
        unit="force-sensor units",
        high="pressed much harder than experienced participants on this task",
        low="had much lighter contact than experienced participants (possible loss of contact)",
        tip_high="Ease off and let the abrasive do the work; aim for steady, moderate pressure.",
        tip_low="Keep the tool in steady contact with the wood for the whole stroke.",
    ),
    "force_variability": dict(
        label="Pressure steadiness", signal="force", stat="std", direction="high",
        unit="force-sensor units",
        high="pressure rose and fell much more than for experienced participants (uneven pressure)",
        tip_high="Keep the pressure constant through each stroke; slow down if you feel it changing.",
    ),
    "force_abruptness": dict(
        label="Sudden pressure changes", signal="force_rate", stat="p95abs", direction="high",
        unit="force-sensor units per second",
        high="pressure changed suddenly (a jab, slip or abrupt lift-off)",
        tip_high="Start and finish each stroke gently instead of pushing in or lifting off abruptly.",
    ),
    "torque_variability": dict(
        label="Twisting steadiness", signal="torque", stat="std", direction="high",
        unit="torque-sensor units",
        high="twisting force was much more uneven than for experienced participants",
        tip_high="Keep your grip and wrist angle fixed during a stroke; re-grip between strokes, not during them.",
    ),
    "movement_speed": dict(
        label="Movement speed", signal="speed", stat="median", direction="both",
        unit="position units per second",
        high="moved much faster than experienced participants (rushing)",
        low="moved much more slowly or paused (hesitation)",
        tip_high="Slow down and use full, controlled strokes.",
        tip_low="Keep a steady rhythm; plan the next stroke before you start it.",
    ),
    "movement_jerk": dict(
        label="Smoothness of motion", signal="accel", stat="p95abs", direction="high",
        unit="position units per second²",
        high="movement was jerky, with sudden starts and stops",
        tip_high="Move in smooth, continuous strokes; avoid sudden starts and stops.",
    ),
    "rotation_rate": dict(
        label="Turning the piece", signal="angular_speed", stat="median", direction="high",
        unit="radians per second",
        high="the piece or tool was turned much more than by experienced participants",
        tip_high="Hold the piece steady during a stroke and turn it deliberately between strokes.",
    ),
}


def fmt_time(seconds: float) -> str:
    """Seconds -> 'm:ss.s' (e.g. 0:12.4, 1:05.0)."""
    if seconds is None or not np.isfinite(seconds):
        return "?"
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds - 60 * minutes:04.1f}"


# ------------------------------------------------------------------
# Signals
# ------------------------------------------------------------------
def compute_signals(segment: pd.DataFrame) -> pd.DataFrame:
    """Derived per-sample signals for one continuous segment."""
    t = segment[TIME_COLUMN].to_numpy(dtype=float)
    n = len(t)
    out = pd.DataFrame({"t": t})

    def magnitude(cols):
        if not all(c in segment.columns for c in cols):
            return np.full(n, np.nan)
        arr = segment[cols].to_numpy(dtype=float)
        return np.sqrt(np.sum(arr ** 2, axis=1))

    out["force"] = magnitude(["FX", "FY", "FZ"])
    out["torque"] = magnitude(["TX", "TY", "TZ"])

    dt_all = np.diff(t)
    positive = dt_all[dt_all > 0]
    median_dt = float(np.median(positive)) if len(positive) else np.nan
    lag = 1
    if np.isfinite(median_dt) and median_dt > 0:
        lag = max(1, int(round(RATE_LAG_SECONDS / median_dt)))

    def lagged_rate(values):
        rate = np.full(n, np.nan)
        if n > lag:
            dt = t[lag:] - t[:-lag]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (values[lag:] - values[:-lag]) / dt
            r[~np.isfinite(r)] = np.nan
            rate[lag:] = r
        return rate

    out["force_rate"] = lagged_rate(out["force"].to_numpy())

    if all(c in segment.columns for c in ["X", "Y", "Z"]):
        pos = segment[["X", "Y", "Z"]].to_numpy(dtype=float)
        speed = np.full(n, np.nan)
        if n > lag:
            dt = t[lag:] - t[:-lag]
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.linalg.norm(pos[lag:] - pos[:-lag], axis=1) / dt
            s[~np.isfinite(s)] = np.nan
            speed[lag:] = s
        out["speed"] = speed
        out["accel"] = lagged_rate(speed)
    else:
        out["speed"] = np.nan
        out["accel"] = np.nan

    out["angular_speed"] = np.nan
    if all(c in segment.columns for c in ["RX", "RY", "RZ", "RW"]):
        q = segment[["RX", "RY", "RZ", "RW"]].to_numpy(dtype=float)
        norms = np.linalg.norm(q, axis=1)
        # Only interpret as a rotation if these really are unit quaternions.
        if np.nanmedian(norms) > 0.9 and np.nanmedian(norms) < 1.1 and n > lag:
            qn = q / norms[:, None]
            dot = np.abs(np.sum(qn[lag:] * qn[:-lag], axis=1))
            angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
            dt = t[lag:] - t[:-lag]
            with np.errstate(divide="ignore", invalid="ignore"):
                w = angle / dt
            w[~np.isfinite(w)] = np.nan
            ang = np.full(n, np.nan)
            ang[lag:] = w
            out["angular_speed"] = ang
    return out


def window_statistic(values: np.ndarray, stat: str) -> float:
    v = values[np.isfinite(values)]
    if v.size < 3:
        return np.nan
    if stat == "mean":
        return float(np.mean(v))
    if stat == "std":
        return float(np.std(v))
    if stat == "median":
        return float(np.median(v))
    if stat == "p95abs":
        return float(np.percentile(np.abs(v), 95))
    raise ValueError(stat)


def peak_index(values: np.ndarray, stat: str, sign: int) -> int | None:
    """Sample (within the window) where the relevant signal was most extreme."""
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).any():
        return None
    if stat == "std":
        target = np.abs(v - np.nanmedian(v))
    elif stat == "p95abs":
        target = np.abs(v)
    else:  # mean / median: highest for "too high", lowest for "too low"
        target = v if sign > 0 else -v
    return int(np.nanargmax(target))


# ------------------------------------------------------------------
# Per-recording window metrics
# ------------------------------------------------------------------
def analyse_recording(recording_id: str, df: pd.DataFrame):
    """Return (window rows, per-window signal slices) for one recording."""
    from preprocessing import preprocess_recording

    result = preprocess_recording(df)
    cleaned = result["cleaned_data"]
    t0 = float(cleaned[TIME_COLUMN].iloc[0])
    participant = str(cleaned["participant_folder"].iloc[0]).zfill(2)
    shape = str(cleaned["object_type"].iloc[0])

    rows, slices = [], {}
    signals_by_segment = {
        seg_id: compute_signals(segment)
        for seg_id, segment in enumerate(result["segments"])
    }
    for window in result["windows"]:
        seg_id = window.get("segment_id", 0)
        sig = signals_by_segment[seg_id]
        start = window["start_sample"]
        part = sig.iloc[start:start + window["n_samples"]]
        row = {
            "participant_folder": participant,
            "object_type": shape,
            "recording_id": recording_id,
            "segment_id": seg_id,
            "window_id": window["window_id"],
            "start_time_raw": window["start_time"],
            "end_time_raw": window["end_time"],
            "start_s": window["start_time"] - t0,
            "end_s": window["end_time"] - t0,
            "recording_t0": t0,
        }
        for key, spec in METRICS.items():
            row[key] = window_statistic(part[spec["signal"]].to_numpy(), spec["stat"])
        rows.append(row)
        slices[(seg_id, window["window_id"])] = part
    duration = float(cleaned[TIME_COLUMN].iloc[-1] - t0)
    return rows, slices, duration


# ------------------------------------------------------------------
# Reference and scoring
# ------------------------------------------------------------------
def robust_reference(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size < 10:
        return {"median": np.nan, "scale": np.nan, "q25": np.nan, "q75": np.nan, "n": int(v.size)}
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    q25, q75 = np.percentile(v, [25, 75])
    scale = 1.4826 * mad
    if scale <= 0:
        scale = (q75 - q25) / 1.349
    if scale <= 0:
        scale = float(np.std(v))
    return {"median": med, "scale": scale if scale > 0 else np.nan,
            "q25": float(q25), "q75": float(q75), "n": int(v.size)}


def load_shape_status() -> pd.DataFrame | None:
    if not LABELS_FILE.exists():
        return None
    labels = pd.read_csv(LABELS_FILE)
    labels["participant_folder"] = (
        labels["participant_folder"].astype(str).str.strip()
        .str.replace(r"\.0$", "", regex=True).str.zfill(2)
    )
    return labels


def describe_moment(metric: str, sign: int, ratio: float) -> str:
    spec = METRICS[metric]
    text = spec["high"] if sign > 0 else spec.get("low", spec["high"])
    if np.isfinite(ratio) and ratio > 0:
        if sign > 0:
            text += f" — about {ratio:.1f}× the typical experienced value"
        else:
            text += f" — about {ratio * 100:.0f}% of the typical experienced value"
    return text


def tip_for(metric: str, sign: int) -> str:
    spec = METRICS[metric]
    return spec["tip_high"] if sign > 0 else spec.get("tip_low", spec["tip_high"])


def main():
    from data_loader import load_all_recordings

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("TIMING ANALYSIS — WHEN DID DEVIATIONS HAPPEN?")
    print("=" * 65)

    recordings = load_all_recordings()

    all_rows, slices, durations = [], {}, {}
    for recording_id, df in recordings.items():
        try:
            rows, rec_slices, duration = analyse_recording(recording_id, df)
        except Exception as exc:  # keep going; report the recording
            print(f"WARNING: {recording_id} skipped: {exc}")
            continue
        all_rows.extend(rows)
        durations[recording_id] = duration
        for key, part in rec_slices.items():
            slices[(recording_id,) + key] = part
    if not all_rows:
        raise RuntimeError("No windows could be analysed.")

    windows = pd.DataFrame(all_rows)
    labels = load_shape_status()

    # ---- Scoring against leave-one-out experienced reference -------------
    ref_rows = []
    for metric in METRICS:
        windows[f"z_{metric}"] = np.nan

    for (pid, shape), idx in windows.groupby(["participant_folder", "object_type"]).groups.items():
        others = windows[(windows["object_type"] == shape) & (windows["participant_folder"] != pid)]
        group = "all other participants"
        if labels is not None and f"{shape}_status" in labels.columns:
            experienced_ids = set(
                labels.loc[labels[f"{shape}_status"] == EXPERIENCED, "participant_folder"]
            ) - {pid}
            if len(experienced_ids) >= MIN_REFERENCE_PARTICIPANTS:
                others = others[others["participant_folder"].isin(experienced_ids)]
                group = "inferred-experienced participants (same task)"
        n_ref_participants = others["participant_folder"].nunique()

        for metric in METRICS:
            ref = robust_reference(others[metric].to_numpy(dtype=float))
            ref_rows.append({
                "participant_folder": pid, "object_type": shape, "metric": metric,
                "reference_group": group, "reference_participants": n_ref_participants,
                "reference_windows": ref["n"], "reference_median": ref["median"],
                "reference_scale": ref["scale"], "reference_q25": ref["q25"],
                "reference_q75": ref["q75"],
            })
            if np.isfinite(ref["scale"]) and ref["scale"] > 0:
                windows.loc[idx, f"z_{metric}"] = (
                    (windows.loc[idx, metric] - ref["median"]) / ref["scale"]
                )

    reference = pd.DataFrame(ref_rows)
    ref_lookup = reference.set_index(["participant_folder", "object_type", "metric"])

    # ---- Flag windows and merge into moments -----------------------------
    moments = []
    for recording_id in windows["recording_id"].unique():
        rec =windows[windows["recording_id"] == recording_id].sort_values(["segment_id", "start_s"])
        pid, shape = rec["participant_folder"].iloc[0], rec["object_type"].iloc[0]
        rec_moments = []
        for metric, spec in METRICS.items():
            ref = ref_lookup.loc[(pid, shape, metric)]
            med = ref["reference_median"]
            z = rec[f"z_{metric}"].to_numpy(dtype=float)
            vals = rec[metric].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(vals / med - 1.0) if np.isfinite(med) and med != 0 else np.full(len(vals), np.inf)
            for sign in (+1, -1):
                if spec["direction"] == "high" and sign < 0:
                    continue
                flagged = (sign * z >= Z_THRESHOLD) & (rel >= MIN_RELATIVE_DIFFERENCE) & np.isfinite(z)
                if not flagged.any():
                    continue
                # Merge consecutive / overlapping flagged windows.
                episode = []
                rows_list = list(rec.itertuples(index=False))
                for i, row in enumerate(rows_list):
                    if flagged[i] and (
                        not episode
                        or (row.segment_id == episode[-1][1].segment_id
                            and row.start_s <= episode[-1][1].end_s + 1e-6)
                    ):
                        episode.append((i, row))
                        continue
                    if episode:
                        rec_moments.append(_make_moment(episode, z, metric, sign, ref, slices, recording_id, durations[recording_id]))
                        episode = []
                    if flagged[i]:
                        episode = [(i, row)]
                if episode:
                    rec_moments.append(_make_moment(episode, z, metric, sign, ref, slices, recording_id, durations[recording_id]))
        rec_moments.sort(key=lambda m: -m["max_abs_z"])
        for rank, m in enumerate(rec_moments[:MAX_MOMENTS_PER_RECORDING], start=1):
            m["rank_in_recording"] = rank
            moments.append(m)

    moments_df = pd.DataFrame(moments)
    if not moments_df.empty:
        moments_df = moments_df.sort_values(["participant_folder", "max_abs_z"], ascending=[True, False])
        moments_df["rank_in_participant"] = moments_df.groupby("participant_folder").cumcount() + 1
        moments_df["recording_duration_s"] = moments_df["recording_id"].map(durations)

    windows["recording_duration_s"] = windows["recording_id"].map(durations)
    windows.to_csv(WINDOW_METRICS_FILE, index=False)
    reference.to_csv(REFERENCE_FILE, index=False)
    moments_df.to_csv(MOMENTS_FILE, index=False)

    summary = {
        "z_threshold": Z_THRESHOLD,
        "min_relative_difference": MIN_RELATIVE_DIFFERENCE,
        "min_reference_participants": MIN_REFERENCE_PARTICIPANTS,
        "rate_lag_seconds": RATE_LAG_SECONDS,
        "max_moments_per_recording": MAX_MOMENTS_PER_RECORDING,
        "labels_file_used": LABELS_FILE.exists(),
        "recordings_analysed": int(windows["recording_id"].nunique()),
        "windows": int(len(windows)),
        "moments": int(len(moments_df)),
        "metrics": {k: {kk: v for kk, v in spec.items()} for k, spec in METRICS.items()},
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWindows analysed: {len(windows):,}")
    print(f"Timed moments found: {len(moments_df):,}")
    if not moments_df.empty:
        print("\nMoments per participant:")
        print(moments_df.groupby("participant_folder").size().to_string())
    print(f"\nSaved:\n  {WINDOW_METRICS_FILE}\n  {REFERENCE_FILE}\n  {MOMENTS_FILE}\n  {SUMMARY_FILE}")
    print(
        "\nIMPORTANT: moments are periods where sensor patterns differed strongly "
        "from inferred-experienced participants on the same task. They indicate "
        "when to look, not confirmed mistakes."
    )


def _make_moment(episode, z, metric, sign, ref, slices, recording_id, duration):
    first, last = episode[0][1], episode[-1][1]
    worst_pos, (worst_i, worst_row) = max(enumerate(episode), key=lambda item: sign * z[item[1][0]])

    # Worst stretch: neighbouring windows scoring >= half the strongest score.
    cutoff = max(Z_THRESHOLD, WORST_STRETCH_FRACTION * sign * z[worst_i])
    lo = hi = worst_pos
    while lo > 0 and sign * z[episode[lo - 1][0]] >= cutoff:
        lo -= 1
    while hi < len(episode) - 1 and sign * z[episode[hi + 1][0]] >= cutoff:
        hi += 1
    worst_start, worst_end = float(episode[lo][1].start_s), float(episode[hi][1].end_s)
    span = float(last.end_s - first.start_s)
    coverage = span / duration if duration and duration > 0 else np.nan
    throughout = bool(np.isfinite(coverage) and coverage >= THROUGHOUT_COVERAGE)
    spec = METRICS[metric]
    part = slices.get((recording_id, worst_row.segment_id, worst_row.window_id))
    peak_s, peak_raw = np.nan, np.nan
    if part is not None:
        k = peak_index(part[spec["signal"]].to_numpy(), spec["stat"], sign)
        if k is not None:
            peak_raw = float(part["t"].iloc[k])
            peak_s = peak_raw - worst_row.recording_t0
    value = float(getattr(worst_row, metric))
    med = float(ref["reference_median"])
    ratio = value / med if np.isfinite(med) and med != 0 else np.nan
    return {
        "participant_folder": worst_row.participant_folder,
        "object_type": worst_row.object_type,
        "task": SHAPE_LABELS.get(worst_row.object_type, worst_row.object_type),
        "recording_id": recording_id,
        "metric": metric,
        "metric_label": spec["label"],
        "direction": "higher" if sign > 0 else "lower",
        "start_s": round(float(first.start_s), 2),
        "end_s": round(float(last.end_s), 2),
        "peak_s": round(float(peak_s), 2) if np.isfinite(peak_s) else np.nan,
        "start_time_raw": float(first.start_time_raw),
        "end_time_raw": float(last.end_time_raw),
        "peak_time_raw": peak_raw,
        "start_label": fmt_time(first.start_s),
        "end_label": fmt_time(last.end_s),
        "peak_label": fmt_time(peak_s),
        "value": value,
        "reference_median": med,
        "ratio_to_reference": ratio,
        "max_abs_z": float(abs(z[worst_i])),
        "n_windows": len(episode),
        "pattern": "throughout" if throughout else "episode",
        "coverage": round(coverage, 3) if np.isfinite(coverage) else np.nan,
        "worst_start_s": round(worst_start, 2),
        "worst_end_s": round(worst_end, 2),
        "worst_start_label": fmt_time(worst_start),
        "worst_end_label": fmt_time(worst_end),
        "description": (
            "throughout most of the task, " if throughout else ""
        ) + describe_moment(metric, sign, ratio),
        "tip": tip_for(metric, sign),
        "reference_group": ref["reference_group"],
    }


if __name__ == "__main__":
    main()
