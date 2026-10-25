"""Step 2 - Mendeley EMG+IMU adapter.

Turns the per-subject CSV folders into the engine's common tables:

    out/signals.parquet   one row per 200 Hz sample: subject, exercise, rep, phase,
                          raw EMG, EMG envelope and quaternion (w,x,y,z) per sensor
    out/labels.csv        one row per subject x exercise: every yes/no rule (1 = done
                          correctly, 0 = error), overall score, error count, is_expert
    out/inventory.csv     what was loaded for each subject x exercise, and why not
    out/rule_catalog.csv  every rule with whether the sensors can observe it

Usage (from the dataset root):
    python -m skilltwin.adapters.mendeley --root .
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from skilltwin.config import (EXERCISES, EXPERT_MIN_SCORE, FILE_ALIASES, FS, OBSERVABILITY,
                              OUT_DIR, QUAT, REFERENCE_FILES, SENSORS)

PHASE_WRAP = -50        # phase drop that marks the start of a new repetition
MIN_CYCLE_SWEEP = 80    # a complete repetition sweeps at least this much of the 0-100 phase


def norm(name):
    """Compare names ignoring case, spaces and the .csv suffix."""
    return re.sub(r"\s+", "", name.lower().removesuffix(".csv"))


_CANONICAL = {norm(k): k for k in EXERCISES}
_CANONICAL.update(FILE_ALIASES)
_CANONICAL.update(REFERENCE_FILES)


def canonical(filename):
    """Map a CSV file name to its canonical exercise name, or None if unused."""
    return _CANONICAL.get(norm(filename))


def list_subjects(root):
    return sorted((int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()))


# ---------------------------------------------------------------------------
# Labels

def read_label_file(path):
    """Return (rules dict, overall score, list of problems) for one ExerciseQuality file."""
    df = pd.read_csv(path)
    problems = []
    if len(df) != 1:
        problems.append(f"{len(df)} rows instead of 1")
    row = df.iloc[0]
    overall = row.pop("overall quality")
    rules = {k: v for k, v in row.items()}
    bad = [k for k, v in rules.items() if v not in (0, 1)]
    if bad:
        problems.append(f"non-binary rules {bad}")
    if not 0 <= overall <= 10:
        problems.append(f"overall {overall} outside 0-10")
    return rules, overall, problems


def load_labels(root):
    rows, problems = [], []
    for s in list_subjects(root):
        for path in sorted((root / str(s) / "ExerciseQuality").glob("*.csv")):
            ex = canonical(path.name)
            if ex not in EXERCISES:
                problems.append(f"{s}/{path.name}: unknown exercise name")
                continue
            rules, overall, issues = read_label_file(path)
            problems += [f"{s}/{path.name}: {msg}" for msg in issues]
            family, mode, side = EXERCISES[ex]
            n_errors = sum(1 for v in rules.values() if v == 0)
            rows.append({
                "subject": s, "exercise": ex, "family": family, "mode": mode, "side": side,
                "overall": overall, "n_rules": len(rules), "n_errors": n_errors,
                "is_expert": bool(overall >= EXPERT_MIN_SCORE and n_errors == 0),
                **rules,
            })
    labels = pd.DataFrame(rows)
    meta = ["subject", "exercise", "family", "mode", "side", "overall", "n_rules",
            "n_errors", "is_expert"]
    rule_cols = sorted(c for c in labels.columns if c not in meta)
    labels[rule_cols] = labels[rule_cols].astype("Int8")
    return labels[meta + rule_cols], problems


def rule_catalog(labels):
    """One row per (exercise, rule) present in the labels, with observability."""
    meta = {"subject", "exercise", "family", "mode", "side", "overall", "n_rules",
            "n_errors", "is_expert"}
    rows = []
    for ex, grp in labels.groupby("exercise"):
        for rule in (c for c in labels.columns if c not in meta):
            vals = grp[rule].dropna()
            if vals.empty:
                continue
            level, how = OBSERVABILITY.get(rule, ("unknown", "not in OBSERVABILITY"))
            rows.append({"exercise": ex, "family": EXERCISES[ex][0], "rule": rule,
                         "observable": level, "how": how, "n_sets": len(vals),
                         "n_errors": int((vals == 0).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signals

def _column_map():
    cols = {"Sample": "t_abs", "Exercise_Phase": "phase"}
    for key, (prefix, _) in SENSORS.items():
        cols[f"{prefix}_EMG"] = f"emg_{key}"
        cols[f"{prefix}_EMG_Envelope"] = f"env_{key}"
        for i, comp in enumerate(QUAT, start=1):
            cols[f"{prefix}_Q{i}"] = f"q_{key}_{comp}"
    return cols


COLUMN_MAP = _column_map()
SIGNAL_COLS = [c for c in COLUMN_MAP.values() if c not in ("t_abs", "phase")]


def assign_reps(phase):
    """Repetition index per sample from the 0-100 phase; -1 for incomplete pieces.

    Each file ends with a short stub after the last phase restart, so only pieces
    that sweep most of the cycle count as repetitions.
    """
    phase = np.nan_to_num(phase, nan=0.0)
    starts = np.r_[0, np.where(np.diff(phase) < PHASE_WRAP)[0] + 1]
    ends = np.r_[starts[1:], len(phase)]
    rep = np.full(len(phase), -1, dtype=np.int16)
    k = 0
    for a, b in zip(starts, ends):
        if np.ptp(phase[a:b]) >= MIN_CYCLE_SWEEP:
            rep[a:b] = k
            k += 1
    return rep


def read_segment(path):
    """Read one Segmented CSV into engine columns; None if the file has no data."""
    df = pd.read_csv(path)
    if df.empty:
        return None
    missing = [c for c in COLUMN_MAP if c not in df.columns and c != "Exercise_Phase"]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.rename(columns=COLUMN_MAP)
    out = pd.DataFrame({"t_abs": df["t_abs"].to_numpy(np.float64)})
    out["t"] = (out["t_abs"] - out["t_abs"].iloc[0]).astype(np.float32)
    if "phase" in df:
        out["phase"] = df["phase"].to_numpy(np.float32)
        out["rep"] = assign_reps(out["phase"].to_numpy())
    else:
        out["phase"] = np.float32(np.nan)
        out["rep"] = np.int16(-1)
    for c in SIGNAL_COLS:
        out[c] = df[c].to_numpy(np.float32)
    return out


def _segment_files(subject_dir, kind):
    """canonical name -> list of file paths in Segmented/<kind>/."""
    found = {}
    folder = subject_dir / "Segmented" / kind
    if folder.is_dir():
        for path in sorted(folder.glob("*.csv")):
            ex = canonical(path.name)
            if ex:
                found.setdefault(ex, []).append(path)
    return found


def load_signals(root, labels):
    """Load every labelled exercise (plus reference files) for every subject.

    Prefers Segmented/Repetitive (has the phase); falls back to NonRepetitive when
    the repetitive file is missing or header-only.
    """
    labelled = set(zip(labels["subject"], labels["exercise"]))
    frames, inventory = [], []
    for s in list_subjects(root):
        sdir = root / str(s)
        rep_files = _segment_files(sdir, "Repetitive")
        non_files = _segment_files(sdir, "NonRepetitive")
        wanted = sorted({ex for (subj, ex) in labelled if subj == s}
                        | {ex for ex in non_files if ex.startswith("ref:")})
        for ex in wanted:
            candidates = ([("repetitive", p) for p in rep_files.get(ex, [])]
                          + [("nonrepetitive", p) for p in non_files.get(ex, [])])
            record = {"subject": s, "exercise": ex, "labelled": (s, ex) in labelled,
                      "source": None, "file": None, "n_rows": 0, "duration_s": 0.0,
                      "n_reps": 0, "imu_missing": None, "phase_missing": None,
                      "status": "missing", "note": ""}
            skipped = []
            for source, path in candidates:
                seg = read_segment(path)
                if seg is None:
                    skipped.append(f"{source}/{path.name} empty")
                    continue
                n_reps = int(seg["rep"].max()) + 1
                q_cols = [c for c in seg.columns if c.startswith("q_")]
                record.update(source=source, file=str(path.relative_to(root)),
                              n_rows=len(seg), duration_s=round(float(seg["t"].iloc[-1]), 2),
                              n_reps=n_reps,
                              imu_missing=round(float(seg[q_cols].isna().any(axis=1).mean()), 3),
                              phase_missing=bool(source == "repetitive"
                                                 and seg["phase"].isna().all()))
                dynamic = not ex.startswith("ref:") and EXERCISES[ex][1] == "dynamic"
                record["status"] = ("fallback_no_phase"
                                    if dynamic and source == "nonrepetitive" and rep_files.get(ex)
                                    else "ok")
                seg.insert(0, "source", source)
                seg.insert(0, "exercise", ex)
                seg.insert(0, "subject", np.int16(s))
                frames.append(seg)
                break
            if skipped:
                record["note"] = "; ".join(skipped)
                if record["status"] == "missing":
                    record["status"] = "empty"
            inventory.append(record)
    signals = pd.concat(frames, ignore_index=True)
    signals["exercise"] = signals["exercise"].astype("category")
    signals["source"] = signals["source"].astype("category")
    return signals, pd.DataFrame(inventory)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="dataset root (folder with 1/, 2/, ...)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels, problems = load_labels(root)
    catalog = rule_catalog(labels)
    signals, inventory = load_signals(root, labels)

    labels.to_csv(out / "labels.csv", index=False)
    catalog.to_csv(out / "rule_catalog.csv", index=False)
    inventory.to_csv(out / "inventory.csv", index=False)
    signals.to_parquet(out / "signals.parquet", index=False)

    print(f"subjects: {list_subjects(root)}")
    print(f"labels: {len(labels)} sets, {int(labels['is_expert'].sum())} expert")
    print(f"label problems: {problems or 'none'}")
    print(f"signals: {len(signals):,} rows x {signals.shape[1]} cols "
          f"({signals.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory), {FS} Hz")
    print("\ninventory status:")
    print(inventory.groupby(["status"]).size().to_string())
    no_imu = inventory[inventory["imu_missing"] > 0]
    print(f"\nsets with missing quaternions (EMG still usable): {len(no_imu)}")
    print(no_imu.groupby("subject")["exercise"].apply(len).to_string())
    print(f"repetitive files with no phase: "
          f"{inventory.loc[inventory['phase_missing'] == True, ['subject', 'exercise']].values.tolist()}")
    print("\nnot ok:")
    print(inventory.loc[inventory["status"] != "ok",
                        ["subject", "exercise", "status", "note"]].to_string(index=False))
    dyn = (inventory[inventory["status"] == "ok"]
           .merge(labels[["subject", "exercise", "mode", "is_expert"]], on=["subject", "exercise"])
           .query("mode == 'dynamic'"))
    dyn["expert_reps"] = dyn["n_reps"].where(dyn["is_expert"], 0)
    per_ex = dyn.groupby("exercise").agg(
        sets=("n_reps", "size"), sets_with_reps=("n_reps", lambda r: int((r > 0).sum())),
        reps=("n_reps", "sum"), expert_sets=("is_expert", "sum"), expert_reps=("expert_reps", "sum"))
    print("\ndynamic exercises with signal:")
    print(per_ex.astype(int).to_string())
    print("\nrule observability (sets per level):")
    print(catalog.groupby("observable")[["n_sets", "n_errors"]].sum().to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
