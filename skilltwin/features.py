"""Steps 4 + 5 - Feature curves, repetitions, per-repetition and per-set summaries.

    out/clean.parquet -> out/reps.csv         one row per repetition
                         out/curves.npz       each repetition resampled to 0-100 %
                         out/set_features.csv one row per set of a repetitive exercise
                         out/holds.csv        one row per set of a hold exercise

Angles are rotations away from the start-of-repetition pose (phase ~0), so they do
not depend on how each sensor was strapped on (mounting is undocumented):
    knee   - thigh-vs-shank relative rotation (knee bend)
    thigh, shank, uarm - rotation of each segment

Usage: python -m skilltwin.features
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from skilltwin import quat
from skilltwin.config import EXERCISES, OUT_DIR, QUAT

FS = 100
N_POINTS = 101                     # 0..100 % of a repetition
REF_PHASE = 3                      # samples with phase below this define the start pose
VALID_DUR = (0.4, 2.5)             # rep duration as a multiple of its set median
SEGMENTS = {"thigh": "rectus", "shank": "gastroc", "uarm": "biceps"}
MUSCLES = ["biceps", "triceps", "rectus", "gastroc"]
CURVES = ["knee", "thigh", "shank", "uarm"] + [f"env_{m}" for m in MUSCLES]
# The angle that defines "depth" for each family.
PRIMARY = {"squat": "knee", "lunge": "knee", "burpee": "knee", "pushup": "uarm"}


def _lowpass(x, hz, order=2):
    b, a = butter(order, hz / (FS / 2))
    return filtfilt(b, a, x, axis=0) if len(x) > 3 * max(len(a), len(b)) else x


def _bandpass(x, lo, hi):
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x, axis=0) if len(x) > 3 * max(len(a), len(b)) else x * 0


def segment_quats(d):
    q = {s: d[[f"q_{m}_{c}" for c in QUAT]].to_numpy(np.float64) for s, m in SEGMENTS.items()}
    q["knee"] = quat.mul(quat.conj(q["thigh"]), q["shank"])
    return q


def angle_curves(d, ref_mask):
    """Per-sample rotation (deg) of knee and each segment away from the reference pose."""
    q = segment_quats(d)
    return {name: quat.angle_between(quat.mean(qq[ref_mask]), qq) for name, qq in q.items()}, q


def ldlj(velocity, fs=FS):
    """Log dimensionless jerk of a velocity trace (n x d), higher = smoother."""
    speed = np.linalg.norm(velocity, axis=1)
    if len(velocity) < 10 or speed.max() < 1e-6:
        return np.nan
    jerk = np.gradient(np.gradient(velocity, 1 / fs, axis=0), 1 / fs, axis=0)
    dur = len(velocity) / fs
    return -np.log(dur ** 3 / speed.max() ** 2 * np.sum(jerk ** 2) / fs)


def coactivation(a, b):
    return float(np.sum(np.minimum(a, b)) / max(np.sum(np.maximum(a, b)), 1e-9))


def resample(x, n=N_POINTS):
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


# ---------------------------------------------------------------------------
# Repetitive exercises

def process_reps(d, subject, exercise):
    family = EXERCISES[exercise][0]
    ref = (d["phase"].to_numpy() < REF_PHASE) & (d["rep"].to_numpy() >= 0)
    if ref.sum() < 5:
        ref = np.zeros(len(d), bool)
        ref[:50] = True
    ang, q = angle_curves(d, ref)
    omega = {s: _lowpass(quat.angular_velocity(q[s], FS), 6) for s in SEGMENTS}
    env = {m: d[f"env_{m}"].to_numpy(np.float64) for m in MUSCLES}
    primary = PRIMARY.get(family, "knee")

    rows, curves = [], []
    rep = d["rep"].to_numpy()
    for r in np.unique(rep[rep >= 0]):
        idx = np.where(rep == r)[0]
        row = {"subject": subject, "exercise": exercise, "family": family, "rep": int(r),
               "dur": len(idx) / FS}
        for a in ("knee", "thigh", "shank", "uarm"):
            row[f"{a}_max"] = ang[a][idx].max()
            row[f"{a}_rom"] = np.ptp(ang[a][idx])
        row["t_bottom"] = 100 * np.argmax(ang[primary][idx]) / max(len(idx) - 1, 1)
        for s in SEGMENTS:
            row[f"speed_{s}"] = np.degrees(np.linalg.norm(omega[s][idx], axis=1).max())
            row[f"ldlj_{s}"] = ldlj(omega[s][idx])
        for m in MUSCLES:
            e = env[m][idx]
            row[f"env_peak_{m}"] = e.max()
            row[f"env_mean_{m}"] = e.mean()
            row[f"env_tpeak_{m}"] = 100 * np.argmax(e) / max(len(idx) - 1, 1)
        row["coact_leg"] = coactivation(env["rectus"][idx], env["gastroc"][idx])
        row["coact_arm"] = coactivation(env["biceps"][idx], env["triceps"][idx])
        rows.append(row)
        curves.append(np.stack([resample(ang[c][idx]) if c in ang else resample(env[c[4:]][idx])
                                for c in CURVES]))
    reps = pd.DataFrame(rows)
    if len(reps):
        med = reps["dur"].median()
        reps["valid"] = reps["dur"].between(VALID_DUR[0] * med, VALID_DUR[1] * med)
    return reps, curves


SLOPE_FEATURES = ["dur", "knee_max", "thigh_max", "uarm_max", "speed_thigh", "speed_uarm",
                  "ldlj_thigh", "ldlj_shank", "ldlj_uarm"] + [f"env_mean_{m}" for m in MUSCLES]
CV_FEATURES = ["dur", "knee_max", "thigh_max", "uarm_max", "speed_thigh", "speed_uarm"]


def aggregate_set(reps):
    """Per-set summary of valid reps: means, consistency (CV) and first-to-last trend."""
    r = reps[reps["valid"]]
    out = {"n_reps": len(r)}
    numeric = [c for c in r.columns
               if c not in ("subject", "exercise", "family", "rep", "valid")]
    for c in numeric:
        out[f"{c}_mean"] = r[c].mean()
    for c in CV_FEATURES:
        out[f"{c}_cv"] = r[c].std() / max(abs(r[c].mean()), 1e-9)
    x = np.arange(len(r))
    for c in SLOPE_FEATURES:
        y = r[c].to_numpy()
        ok = np.isfinite(y)
        if ok.sum() >= 3:
            slope = np.polyfit(x[ok], y[ok], 1)[0]
            # relative change from first to last rep implied by the trend
            out[f"{c}_trend"] = slope * (len(r) - 1) / max(abs(np.nanmean(y)), 1e-9)
        else:
            out[f"{c}_trend"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Holds

def process_hold(d, subject, exercise):
    n = len(d)
    head, tail = slice(0, n // 5), slice(n - n // 5, n)
    q = segment_quats(d)
    row = {"subject": subject, "exercise": exercise, "family": EXERCISES[exercise][0],
           "dur": n / FS}
    for s in SEGMENTS:
        qs = q[s]
        slow = quat.normalize(_lowpass(qs, 1.0))
        row[f"sway_{s}"] = np.sqrt(np.mean(quat.angle_between(quat.mean(slow), slow) ** 2))
        row[f"drift_{s}"] = float(quat.angle_between(quat.mean(qs[head]), quat.mean(qs[tail])))
        omega = quat.angular_velocity(qs, FS)
        trem = _bandpass(omega, 4, 12)
        row[f"tremor_{s}"] = np.degrees(np.sqrt(np.mean(np.sum(trem ** 2, axis=1))))
    for m in MUSCLES:
        e = d[f"env_{m}"].to_numpy(np.float64)
        row[f"env_mean_{m}"] = e.mean()
        row[f"env_cv_{m}"] = e.std() / max(e.mean(), 1e-9)
        row[f"env_fatigue_{m}"] = (e[tail].mean() - e[head].mean()) / max(e.mean(), 1e-9)
    return row


# ---------------------------------------------------------------------------

def build(clean):
    rep_tables, curve_list, set_rows, hold_rows = [], [], [], []
    for (s, ex), d in clean.groupby(["subject", "exercise"], observed=True, sort=True):
        d = d.reset_index(drop=True)
        mode = EXERCISES[ex][1]
        if mode == "dynamic" and (d["rep"] >= 0).any():
            reps, curves = process_reps(d, int(s), ex)
            rep_tables.append(reps)
            curve_list += curves
            set_rows.append({"subject": int(s), "exercise": ex, "family": EXERCISES[ex][0],
                             **aggregate_set(reps)})
        elif mode == "hold":
            hold_rows.append(process_hold(d, int(s), ex))
    reps = pd.concat(rep_tables, ignore_index=True)
    return reps, np.stack(curve_list).astype(np.float32), pd.DataFrame(set_rows), pd.DataFrame(hold_rows)


def main():
    clean = pd.read_parquet(OUT_DIR / "clean.parquet")
    reps, curves, sets, holds = build(clean)
    reps.to_csv(OUT_DIR / "reps.csv", index=False)
    np.savez_compressed(OUT_DIR / "curves.npz", curves=curves, names=np.array(CURVES))
    sets.to_csv(OUT_DIR / "set_features.csv", index=False)
    holds.to_csv(OUT_DIR / "holds.csv", index=False)
    print(f"reps: {len(reps)} ({int(reps['valid'].sum())} valid), curves {curves.shape}")
    print(f"repetitive sets: {len(sets)} x {sets.shape[1]} cols; hold sets: {len(holds)}")
    print("\nmedian per exercise (valid reps):")
    cols = ["dur", "knee_max", "thigh_max", "uarm_max", "t_bottom", "speed_thigh", "ldlj_thigh"]
    print(reps[reps["valid"]].groupby("exercise")[cols].median().round(1).to_string())


if __name__ == "__main__":
    main()
