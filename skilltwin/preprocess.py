"""Step 3 - Clean the signals.

    out/signals.parquet -> out/clean.parquet

- Resample 200 Hz -> 100 Hz (the IMU's native rate; the 200 Hz rows repeat each
  quaternion twice, and the EMG envelope is already 100 ms smoothed).
- Quaternions: fix sign flips, low-pass 15 Hz (keeps the 4-12 Hz tremor band),
  renormalise.
- EMG envelope: divide by the subject's 99th percentile over all exercises, per
  muscle. The single-muscle "reference" files are not maximal contractions
  (rectus/gastroc reach ~0.2x of exercise peaks) and are missing for some
  subjects, so they are not used.
- Raw EMG is dropped: at 100 Hz it would alias.

Usage: python -m skilltwin.preprocess
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from skilltwin import quat
from skilltwin.config import FS, OUT_DIR, QUAT, SENSORS

FS_CLEAN = 100
QUAT_CUTOFF_HZ = 15
EMG_NORM_PERCENTILE = 99
EMG_CLIP = 3.0          # artefact spikes above 3x the subject's p99 are clipped


def clean_quaternions(q):
    """Some files have no IMU data at all (all NaN); those stay NaN."""
    if np.isnan(q).any():
        return np.full(q.shape, np.nan)
    q = quat.make_continuous(quat.normalize(q.astype(np.float64)))
    b, a = butter(4, QUAT_CUTOFF_HZ / (FS_CLEAN / 2))
    if len(q) > 3 * max(len(a), len(b)):
        q = filtfilt(b, a, q, axis=0)
    return quat.normalize(q)


def clean(signals):
    step = FS // FS_CLEAN
    keep = ["subject", "exercise", "source", "t", "rep", "phase"]
    env_cols = [f"env_{k}" for k in SENSORS]
    parts = []
    for _, grp in signals.groupby(["subject", "exercise"], observed=True, sort=False):
        d = grp.iloc[::step][keep + env_cols].reset_index(drop=True)
        for k in SENSORS:
            cols = [f"q_{k}_{c}" for c in QUAT]
            d[cols] = clean_quaternions(grp[cols].to_numpy()[::step]).astype(np.float32)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)

    exercise_rows = ~out["exercise"].astype(str).str.startswith("ref:")
    scale = (out[exercise_rows].groupby("subject")[env_cols]
             .quantile(EMG_NORM_PERCENTILE / 100))
    out[env_cols] = (out[env_cols] / scale.loc[out["subject"]].to_numpy()).clip(upper=EMG_CLIP)
    out[env_cols] = out[env_cols].astype(np.float32)
    out = out[exercise_rows].reset_index(drop=True)
    out["exercise"] = out["exercise"].astype(str).astype("category")
    return out, scale


def main():
    signals = pd.read_parquet(OUT_DIR / "signals.parquet")
    out, scale = clean(signals)
    out.to_parquet(OUT_DIR / "clean.parquet", index=False)
    scale.to_csv(OUT_DIR / "emg_scale.csv")
    print(f"clean: {len(out):,} rows x {out.shape[1]} cols at {FS_CLEAN} Hz")
    print(f"EMG p{EMG_NORM_PERCENTILE} scale per subject (uV):\n{scale.round(0).describe().to_string()}")
    clipped = (out[[f'env_{k}' for k in SENSORS]] >= EMG_CLIP).mean()
    print(f"fraction clipped at {EMG_CLIP}x: {clipped.round(4).to_dict()}")


if __name__ == "__main__":
    main()
