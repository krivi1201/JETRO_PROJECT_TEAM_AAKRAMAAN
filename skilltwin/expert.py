"""Step 6 - Expert range: the acceptable band of each feature curve over a repetition.

Expert sets are chosen per exercise, relaxing the bar only when too few exist:
    tier 1  overall >= 8 and zero errors
    tier 2  overall >= 8 and zero errors on rules the sensors can observe
    tier 3  the top 3 sets by overall score
The band is mean +/- 2 SD over all valid expert repetitions, each resampled to
0-100 % of the movement.

Usage: python -m skilltwin.expert   (writes out/bands.npz built from all subjects)
"""

import numpy as np
import pandas as pd

from skilltwin.config import EXERCISES, EXPERT_MIN_SCORE, OBSERVABILITY, OUT_DIR

MIN_EXPERT_SETS = 3
BAND_K = 2.0
SD_FLOOR = {"angle": 3.0, "env": 0.03}     # degrees / fraction of personal EMG p99

FAMILY_CURVES = {
    "squat":  ["knee", "thigh", "shank", "env_rectus", "env_gastroc"],
    "lunge":  ["knee", "thigh", "shank", "env_rectus", "env_gastroc"],
    "burpee": ["knee", "thigh", "uarm", "env_rectus", "env_triceps"],
    "pushup": ["uarm", "thigh", "env_biceps", "env_triceps"],
}


def observable_errors(labels):
    rules = [r for r, (lvl, _) in OBSERVABILITY.items() if lvl == "yes" and r in labels]
    return (labels[rules] == 0).sum(axis=1)


def select_experts(labels, exercise, exclude=(), sets_with_reps=None):
    """Return (set of expert subject ids, tier) for one exercise."""
    lab = labels[(labels["exercise"] == exercise) & ~labels["subject"].isin(exclude)]
    if sets_with_reps is not None:
        lab = lab[lab["subject"].isin(sets_with_reps)]
    tiers = [
        (1, lab[(lab["overall"] >= EXPERT_MIN_SCORE) & (lab["n_errors"] == 0)]),
        (2, lab[(lab["overall"] >= EXPERT_MIN_SCORE) & (observable_errors(lab) == 0)]),
    ]
    for tier, chosen in tiers:
        if len(chosen) >= MIN_EXPERT_SETS:
            return set(chosen["subject"]), tier
    top = lab.sort_values(["overall", "n_errors"], ascending=[False, True]).head(MIN_EXPERT_SETS)
    return set(top["subject"]), 3


def sd_floor(curve_names):
    return np.array([SD_FLOOR["env"] if c.startswith("env_") else SD_FLOOR["angle"]
                     for c in curve_names])


def build_band(reps, curves, all_names, labels, exercise, exclude=()):
    """Band dict for one exercise: mean/sd arrays (F x N) over the family's curves."""
    names = FAMILY_CURVES[EXERCISES[exercise][0]]
    ci = [list(all_names).index(c) for c in names]
    in_ex = (reps["exercise"] == exercise) & reps["valid"]
    experts, tier = select_experts(labels, exercise, exclude,
                                   sets_with_reps=reps.loc[in_ex, "subject"].unique())
    sel = in_ex & reps["subject"].isin(experts)
    x = curves[sel.to_numpy()][:, ci, :]          # NaN where a set had no IMU data
    sd = np.maximum(np.nanstd(x, axis=0), sd_floor(names)[:, None])
    return {"exercise": exercise, "curves": names, "mean": np.nanmean(x, axis=0), "sd": sd,
            "tier": tier, "n_sets": len(experts), "n_reps": len(x),
            "experts": sorted(experts)}


def load_inputs():
    reps = pd.read_csv(OUT_DIR / "reps.csv")
    z = np.load(OUT_DIR / "curves.npz")
    labels = pd.read_csv(OUT_DIR / "labels.csv")
    return reps, z["curves"], list(z["names"]), labels


def main():
    reps, curves, names, labels = load_inputs()
    bands = {}
    print(f"{'exercise':28s} tier sets reps  curves")
    for ex in sorted(reps["exercise"].unique()):
        b = build_band(reps, curves, names, labels, ex)
        bands[ex] = b
        print(f"{ex:28s} {b['tier']:4d} {b['n_sets']:4d} {b['n_reps']:4d}  {', '.join(b['curves'])}")
    np.savez(OUT_DIR / "bands.npz", bands=np.array(bands, dtype=object))
    print(f"\nwritten {OUT_DIR / 'bands.npz'}")


if __name__ == "__main__":
    main()
