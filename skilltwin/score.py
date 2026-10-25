"""Step 7 - Score every repetition against the expert band with DTW.

Each subject is scored against a band built WITHOUT that subject (as a new
trainee would be). DTW aligns the repetition to the expert mean so a slower
person is compared phase by phase. After alignment, every point outside
mean +/- 2 SD counts as a deviation.

    out/rep_scores.csv  per repetition: skill score 0-100, and per curve the
                        fraction out of band and the worst out-of-band stretch
    out/set_scores.csv  per set: median score + the dominant deviation

Usage: python -m skilltwin.score
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from skilltwin.expert import BAND_K, build_band, load_inputs, select_experts
from skilltwin.config import OUT_DIR

DTW_WINDOW = 15          # max shift, in % of the movement
SCORE_SCALE = 2.0        # RMS distance (in band SDs) at which the score falls to ~61


def dtw_batch(x, mean, sd, window=DTW_WINDOW):
    """Align each repetition in x (B x F x N) to the band mean (F x N).

    Returns the warped repetitions (B x F x N) on the expert's time axis.
    """
    b, f, n = x.shape
    zx = x / sd.mean(axis=1)[None, :, None]
    zm = mean / sd.mean(axis=1)[:, None]
    cost = ((zx[:, :, :, None] - zm[None, :, None, :]) ** 2).mean(axis=1)   # B x N x N
    acc = np.full((b, n + 1, n + 1), np.inf)
    acc[:, 0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = max(1, i - window), min(n, i + window)
        for j in range(lo, hi + 1):
            acc[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(
                np.minimum(acc[:, i - 1, j - 1], acc[:, i - 1, j]), acc[:, i, j - 1])
    warped = np.zeros_like(x)
    for k in range(b):
        i, j = n, n
        sums, counts = np.zeros((f, n)), np.zeros(n)
        while i > 0 and j > 0:
            sums[:, j - 1] += x[k, :, i - 1]
            counts[j - 1] += 1
            step = np.argmin([acc[k, i - 1, j - 1], acc[k, i - 1, j], acc[k, i, j - 1]])
            i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
        warped[k] = sums / np.maximum(counts, 1)
    return warped


def longest_run(mask):
    """(start, end) indices of the longest True run, or None."""
    best, start = None, None
    for i, v in enumerate(np.r_[mask, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    return best


def score_reps(x, band):
    """Per-rep scores and deviation records for repetitions x (B x F x N)."""
    w = dtw_batch(x, band["mean"], band["sd"])
    z = (w - band["mean"]) / band["sd"]
    out = np.abs(z) > BAND_K
    rows = []
    for k in range(len(x)):
        dist = np.sqrt(np.mean(z[k] ** 2))
        row = {"score": 100 * np.exp(-0.5 * (dist / SCORE_SCALE) ** 2), "dist": dist,
               "in_band": 100 * (1 - out[k].mean())}
        for fi, c in enumerate(band["curves"]):
            row[f"out_{c}"] = out[k, fi].mean()
            row[f"z_{c}"] = z[k, fi].mean()
            run = longest_run(out[k, fi])
            if run:
                seg = slice(run[0], run[1] + 1)
                row[f"dev_{c}_start"], row[f"dev_{c}_end"] = run
                row[f"dev_{c}_delta"] = float((w[k, fi, seg] - band["mean"][fi, seg]).mean())
        rows.append(row)
    return rows


def dominant_deviation(r, curves):
    """The curve most often out of band across a set's reps, with its typical stretch."""
    frac = {c: r[f"out_{c}"].mean() for c in curves}
    c = max(frac, key=frac.get)
    if frac[c] == 0 or f"dev_{c}_delta" not in r:
        return {"dev_curve": None}
    d = r.dropna(subset=[f"dev_{c}_delta"])
    return {"dev_curve": c, "dev_frac": frac[c],
            "dev_start": int(d[f"dev_{c}_start"].median()),
            "dev_end": int(d[f"dev_{c}_end"].median()),
            "dev_delta": float(d[f"dev_{c}_delta"].median())}


def score_subject(reps, curves, names, labels, ex, s, sel):
    """Score one subject's reps against bands that never contain that subject.

    Every subject is scored against bands of the same size: experts are left out
    of their own band, so for everyone else one expert at a time is also left
    out and the results averaged. Otherwise experts would face a smaller band
    than everyone else.
    """
    full = build_band(reps, curves, names, labels, ex, exclude={s})
    with_reps = reps.loc[(reps["exercise"] == ex) & reps["valid"], "subject"].unique()
    is_expert = s in select_experts(labels, ex, sets_with_reps=with_reps)[0]
    others = [()] if is_expert else [{e} for e in full["experts"]]
    meta = reps.loc[sel, ["subject", "exercise", "family", "rep"]].reset_index(drop=True)
    runs = []
    for drop in others:
        band = build_band(reps, curves, names, labels, ex, exclude={s, *drop})
        ci = [names.index(c) for c in band["curves"]]
        x = curves[sel][:, ci, :]
        have = ~np.isnan(x).any(axis=(0, 2))     # sets without IMU: score EMG curves only
        if not have.all():
            band = {**band, "curves": [c for c, h in zip(band["curves"], have) if h],
                    "mean": band["mean"][have], "sd": band["sd"][have]}
            x = x[:, have, :]
        runs.append(pd.concat([meta, pd.DataFrame(score_reps(x, band))], axis=1))
    r = (pd.concat(runs).groupby(["subject", "exercise", "family", "rep"], as_index=False)
         .mean(numeric_only=True))
    return r, full, have


def main():
    reps, curves, names, labels = load_inputs()
    rep_rows, set_rows = [], []
    for ex in sorted(reps["exercise"].unique()):
        for s in sorted(reps.loc[reps["exercise"] == ex, "subject"].unique()):
            sel = ((reps["exercise"] == ex) & (reps["subject"] == s) & reps["valid"]).to_numpy()
            if not sel.any():
                continue
            r, band, have = score_subject(reps, curves, names, labels, ex, s, sel)
            rep_rows.append(r)
            used = [c for c, h in zip(band["curves"], have) if h]
            set_rows.append({"subject": s, "exercise": ex, "family": r["family"].iloc[0],
                             "band_tier": band["tier"], "imu": bool(have.all()),
                             "score": r["score"].median(),
                             "dist": r["dist"].median(), "in_band": r["in_band"].median(),
                             **{f"out_{c}": r[f"out_{c}"].mean() for c in used},
                             **{f"z_{c}": r[f"z_{c}"].median() for c in used},
                             **dominant_deviation(r, used)})
    rep_scores = pd.concat(rep_rows, ignore_index=True)
    sets = pd.DataFrame(set_rows).merge(
        labels[["subject", "exercise", "overall", "n_errors", "is_expert"]], on=["subject", "exercise"])
    rep_scores.to_csv(OUT_DIR / "rep_scores.csv", index=False)
    sets.to_csv(OUT_DIR / "set_scores.csv", index=False)

    print("Does the band score agree with the physiotherapist? (each subject held out of its band)\n")
    print(f"{'exercise':28s} tier  n  spearman(score, overall)   expert vs rest median score")
    for ex, g in sets.groupby("exercise"):
        rho = spearmanr(g["score"], g["overall"]).statistic
        e, ne = g.loc[g["is_expert"], "score"], g.loc[~g["is_expert"], "score"]
        print(f"{ex:28s} {g['band_tier'].iloc[0]:4d} {len(g):2d}  {rho:+.2f}"
              f"                     {e.median():5.1f} vs {ne.median():5.1f}")
    rho = spearmanr(sets["score"], sets["overall"]).statistic
    p = mannwhitneyu(sets.loc[sets["is_expert"], "score"], sets.loc[~sets["is_expert"], "score"],
                     alternative="greater").pvalue
    print(f"\npooled: spearman {rho:+.2f}; expert sets score higher than the rest, one-sided p={p:.3g}")
    print("\nmost common dominant deviation:")
    print(sets["dev_curve"].value_counts().to_string())


if __name__ == "__main__":
    main()
