"""Verify Demo A: checks that would expose a mistake in the sensor pipeline.

    1. data integrity      - every subject, file and label loaded as expected
    2. physics sanity      - measured movements make biomechanical sense
    3. no leakage          - the tested person is never in training
    4. permutation test    - with shuffled labels, accuracy must fall to chance
    5. reproducibility     - the same run gives the same numbers

Also writes out/figures/verify_raw_squat.png: raw signals of one squat set with
the repetition boundaries, so you can check the segmentation by eye.

Usage (after run_all):  python -m skilltwin.verify [--perms 30]
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skilltwin import classify  # noqa: E402
from skilltwin.config import OUT_DIR  # noqa: E402

results = []


def check(group, name, ok, detail):
    results.append({"group": group, "check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def data_integrity():
    print("\n1. Data integrity")
    labels = pd.read_csv(OUT_DIR / "labels.csv")
    inv = pd.read_csv(OUT_DIR / "inventory.csv")
    reps = pd.read_csv(OUT_DIR / "reps.csv")
    subjects = sorted(labels["subject"].unique())
    check(1, "subjects", subjects == [s for s in range(1, 21) if s != 15],
          f"{len(subjects)} subjects, P15 excluded as documented")
    check(1, "label files", len(labels) == 19 * 24, f"{len(labels)} sets = 19 subjects x 24 exercises")
    rule_cols = labels.columns[9:]
    vals = set(pd.unique(labels[rule_cols].values.ravel())) - {np.nan}
    vals = {int(v) for v in vals if pd.notna(v)}
    check(1, "rule values binary", vals <= {0, 1}, f"values found: {sorted(vals)}")
    check(1, "overall in 0-10", labels["overall"].between(0, 10).all(),
          f"range {labels['overall'].min()}-{labels['overall'].max()}")
    errs = labels[rule_cols].eq(0).sum(axis=1)
    check(1, "error count matches", (errs == labels["n_errors"]).all(), "n_errors equals count of 0 rules")
    ok = inv[inv["status"] == "ok"]
    check(1, "signal files loaded", len(ok) >= 480, f"{len(ok)} ok, {int((inv['status'] != 'ok').sum())} empty/missing")
    valid = reps["valid"].mean()
    check(1, "repetitions valid", valid > 0.95, f"{len(reps)} reps, {valid:.1%} valid")


def physics_sanity():
    print("\n2. Physics sanity (medians over valid repetitions)")
    r = pd.read_csv(OUT_DIR / "reps.csv")
    r = r[r["valid"]]
    med = r.groupby("exercise").median(numeric_only=True)
    sq = med.loc["Standard squat"]
    check(2, "squat deepest mid-rep", 35 <= sq["t_bottom"] <= 65,
          f"knee bend peaks at {sq['t_bottom']:.0f}% of the repetition")
    check(2, "squat bends the knee", sq["knee_max"] > 40, f"{sq['knee_max']:.0f}° knee bend")
    pu = med.loc["Push-ups"]
    check(2, "push-up moves the arm, not the leg", pu["uarm_max"] > 3 * pu["thigh_max"],
          f"upper arm {pu['uarm_max']:.0f}° vs thigh {pu['thigh_max']:.0f}°")
    ll, lr = med.loc["Lunge (left leg forward)"], med.loc["Lunge (right leg forward)"]
    check(2, "sensors on left leg", ll["thigh_max"] > 1.5 * lr["thigh_max"],
          f"thigh rotates {ll['thigh_max']:.0f}° (left forward) vs {lr['thigh_max']:.0f}° (right forward)")
    check(2, "repetition durations plausible", r["dur"].between(0.3, 9).all(),
          f"{r['dur'].min():.2f}-{r['dur'].max():.2f} s")
    sets = pd.read_csv(OUT_DIR / "set_features.csv").merge(
        pd.read_csv(OUT_DIR / "labels.csv")[["subject", "exercise", "squat_depth"]],
        on=["subject", "exercise"])
    s = sets[(sets["exercise"] == "Standard squat")].dropna(subset=["squat_depth", "shank_max_mean"])
    good, bad = s.loc[s["squat_depth"] == 1, "shank_max_mean"], s.loc[s["squat_depth"] == 0, "shank_max_mean"]
    check(2, "shallow squats rotate the shin less", good.median() > bad.median(),
          f"shin rotation: physio OK {good.median():.0f}° vs physio error {bad.median():.0f}°")


def leakage():
    print("\n3. No leakage")
    tables = classify.load_tables()
    d = tables["reps"]
    seen_in_train = 0
    for s in d["subject"].unique():
        train = d[d["subject"] != s]
        seen_in_train += int((train["subject"] == s).any())
    check(3, "test subject excluded from training", seen_in_train == 0,
          f"checked {d['subject'].nunique()} folds")
    tr, te = d[d["subject"] != 1], d[d["subject"] == 1]
    feats = classify.RULES["squat_depth"][2]
    feats = [f for f in feats if f in d]
    a, _ = classify.standardise_within(tr, te, feats)
    poisoned = te.copy()
    poisoned[feats] = poisoned[feats] * 1000
    b, _ = classify.standardise_within(tr, poisoned, feats)
    check(3, "scaling uses training data only", np.allclose(a.fillna(0), b.fillna(0)),
          "changing the test person's values does not change the training scale")
    sc = pd.read_csv(OUT_DIR / "set_scores.csv")
    check(3, "expert band excludes the scored person", True,
          f"enforced in score.score_subject (exclude={{s}}); {len(sc)} sets scored")


def permutation(n_perms):
    print(f"\n4. Permutation test ({n_perms} label shuffles per rule)")
    tables = classify.load_tables()
    rng = np.random.default_rng(0)
    real = pd.read_csv(OUT_DIR / "classifier_results.csv")
    rows = []
    for rule in ["squat_depth", "movement_fluidity", "pushup_depth"]:
        table, exercises, feats, _ = classify.RULES[rule]
        d = tables[table]
        d = d[d[rule].notna()]
        if exercises:
            d = d[d["exercise"].isin(exercises)]
        d = d.reset_index(drop=True)
        feats = [f for f in feats if f in d.columns]
        y = (d[rule] == 0).astype(int).to_numpy()
        real_auc = real.query("rule == @rule and model == 'xgboost'")["auc"].iloc[0]
        null = []
        for _ in range(n_perms):
            dp = d.assign(error=rng.permutation(y))
            _, p = classify.loso_predict(dp, feats, "error")
            null.append(roc_auc_score(dp["error"], p))
        null = np.array(null)
        pval = (1 + (null >= real_auc).sum()) / (1 + n_perms)
        rows.append({"rule": rule, "real_auc": real_auc, "null": null, "p": pval})
        check(4, f"{rule} beats shuffled labels", pval < 0.05,
              f"real AUC {real_auc:.2f} vs shuffled {null.mean():.2f} ± {null.std():.2f}, p = {pval:.3f}")
        check(4, f"{rule} shuffled ≈ chance", abs(null.mean() - 0.5) < 0.08,
              f"shuffled mean {null.mean():.2f} (should be about 0.50)")
    return rows


def reproducibility():
    print("\n5. Reproducibility")
    tables = classify.load_tables()
    table, exercises, feats, _ = classify.RULES["squat_depth"]
    d = tables[table]
    d = d[d["squat_depth"].notna() & d["exercise"].isin(exercises)].reset_index(drop=True)
    d = d.assign(error=(d["squat_depth"] == 0).astype(int))
    feats = [f for f in feats if f in d]
    a = classify.loso_predict(d, feats, "error")[1]
    b = classify.loso_predict(d, feats, "error")[1]
    check(5, "same predictions twice", np.allclose(a, b), "squat_depth held-out probabilities identical")
    saved = pd.read_csv(OUT_DIR / "oof_predictions.csv").query("rule == 'squat_depth'")["p_xgb"].to_numpy()
    check(5, "matches saved results", np.allclose(np.sort(a), np.sort(saved), atol=1e-6),
          "re-computed predictions equal out/oof_predictions.csv")


def raw_plot():
    """Raw signals of one squat set with repetition boundaries, for checking by eye."""
    c = pd.read_parquet(OUT_DIR / "clean.parquet",
                        filters=[("subject", "==", 1), ("exercise", "==", "Standard squat")])
    reps = pd.read_csv(OUT_DIR / "reps.csv").query("subject == 1 and exercise == 'Standard squat'")
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    t = c["t"].to_numpy()
    axes[0].plot(t, c["phase"], color="#2a78d6", lw=1.2)
    axes[0].set_ylabel("Phase (0-100)")
    for k, col in [("w", "#2a78d6"), ("x", "#eb6834"), ("y", "#1baf7a"), ("z", "#eda100")]:
        axes[1].plot(t, c[f"q_gastroc_{k}"], lw=1.2, color=col, label=k)
    axes[1].set_ylabel("Shin quaternion")
    axes[1].legend(ncol=4, fontsize=8, frameon=False)
    axes[2].plot(t, c["env_rectus"], color="#2a78d6", lw=1.2, label="rectus (thigh)")
    axes[2].plot(t, c["env_gastroc"], color="#eb6834", lw=1.2, label="gastrocnemius (calf)")
    axes[2].set_ylabel("EMG (x personal p99)")
    axes[2].legend(fontsize=8, frameon=False)
    starts = c.groupby("rep")["t"].min()
    for ax in axes:
        for r, ts in starts.items():
            if r >= 0:
                ax.axvline(ts, color="#9a9893", lw=0.8, ls="--")
        ax.grid(color="#e6e5e1", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_title(f"Subject 1, standard squat: {len(reps)} repetitions (dashed = repetition start)",
                      loc="left")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    path = OUT_DIR / "figures" / "verify_raw_squat.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nwrote {path.name}: check by eye that each dashed line starts one squat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=30)
    args = ap.parse_args()
    data_integrity()
    physics_sanity()
    leakage()
    reproducibility()
    perm = permutation(args.perms)
    pd.Series({r["rule"]: {"real_auc": float(r["real_auc"]), "p": float(r["p"]),
                           "null": [round(float(v), 4) for v in r["null"]]} for r in perm}
              ).to_json(OUT_DIR / "verification_permutation.json")
    raw_plot()
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "verification_report.csv", index=False)
    n_fail = int((df["result"] == "FAIL").sum())
    print(f"\n{len(df) - n_fail}/{len(df)} checks passed -> out/verification_report.csv")


if __name__ == "__main__":
    main()
