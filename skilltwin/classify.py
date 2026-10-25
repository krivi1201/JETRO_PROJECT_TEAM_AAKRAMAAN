"""Step 8 - Predict the physiotherapist's rule ratings from sensor features.

One small model per rule, pooled across the exercises where that rule applies.
Features are standardised within each exercise (median / IQR from the training
subjects only), so "deep for a wide squat" and "deep for a jump squat" are
comparable. Feature subsets per rule are chosen from biomechanics, not from the
labels.

Evaluation: leave-one-subject-out (train on 18 people, test on the 19th).
Compared against "always predict no error" and logistic regression.
95% CIs come from resampling subjects.

    out/classifier_results.csv   metrics per rule and model
    out/oof_predictions.csv      held-out probability for every set and rule
    out/shap_importance.csv      mean |SHAP| per feature, final model per rule
    out/overall_results.csv      overall 0-10 score regression
    out/overall_predictions.csv  held-out predicted overall score per set
    out/models/<rule>.json       final XGBoost model per rule (trained on everyone)

Usage: python -m skilltwin.classify
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score, roc_auc_score)

from skilltwin.config import OUT_DIR

SEED = 0
MIN_ERRORS = 6
LEG = ["knee", "thigh", "shank"]

# rule -> (table, exercises it is modelled on, features, observability)
# Burpees are left out of squat_depth: the body rotates through a plank, so the
# knee/thigh rotation angles saturate and say nothing about squat depth.
RULES = {
    "squat_depth": ("reps", ["Standard squat", "Wide squat", "Jump squats"],
                    [f"{a}_{s}_mean" for a in LEG for s in ("max", "rom")]
                    + ["t_bottom_mean", "z_knee", "z_thigh", "z_shank",
                       "out_knee", "out_thigh", "out_shank"], "yes"),
    "pushup_depth": ("reps", ["Push-ups", "Triceps push-ups"],
                     ["uarm_max_mean", "uarm_rom_mean", "uarm_max_cv", "speed_uarm_mean",
                      "z_uarm", "out_uarm", "env_peak_triceps_mean", "env_peak_biceps_mean"], "yes"),
    "movement_fluidity": ("reps", None,
                          ["ldlj_thigh_mean", "ldlj_shank_mean", "ldlj_uarm_mean",
                           "speed_thigh_cv", "speed_uarm_cv", "dur_cv", "dur_mean",
                           "speed_thigh_mean", "speed_uarm_mean", "dist"], "yes"),
    "tempo": ("reps", None,
              ["dur_cv", "dur_trend", "dur_mean", "n_reps", "speed_thigh_cv",
               "speed_uarm_cv", "t_bottom_mean"], "yes"),
    "technique": ("reps", None,
                  ["knee_max_trend", "thigh_max_trend", "uarm_max_trend", "dur_trend",
                   "ldlj_thigh_trend", "ldlj_uarm_trend", "speed_thigh_trend",
                   "env_mean_rectus_trend", "env_mean_gastroc_trend", "env_mean_biceps_trend",
                   "env_mean_triceps_trend", "knee_max_cv", "uarm_max_cv", "dist"], "yes"),
    "stability": ("holds", None,
                  [f"{m}_{s}" for m in ("sway", "tremor", "drift") for s in ("thigh", "shank", "uarm")]
                  + [f"env_cv_{m}" for m in ("biceps", "triceps", "rectus", "gastroc")], "yes"),
    "hip_angle": ("reps", ["Push-ups", "Triceps push-ups"],
                  ["thigh_max_mean", "thigh_rom_mean", "z_thigh", "out_thigh",
                   "uarm_max_mean", "env_mean_rectus_mean"], "partial"),
    "knee_joint_axis": ("reps", None,
                        [f"{a}_{s}_mean" for a in LEG for s in ("max", "rom")]
                        + ["ldlj_thigh_mean", "ldlj_shank_mean", "coact_leg_mean"], "partial"),
}
# hold_and_tempo (burpees, jump squats, ...) asks the same question as tempo.
RULE_MERGE = {"hold_and_tempo": "tempo"}


def load_tables():
    labels = pd.read_csv(OUT_DIR / "labels.csv")
    for src, dst in RULE_MERGE.items():
        labels[dst] = labels[dst].fillna(labels[src])
    keys = ["subject", "exercise"]
    sets = pd.read_csv(OUT_DIR / "set_features.csv").merge(
        pd.read_csv(OUT_DIR / "set_scores.csv").drop(
            columns=["family", "overall", "n_errors", "is_expert"]), on=keys, how="left")
    reps = sets.merge(labels, on=keys + ["family"])
    holds = pd.read_csv(OUT_DIR / "holds.csv").merge(labels, on=keys + ["family"])
    return {"reps": reps, "holds": holds}


def standardise_within(train, test, feats):
    """Scale each feature by its per-exercise median / IQR on the training rows."""
    out = []
    for df in (train, test):
        x = df[feats].astype(float).copy()
        for ex in df["exercise"].unique():
            ref = train.loc[train["exercise"] == ex, feats].astype(float)
            if ref.empty:
                ref = train[feats].astype(float)
            med = ref.median()
            iqr = (ref.quantile(0.75) - ref.quantile(0.25)).replace(0, 1).fillna(1)
            m = df["exercise"] == ex
            x.loc[m] = (x.loc[m] - med) / iqr
        out.append(x)
    return out


def xgb_model(pos_weight):
    return xgb.XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.7, min_child_weight=2,
                             reg_lambda=2.0, scale_pos_weight=pos_weight,
                             eval_metric="logloss", random_state=SEED, verbosity=0)


def loso_predict(d, feats, target):
    """Out-of-fold error probabilities for logistic regression and XGBoost."""
    p_lr, p_xgb = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for s in d["subject"].unique():
        tr, te = d["subject"] != s, d["subject"] == s
        y_tr = d.loc[tr, target]
        if y_tr.nunique() < 2:
            continue
        x_tr, x_te = standardise_within(d[tr], d[te], feats)
        pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        p_xgb[te.to_numpy()] = xgb_model(pw).fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
        fill = x_tr.median()
        lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)
        lr.fit(x_tr.fillna(fill).fillna(0), y_tr)
        p_lr[te.to_numpy()] = lr.predict_proba(x_te.fillna(fill).fillna(0))[:, 1]
    return p_lr, p_xgb


def subject_bootstrap_auc(d, y, p, n=1000):
    rng = np.random.default_rng(SEED)
    subjects = d["subject"].unique()
    idx = {s: np.where(d["subject"].to_numpy() == s)[0] for s in subjects}
    aucs = []
    for _ in range(n):
        rows = np.concatenate([idx[s] for s in rng.choice(subjects, len(subjects))])
        if 0 < y[rows].sum() < len(rows):
            aucs.append(roc_auc_score(y[rows], p[rows]))
    return np.percentile(aucs, [2.5, 97.5])


def metrics(y, p, d):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    lo, hi = subject_bootstrap_auc(d, y, p)
    return {"auc": roc_auc_score(y, p), "auc_lo": lo, "auc_hi": hi,
            "bal_acc": balanced_accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def shap_importance(model, x):
    contrib = model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=x.columns).sort_values(ascending=False)


def run_rules(tables):
    results, oof, shap_rows = [], [], []
    (OUT_DIR / "models").mkdir(exist_ok=True)
    for rule, (table, exercises, feats, level) in RULES.items():
        d = tables[table]
        d = d[d[rule].notna()]
        if exercises:
            d = d[d["exercise"].isin(exercises)]
        d = d.reset_index(drop=True).assign(error=lambda t: (t[rule] == 0).astype(int))
        feats = [f for f in feats if f in d.columns]
        y = d["error"].to_numpy()
        if y.sum() < MIN_ERRORS:
            print(f"skip {rule}: only {y.sum()} errors")
            continue
        p_lr, p_xgb = loso_predict(d, feats, "error")
        base = {"rule": rule, "observable": level, "n_sets": len(d), "n_errors": int(y.sum()),
                "n_subjects": d["subject"].nunique(), "exercises": "; ".join(sorted(d["exercise"].unique()))}
        results.append({**base, "model": "always_no_error",
                        **metrics(y, np.zeros(len(y)) + 1e-9 * np.arange(len(y)), d),
                        "auc": 0.5, "auc_lo": 0.5, "auc_hi": 0.5})
        results.append({**base, "model": "logreg", **metrics(y, p_lr, d)})
        results.append({**base, "model": "xgboost", **metrics(y, p_xgb, d)})
        oof.append(d[["subject", "exercise"]].assign(rule=rule, error=y, p_logreg=p_lr, p_xgb=p_xgb))

        x_all, _ = standardise_within(d, d.iloc[:0], feats)
        final = xgb_model((y == 0).sum() / y.sum()).fit(x_all, y)
        final.save_model(OUT_DIR / "models" / f"{rule}.json")
        stats = {ex: {"median": g[feats].median().to_dict(),
                      "iqr": (g[feats].quantile(.75) - g[feats].quantile(.25)).to_dict()}
                 for ex, g in d.groupby("exercise")}
        (OUT_DIR / "models" / f"{rule}_scaling.json").write_text(json.dumps(
            {"features": feats, "per_exercise": stats}, default=float))
        for f, v in shap_importance(final, x_all).items():
            shap_rows.append({"rule": rule, "feature": f, "mean_abs_shap": v})
    return pd.DataFrame(results), pd.concat(oof), pd.DataFrame(shap_rows)


def run_overall(tables):
    """Overall 0-10 score: XGBoost regressor vs predicting the training mean."""
    rows, preds = [], []
    for table, d in tables.items():
        d = d.dropna(subset=["overall"]).reset_index(drop=True)
        drop = {"subject", "exercise", "family", "mode", "side", "overall", "n_rules",
                "n_errors", "is_expert", "dev_curve", "imu", "band_tier"}
        feats = [c for c in d.columns if c not in drop and d[c].dtype != object
                 and not c.startswith(tuple(RULES)) and c not in RULE_MERGE]
        feats = [c for c in feats if c in tables[table].columns and not d[c].isin([0, 1]).all()]
        pred, base = np.zeros(len(d)), np.zeros(len(d))
        for s in d["subject"].unique():
            tr, te = d["subject"] != s, d["subject"] == s
            x_tr, x_te = standardise_within(d[tr], d[te], feats)
            m = xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.5, reg_lambda=2.0, random_state=SEED)
            pred[te.to_numpy()] = m.fit(x_tr, d.loc[tr, "overall"]).predict(x_te)
            base[te.to_numpy()] = d.loc[tr, "overall"].mean()
        y = d["overall"].to_numpy()
        rows.append({"table": table, "n_sets": len(d), "n_features": len(feats),
                     "mae_model": mean_absolute_error(y, pred),
                     "mae_mean_baseline": mean_absolute_error(y, base),
                     "spearman": spearmanr(y, pred).statistic})
        preds.append(d[["subject", "exercise", "overall"]].assign(table=table, predicted=pred))
    return pd.DataFrame(rows), pd.concat(preds)


def main():
    tables = load_tables()
    results, oof, shap = run_rules(tables)
    overall, overall_pred = run_overall(tables)
    overall_pred.to_csv(OUT_DIR / "overall_predictions.csv", index=False)
    results.to_csv(OUT_DIR / "classifier_results.csv", index=False)
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    shap.to_csv(OUT_DIR / "shap_importance.csv", index=False)
    overall.to_csv(OUT_DIR / "overall_results.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["rule", "observable", "n_sets", "n_errors", "model", "auc", "auc_lo", "auc_hi",
            "bal_acc", "f1", "precision", "recall", "tp", "fn", "fp"]
    print("Leave-one-subject-out results (error class = positive):\n")
    print(results[results["model"] != "always_no_error"][cols].round(2).to_string(index=False))
    print("\nalways_no_error baseline: AUC 0.50, balanced accuracy 0.50, F1 0.00 for every rule")
    print("\nTop SHAP features per rule (final XGBoost):")
    for rule, g in shap.groupby("rule", sort=False):
        print(f"  {rule:18s} " + ", ".join(f"{f} {v:.2f}" for f, v in g.head(3)[["feature", "mean_abs_shap"]].values))
    print("\nOverall 0-10 score:")
    print(overall.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
