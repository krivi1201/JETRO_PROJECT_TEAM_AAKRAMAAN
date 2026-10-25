"""Step 10 - Figures for the judges.

    out/figures/1_band_vs_trainee.png   expert band vs a held-out trainee repetition
    out/figures/2_rule_auc.png          held-out AUC with 95% CI per rule
    out/figures/3_confusion.png         confusion matrices of the validated rules
    out/figures/4_shap.png              what drives each prediction
    out/figures/5_overall_score.png     predicted vs physiotherapist overall score
    out/figures/example_card.html       sensor -> error -> cue in 4 languages

Usage: python -m skilltwin.charts
"""

import html
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skilltwin.config import OUT_DIR  # noqa: E402
from skilltwin.expert import build_band, load_inputs  # noqa: E402
from skilltwin.score import dtw_batch  # noqa: E402

FIG_DIR = OUT_DIR / "figures"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_SEQ = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CURVE_LABELS = {"knee": "Knee bend", "thigh": "Thigh rotation", "shank": "Shin rotation",
                "uarm": "Upper-arm rotation"}
RULE_LABELS = {"squat_depth": "Squat depth", "pushup_depth": "Push-up depth",
               "movement_fluidity": "Movement fluidity", "tempo": "Tempo consistency",
               "technique": "Technique holds up", "stability": "Hold stability",
               "hip_angle": "Hip line (proxy)", "knee_joint_axis": "Knee alignment (proxy)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def example_set():
    """Held-out set used across the figures: the coach report's squat example."""
    report = json.loads((OUT_DIR / "coach_report.json").read_text(encoding="utf-8"))
    for r in report:
        for c in r["cues"]:
            if (r["exercise"] == "Standard squat" and c["code"] == "SQUAT_DEPTH_LOW"
                    and c["physio_says_error"] and c["evidence"]):
                return r, c
    raise RuntimeError("no squat example in coach_report.json")


def fig_band(reps, curves, names, labels, report, cue):
    s, ex, seg = report["subject"], report["exercise"], cue["evidence"]["segment"]
    band = build_band(reps, curves, names, labels, ex, exclude={s})
    fi = band["curves"].index(seg)
    rep_scores = pd.read_csv(OUT_DIR / "rep_scores.csv")
    r = rep_scores[(rep_scores["subject"] == s) & (rep_scores["exercise"] == ex)]
    rep = int(r.iloc[(r[f"dev_{seg}_delta"] - cue["evidence"]["delta"]).abs().argsort().iloc[0]]["rep"])
    sel = ((reps["subject"] == s) & (reps["exercise"] == ex) & (reps["rep"] == rep)).to_numpy()
    ci = [names.index(c) for c in band["curves"]]
    warped = dtw_batch(curves[sel][:, ci, :], band["mean"], band["sd"])[0, fi]
    m, sd = band["mean"][fi], band["sd"][fi]
    x = np.arange(len(m))
    out = np.abs(warped - m) > 2 * sd

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(x, m - 2 * sd, m + 2 * sd, color=BLUE, alpha=0.14, linewidth=0,
                    label=f"Expert range (mean ± 2 SD, {band['n_sets']} expert people)")
    ax.plot(x, m, color=BLUE, lw=2, label="Expert average")
    ax.plot(x, warped, color=ORANGE, lw=2, label=f"Trainee (subject {s}, rep {rep + 1}, never in training)")
    ax.fill_between(x, warped, m - 2 * sd, where=out & (warped < m), color=ORANGE, alpha=0.25,
                    linewidth=0, interpolate=True)
    ev = cue["evidence"]
    peak = ev["start"] + int(np.argmax(m[ev["start"]:ev["end"] + 1]))
    ax.annotate(f"{abs(ev['delta']):.0f}° below the expert average\nat {ev['start']}–{ev['end']}% "
                f"of the squat", xy=(peak, warped[peak]),
                xytext=(peak + 4, m[peak] - 2 * sd[peak] - 0.55 * (warped[peak] - ax.get_ylim()[0])),
                color=INK, fontsize=10.5, arrowprops=dict(arrowstyle="-", color=INK_2, lw=1))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Progress through one repetition (%)")
    ax.set_ylabel(f"{CURVE_LABELS[seg]} from standing (°)")
    ax.set_title(f"{ex}: trainee vs expert range — {CURVE_LABELS[seg].lower()}")
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, "1_band_vs_trainee.png")


def fig_auc(results):
    res = results[results["model"].isin(["xgboost", "logreg"])].copy()
    order = (res[res["model"] == "xgboost"].sort_values("auc")["rule"]).tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(0.495, -0.55, "chance", color=INK_2, fontsize=9.5, ha="right")
    for model, color, dy, label in [("xgboost", BLUE, 0.13, "XGBoost (primary model)"),
                                    ("logreg", ORANGE, -0.13, "Logistic regression (comparison)")]:
        g = res[res["model"] == model].set_index("rule").loc[order]
        y = np.arange(len(order)) + dy
        ax.hlines(y, g["auc_lo"], g["auc_hi"], color=color, lw=2, alpha=0.55)
        ax.scatter(g["auc"], y, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
    xg = res[res["model"] == "xgboost"].set_index("rule").loc[order]
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([f"{RULE_LABELS[r]}  ({int(xg.loc[r, 'n_errors'])}/{int(xg.loc[r, 'n_sets'])} errors)"
                        for r in order])
    for t, r in zip(ax.get_yticklabels(), order):
        t.set_color(INK if xg.loc[r, "auc_lo"] > 0.5 else MUTED)
    ax.set_xlim(0.15, 1.02)
    ax.set_xlabel("AUC on people the model never saw (dot) with 95% CI (line)")
    ax.set_title("Which physiotherapist rules the sensors can predict")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    save(fig, "2_rule_auc.png")


def fig_confusion(results, rules):
    fig, axes = plt.subplots(1, len(rules), figsize=(4.4 * len(rules), 4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", BLUE_SEQ)
    for ax, rule in zip(np.atleast_1d(axes), rules):
        r = results[(results["rule"] == rule) & (results["model"] == "xgboost")].iloc[0]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=int)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                frac = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]}\n({frac:.0%})", ha="center", va="center", fontsize=12,
                        color=SURFACE if frac > 0.55 else INK)
        ax.set_xticks([0, 1], ["Predicted OK", "Predicted error"])
        ax.set_yticks([0, 1], ["Physio: OK", "Physio: error"])
        ax.grid(False)
        ax.set_title(f"{RULE_LABELS[rule]}\nbalanced acc. {r['bal_acc']:.2f} · F1 {r['f1']:.2f}",
                     fontsize=11.5)
    fig.suptitle("Held-out confusion matrices (each person tested by a model trained without them)",
                 x=0.02, ha="left", fontsize=12, color=INK_2)
    fig.tight_layout()
    save(fig, "3_confusion.png")


FEATURE_LABELS = {
    "shank_max_mean": "Max shin rotation", "shank_rom_mean": "Shin range of motion",
    "thigh_max_mean": "Max thigh rotation", "thigh_rom_mean": "Thigh range of motion",
    "knee_max_mean": "Max knee bend", "knee_rom_mean": "Knee range of motion",
    "t_bottom_mean": "Timing of lowest point", "uarm_rom_mean": "Upper-arm range of motion",
    "uarm_max_mean": "Max upper-arm rotation", "uarm_max_cv": "Upper-arm depth variability",
    "out_uarm": "Upper arm outside expert band", "z_uarm": "Upper arm vs expert (SD)",
    "z_thigh": "Thigh vs expert (SD)", "z_shank": "Shin vs expert (SD)", "z_knee": "Knee vs expert (SD)",
    "ldlj_shank_mean": "Shin jerkiness (LDLJ)", "ldlj_thigh_mean": "Thigh jerkiness (LDLJ)",
    "ldlj_uarm_mean": "Upper-arm jerkiness (LDLJ)", "speed_thigh_cv": "Thigh speed variability",
    "speed_uarm_cv": "Upper-arm speed variability", "dur_cv": "Rep-time variability",
    "dur_mean": "Rep duration", "dist": "Distance from expert band",
    "speed_thigh_mean": "Thigh peak speed", "speed_uarm_mean": "Upper-arm peak speed",
    "env_peak_triceps_mean": "Triceps peak EMG", "env_peak_biceps_mean": "Biceps peak EMG",
    "out_thigh": "Thigh outside expert band", "out_shank": "Shin outside expert band",
    "out_knee": "Knee outside expert band",
}


def fig_shap(shap, rules):
    fig, axes = plt.subplots(1, len(rules), figsize=(4.6 * len(rules), 3.6))
    for ax, rule in zip(np.atleast_1d(axes), rules):
        g = shap[shap["rule"] == rule].head(5).iloc[::-1]
        y = np.arange(len(g))
        ax.barh(y, g["mean_abs_shap"], height=0.6, color=BLUE)
        ax.set_yticks(y, [FEATURE_LABELS.get(f, f) for f in g["feature"]], fontsize=10)
        for yi, v in zip(y, g["mean_abs_shap"]):
            ax.text(v, yi, f" {v:.2f}", va="center", fontsize=9.5, color=INK_2)
        ax.set_xlim(0, g["mean_abs_shap"].max() * 1.25)
        ax.set_title(RULE_LABELS[rule], fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean |SHAP| (log-odds)")
    fig.suptitle("What the model looks at — top features per rule", x=0.02, ha="left",
                 fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, "4_shap.png")


def fig_overall(pred, overall):
    d = pred[pred["table"] == "reps"]
    o = overall[overall["table"] == "reps"].iloc[0]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 10], [0, 10], color=MUTED, lw=1, ls="--")
    ax.scatter(d["overall"] + rng.uniform(-0.15, 0.15, len(d)), d["predicted"], s=40, color=BLUE,
               alpha=0.75, edgecolor=SURFACE, linewidth=1)
    ax.text(0.04, 0.96, f"Spearman {o['spearman']:.2f}\nmean error {o['mae_model']:.2f} points\n"
            f"(predicting the average: {o['mae_mean_baseline']:.2f})", transform=ax.transAxes,
            va="top", fontsize=10.5, color=INK)
    ax.set_xlim(1.5, 10.5)
    ax.set_ylim(1.5, 10.5)
    ax.set_xlabel("Physiotherapist overall score (0–10)")
    ax.set_ylabel("Sensor model prediction, held out")
    ax.set_title(f"Overall quality, {len(d)} repetitive sets")
    save(fig, "5_overall_score.png")


def example_card(report, cue):
    names = {"en": "English", "ja": "日本語", "vi": "Tiếng Việt", "hi": "हिन्दी"}
    rows = "".join(f"<tr><th>{names[k]}</th><td lang='{k}'>{html.escape(v)}</td></tr>"
                   for k, v in cue["text"].items())
    ev = cue["evidence"]
    page = f"""<!doctype html><meta charset="utf-8"><title>Skill Twin example</title>
<style>
body{{margin:0;padding:32px 16px;background:#fcfcfb;color:#0b0b0b;
  font:15px/1.5 "Segoe UI","Yu Gothic","Nirmala UI",system-ui,sans-serif}}
main{{max-width:880px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} p.sub{{color:#52514e;margin:0 0 24px}}
ol{{display:flex;flex-wrap:wrap;gap:12px;padding:0;list-style:none;margin:0 0 24px}}
ol li{{flex:1 1 180px;border:1px solid #e6e5e1;border-radius:8px;padding:12px}}
ol b{{display:block;color:#2a78d6;font-size:12px;letter-spacing:.04em;text-transform:uppercase}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;vertical-align:top;
  padding:10px 12px;border-top:1px solid #e6e5e1}} th{{width:110px;color:#52514e;font-weight:600}}
img{{max-width:100%;border:1px solid #e6e5e1;border-radius:8px;margin-top:24px}}
</style>
<main>
<h1>Skill Twin — one trainee, end to end</h1>
<p class="sub">Subject {report['subject']} · {report['exercise']} · this person was never in the training data</p>
<ol>
<li><b>1 · Sensors</b>EMG on 4 muscles + IMU on upper arm, thigh and shin</li>
<li><b>2 · Expert range</b>Compared with expert sets (score ≥ 8, no errors)</li>
<li><b>3 · Error detected</b>{cue['code']} · model probability {cue['probability']:.2f} ·
  physiotherapist also marked this error: {'yes' if cue['physio_says_error'] else 'no'}</li>
<li><b>4 · Evidence</b>{CURVE_LABELS[ev['segment']]} {abs(ev['delta']):.0f}°
  {'below' if ev['delta'] < 0 else 'above'} expert average at {ev['start']}–{ev['end']}% of the movement</li>
</ol>
<table>{rows}</table>
<img src="1_band_vs_trainee.png" alt="Expert range versus the trainee's repetition">
</main>"""
    (FIG_DIR / "example_card.html").write_text(page, encoding="utf-8")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    reps, curves, names, labels = load_inputs()
    results = pd.read_csv(OUT_DIR / "classifier_results.csv")
    shap = pd.read_csv(OUT_DIR / "shap_importance.csv")
    validated = results[(results["model"] == "xgboost") & (results["auc_lo"] > 0.5)]["rule"].tolist()
    report, cue = example_set()

    fig_band(reps, curves, names, labels, report, cue)
    fig_auc(results)
    fig_confusion(results, validated)
    fig_shap(shap, ["squat_depth", "pushup_depth", "movement_fluidity"])
    fig_overall(pd.read_csv(OUT_DIR / "overall_predictions.csv"),
                pd.read_csv(OUT_DIR / "overall_results.csv"))
    example_card(report, cue)
    print("figures:", sorted(p.name for p in FIG_DIR.iterdir()))


if __name__ == "__main__":
    main()
