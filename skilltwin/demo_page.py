"""Build the judges' demo page: every number and curve comes from out/.

    skilltwin/demo_template.html + out/* -> out/demo/skill_twin.html

The page is self-contained (data embedded), so it works offline at the venue.
Usage (after run_all and verify):  python -m skilltwin.demo_page
"""

import json

import numpy as np
import pandas as pd

from skilltwin import quat
from skilltwin.config import OBSERVABILITY, OUT_DIR, PROJECT_DIR, QUAT
from skilltwin.expert import build_band, load_inputs
from skilltwin.score import dtw_batch

STEP = 2                                  # keep every 2nd point of the 0-100 % curves
SHOW_RULES = ["squat_depth", "pushup_depth", "movement_fluidity", "technique", "tempo", "stability"]


def r(x, nd=1):
    """Round arrays/scalars for JSON; NaN -> None."""
    a = np.asarray(x, dtype=float)
    out = np.where(np.isfinite(a), np.round(a, nd), np.nan)
    return [None if np.isnan(v) else float(v) for v in out.ravel()] if a.ndim else (
        None if np.isnan(out) else float(out))


def raw_signal():
    c = pd.read_parquet(OUT_DIR / "clean.parquet",
                        filters=[("subject", "==", 1), ("exercise", "==", "Standard squat")])
    q = c[[f"q_gastroc_{k}" for k in QUAT]].to_numpy(float)
    ref = quat.mean(q[(c["phase"] < 3).to_numpy() & (c["rep"] >= 0).to_numpy()])
    shin = quat.angle_between(ref, q)
    k = 5                                  # 100 Hz -> 20 Hz for the page
    starts = c[c["rep"] >= 0].groupby("rep")["t"].min().round(2).tolist()
    return {"t": r(c["t"].to_numpy()[::k], 2), "phase": r(c["phase"].to_numpy()[::k]),
            "shin": r(shin[::k]), "rectus": r(c["env_rectus"].to_numpy()[::k], 3),
            "gastroc": r(c["env_gastroc"].to_numpy()[::k], 3), "rep_starts": starts}


def set_explorer():
    reps, curves, names, labels = load_inputs()
    oof = pd.read_csv(OUT_DIR / "oof_predictions.csv")
    scores = pd.read_csv(OUT_DIR / "set_scores.csv")
    coach = {(c["subject"], c["exercise"]): c["cues"]
             for c in json.loads((OUT_DIR / "coach_report.json").read_text(encoding="utf-8"))}
    out = {}
    for ex in sorted(reps["exercise"].unique()):
        ex_out = {}
        for s in sorted(reps.loc[reps["exercise"] == ex, "subject"].unique()):
            sel = ((reps["exercise"] == ex) & (reps["subject"] == s) & reps["valid"]).to_numpy()
            if not sel.any():
                continue
            band = build_band(reps, curves, names, labels, ex, exclude={s})
            ci = [names.index(c) for c in band["curves"]]
            x = curves[sel][:, ci, :]
            have = ~np.isnan(x).any(axis=(0, 2))
            trainee = np.full(band["mean"].shape, np.nan)
            if have.any():
                m, sd = band["mean"][have], band["sd"][have]
                trainee[have] = np.median(dtw_batch(x[:, have, :], m, sd), axis=0)
            lab = labels[(labels["subject"] == s) & (labels["exercise"] == ex)].iloc[0]
            sc = scores[(scores["subject"] == s) & (scores["exercise"] == ex)]
            p = oof[(oof["subject"] == s) & (oof["exercise"] == ex)]
            ex_out[str(s)] = {
                "mean": [r(v[::STEP]) for v in band["mean"]],
                "sd": [r(v[::STEP]) for v in band["sd"]],
                "trainee": [r(v[::STEP]) for v in trainee],
                "n_reps": int(sel.sum()), "band_people": band["n_sets"],
                "overall": int(lab["overall"]), "n_errors": int(lab["n_errors"]),
                "is_expert": bool(lab["is_expert"]),
                "rules": {k: int(lab[k]) for k in SHOW_RULES if pd.notna(lab.get(k))},
                "p": {row["rule"]: r(row["p_xgb"], 2) for _, row in p.iterrows()},
                "score": r(sc["score"].iloc[0]) if len(sc) else None,
                "cues": coach.get((s, ex), []),
            }
        out[ex] = {"curves": band["curves"], "subjects": ex_out}
    return out


def live_series(sets, fs_out=20):
    """Full time series of every set at fs_out Hz, for the live replay.

    Angles use the same start-of-repetition reference as features.py; phase is the
    dataset's own 0-100 % marker, so each sample can be compared with the expert
    band at the same moment of the movement. Invalid repetitions get rep = -1.
    """
    from skilltwin.features import REF_PHASE, angle_curves
    clean = pd.read_parquet(OUT_DIR / "clean.parquet")
    reps = pd.read_csv(OUT_DIR / "reps.csv")
    step = 100 // fs_out
    out = {}
    for (s, ex), g in clean.groupby(["subject", "exercise"], observed=True):
        if ex not in sets or str(s) not in sets[ex]["subjects"]:
            continue
        g = g.reset_index(drop=True)
        ref = ((g["phase"] < REF_PHASE) & (g["rep"] >= 0)).to_numpy()
        ang, _ = angle_curves(g, ref)
        valid = reps.loc[(reps["subject"] == s) & (reps["exercise"] == ex) & reps["valid"], "rep"]
        rep = g["rep"].where(g["rep"].isin(valid), -1).to_numpy()[::step]
        curves = {}
        for c in sets[ex]["curves"]:
            x = ang[c] if c in ang else g[c].to_numpy(float)
            x = x[::step]
            curves[c] = r(x, 1 if c in ang else 3) if np.isfinite(x).any() else None
        out.setdefault(ex, {})[str(s)] = {"fs": fs_out, "phase": r(g["phase"].to_numpy()[::step]),
                                          "rep": [int(v) for v in rep], "curves": curves}
    return out


HOLD_SEGMENTS = {"hold_thigh": "rectus", "hold_shank": "gastroc", "hold_uarm": "biceps"}


def hold_sets(labels, fs_out=20, n_band=51):
    """Hold exercises: how far each segment moves away from its held position.

    For every hold set and segment, the angle between each sample's orientation
    and the set's average orientation (deg). The expert range is built from the
    expert holds (same tiers as the repetitions, this person excluded) on a
    0-100 % time axis; more movement than the range is flagged.
    """
    from skilltwin.config import EXERCISES
    from skilltwin.expert import select_experts
    holds = [e for e, (_, mode, _) in EXERCISES.items() if mode == "hold"]
    clean = pd.read_parquet(OUT_DIR / "clean.parquet",
                            filters=[("exercise", "in", holds)])
    step = 100 // fs_out
    per = {}
    for (s, ex), g in clean.groupby(["subject", "exercise"], observed=True):
        if len(g) < 500:                                   # under 5 s of data
            continue
        series, c51 = {}, {}
        for c, m in HOLD_SEGMENTS.items():
            q = g[[f"q_{m}_{k}" for k in QUAT]].to_numpy(float)
            if np.isnan(q).any():
                continue
            dev = quat.angle_between(quat.mean(q), q)
            series[c] = dev[::step]
            c51[c] = np.interp(np.linspace(0, len(dev) - 1, n_band), np.arange(len(dev)), dev)
        if series:
            per.setdefault(ex, {})[int(s)] = (series, c51)
    names = list(HOLD_SEGMENTS)
    sets, live = {}, {}
    for ex, subj in per.items():
        ex_sets = {}
        for s, (series, _) in subj.items():
            experts, _ = select_experts(labels, ex, exclude={s}, sets_with_reps=list(subj))
            x = np.array([[subj[e][1].get(c, np.full(n_band, np.nan)) for c in names] for e in experts])
            with np.errstate(all="ignore"):
                mean, sd = np.nanmean(x, axis=0), np.maximum(np.nanstd(x, axis=0), 0.5)
            lab = labels[(labels["subject"] == s) & (labels["exercise"] == ex)].iloc[0]
            n = len(next(iter(series.values())))
            ex_sets[str(s)] = {
                "mean": [r(v) for v in mean], "sd": [r(v) for v in sd],
                "n_reps": 1, "band_people": len(experts), "overall": int(lab["overall"]),
                "n_errors": int(lab["n_errors"]), "is_expert": bool(lab["is_expert"]),
                "rules": {k: int(lab[k]) for k in ("stability", "technique") if pd.notna(lab.get(k))},
                "p": {}, "score": None, "cues": []}
            live.setdefault(ex, {})[str(s)] = {
                "fs": fs_out, "phase": r(np.linspace(0, 100, n)), "rep": [0] * n,
                "curves": {c: (r(series[c]) if c in series else None) for c in names}}
        sets[ex] = {"curves": names, "mode": "hold", "subjects": ex_sets}
    return sets, live


def catalog():
    """Physiotherapist's checklist per exercise, with what the sensors can see."""
    cat = pd.read_csv(OUT_DIR / "rule_catalog.csv")
    return {ex: g[["rule", "observable"]].values.tolist() for ex, g in cat.groupby("exercise")}


MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webm": "video/webm"}
WIKI_FOR_COACH = 3000      # characters of the Wikipedia article given to the AI coach


def skills():
    """Skill guides, credited Commons media and Wikipedia background, per skill family.

    Small files are embedded so the published page shows them; files marked
    local_only are referenced from media/ and only appear when served locally.
    """
    import base64
    from skilltwin.skill_guide import FAMILY, GUIDE, WIKIPEDIA, WITH_WEIGHT
    media_dir = OUT_DIR / "demo" / "media"
    manifest = json.loads((media_dir / "manifest.json").read_text(encoding="utf-8"))
    wiki = json.loads((OUT_DIR / "skill_wiki.json").read_text(encoding="utf-8"))
    out = {}
    for fam, guide in GUIDE.items():
        items = []
        for m in manifest.get(fam, []):
            path = media_dir / m["file"]
            if m["local_only"]:
                src = "media/" + m["file"]
            else:
                src = f"data:{MIME[path.suffix]};base64," + base64.b64encode(path.read_bytes()).decode()
            items.append({k: m[k] for k in ("kind", "title", "author", "license", "license_url", "source", "local_only")}
                         | {"src": src})
        w = wiki.get(WIKIPEDIA[fam], {})
        out[fam] = {"guide": guide, "media": items, "with_weight": fam in WITH_WEIGHT,
                    "wiki": {"title": w.get("title", ""), "url": w.get("url", ""),
                             "text": w.get("text", "")[:WIKI_FOR_COACH]}}
    return {"skills": out, "skillOf": FAMILY}


def results():
    res = pd.read_csv(OUT_DIR / "classifier_results.csv")
    res = res[res["model"] != "always_no_error"]
    cols = ["rule", "model", "observable", "n_sets", "n_errors", "auc", "auc_lo", "auc_hi",
            "bal_acc", "f1", "tn", "fp", "fn", "tp"]
    rules = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
             for row in res[cols].to_dict("records")]
    for row in rules:
        for k in ("n_sets", "n_errors", "tn", "fp", "fn", "tp"):
            row[k] = int(row[k])
    pred = pd.read_csv(OUT_DIR / "overall_predictions.csv").query("table == 'reps'")
    overall = pd.read_csv(OUT_DIR / "overall_results.csv").set_index("table").loc["reps"]
    shap = pd.read_csv(OUT_DIR / "shap_importance.csv")
    return {
        "rules": rules,
        "overall": {"mae": r(overall["mae_model"], 2), "base": r(overall["mae_mean_baseline"], 2),
                    "rho": r(overall["spearman"], 2), "n": int(overall["n_sets"]),
                    "points": [[int(a), r(b, 2), e, int(s)] for a, b, e, s in
                               pred[["overall", "predicted", "exercise", "subject"]].values]},
        "shap": {rule: [[f, r(v, 2)] for f, v in g.head(5)[["feature", "mean_abs_shap"]].values]
                 for rule, g in shap.groupby("rule") if rule in ("squat_depth", "pushup_depth",
                                                                   "movement_fluidity")},
    }


def verification():
    checks = pd.read_csv(OUT_DIR / "verification_report.csv")
    checks["detail"] = checks["detail"].str.replace(r"np\.float64\(([^)]*)\)", r"\1", regex=True)
    perm = json.loads((OUT_DIR / "verification_permutation.json").read_text())
    labels = pd.read_csv(OUT_DIR / "labels.csv")
    return {"checks": checks.to_dict("records"), "perm": perm,
            "n_subjects": int(labels["subject"].nunique()), "n_sets": len(labels),
            "observability": {lvl: sorted(k for k, (l, _) in OBSERVABILITY.items() if l == lvl)
                              for lvl in ("yes", "partial", "no")}}


def main():
    from skilltwin.coach import CUES
    sets = set_explorer()
    live = live_series(sets)
    for ex in sets:
        sets[ex]["mode"] = "reps"
    h_sets, h_live = hold_sets(pd.read_csv(OUT_DIR / "labels.csv"))
    sets.update(h_sets)
    live.update(h_live)
    code, texts = CUES["stability"]
    data = {"raw": raw_signal(), "sets": sets, "live": live, "catalog": catalog(),
            "holdCue": {"code": code, "text": texts}, **skills(), **results(), "verify": verification()}
    page = (PROJECT_DIR / "demo_template.html").read_text(encoding="utf-8")
    page = page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out = OUT_DIR / "demo" / "skill_twin.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
