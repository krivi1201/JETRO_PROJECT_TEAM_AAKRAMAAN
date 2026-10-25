"""Step 9 - Turn detected errors into coaching cues in the trainee's language.

The error is decided by the models and the expert band, never by the LLM:
    1. a rule's cue fires only if that rule's held-out model is validated
       (95% CI of AUC above 0.5) and the set's held-out probability >= 0.5
    2. the evidence comes from the expert band (where / how much) or from the
       rule's key feature versus expert sets
    3. the cue text comes from the approved library below
    4. optionally a local LLM (Ollama) rephrases it; the result is rejected if it
       changes any number, and the approved text is used instead

    out/coach_report.json  cues for every held-out set

Usage: python -m skilltwin.coach [--llm MODEL]   e.g. --llm qwen2.5:3b
Translations are drafts and need a native-speaker check before real use.
"""

import argparse
import json
import re

import pandas as pd
import requests

from skilltwin.config import OUT_DIR

LANGS = ["en", "ja", "vi", "hi"]
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

CUES = {
    "squat_depth": ("SQUAT_DEPTH_LOW", {
        "en": "Lower your hips until your thighs are level with your knees.",
        "ja": "太ももが膝と水平になるまで、腰をもっと下げましょう。",
        "vi": "Hạ hông xuống cho đến khi đùi ngang bằng với đầu gối.",
        "hi": "अपने कूल्हों को तब तक नीचे लाएँ जब तक आपकी जाँघें घुटनों के बराबर न हो जाएँ।"}),
    "pushup_depth": ("PUSHUP_DEPTH_LOW", {
        "en": "Lower your chest at least to elbow height on every push-up.",
        "ja": "毎回、胸が肘の高さに来るまで体を下ろしましょう。",
        "vi": "Mỗi lần hít đất, hạ ngực xuống ít nhất ngang tầm khuỷu tay.",
        "hi": "हर पुश-अप में अपनी छाती को कम से कम कोहनी की ऊँचाई तक नीचे लाएँ।"}),
    "movement_fluidity": ("MOVEMENT_JERKY", {
        "en": "Move smoothly: control the way down and the way up without jerks or pauses.",
        "ja": "動きをなめらかに。下ろす時も上げる時も、反動や止まりを作らずにコントロールしましょう。",
        "vi": "Chuyển động nhịp nhàng: kiểm soát cả lúc xuống và lúc lên, không giật hay dừng đột ngột.",
        "hi": "सहज गति से करें: नीचे जाते और ऊपर आते समय बिना झटके या रुकावट के नियंत्रण रखें।"}),
    "tempo": ("TEMPO_INCONSISTENT", {
        "en": "Keep the same pace on every repetition.",
        "ja": "毎回同じテンポで動きましょう。",
        "vi": "Giữ nhịp độ giống nhau ở mỗi lần lặp lại.",
        "hi": "हर दोहराव में एक जैसी गति बनाए रखें।"}),
    "technique": ("TECHNIQUE_DETERIORATES", {
        "en": "Your form changes as you tire. Rest briefly, or do fewer repetitions at the same quality.",
        "ja": "疲れるとフォームが崩れています。少し休むか、同じ質を保てる回数に減らしましょう。",
        "vi": "Tư thế của bạn thay đổi khi mệt. Hãy nghỉ ngắn hoặc giảm số lần nhưng giữ chất lượng.",
        "hi": "थकने पर आपकी तकनीक बदल रही है। थोड़ा आराम करें या कम दोहराव करें, पर गुणवत्ता वही रखें।"}),
    "stability": ("HOLD_UNSTABLE", {
        "en": "Hold still: brace your core and keep the position without shaking.",
        "ja": "体幹に力を入れて、ぐらつかずに姿勢をキープしましょう。",
        "vi": "Giữ yên: siết cơ bụng và giữ tư thế không rung lắc.",
        "hi": "स्थिर रहें: कोर को कसें और बिना हिले स्थिति बनाए रखें।"}),
    "hip_angle": ("HIP_LINE", {
        "en": "Keep your body in one straight line from shoulders to heels.",
        "ja": "肩からかかとまで体を一直線に保ちましょう。",
        "vi": "Giữ cơ thể thành một đường thẳng từ vai đến gót chân.",
        "hi": "कंधों से एड़ियों तक शरीर को एक सीधी रेखा में रखें।"}),
    "knee_joint_axis": ("KNEE_VALGUS", {
        "en": "Keep your knees in line with your toes; do not let them cave inward.",
        "ja": "膝はつま先と同じ方向に向け、内側に入らないようにしましょう。",
        "vi": "Giữ đầu gối thẳng hướng mũi chân, đừng để đầu gối khuỵu vào trong.",
        "hi": "घुटनों को पंजों की सीध में रखें; उन्हें अंदर की ओर न मुड़ने दें।"}),
}

SEGMENT_NAMES = {
    "knee":  {"en": "knee", "ja": "膝", "vi": "đầu gối", "hi": "घुटने"},
    "thigh": {"en": "thigh", "ja": "太もも", "vi": "đùi", "hi": "जाँघ"},
    "shank": {"en": "shin", "ja": "すね", "vi": "cẳng chân", "hi": "पिंडली"},
    "uarm":  {"en": "upper arm", "ja": "上腕", "vi": "cánh tay trên", "hi": "ऊपरी बाँह"},
}
BAND_EVIDENCE = {
    "en": "At {start}–{end}% of the movement your {segment} angle is {amount:.0f}° {dir} the expert average.",
    "ja": "動作の{start}〜{end}%で、{segment}の角度が熟練者の平均より{amount:.0f}°{dir}。",
    "vi": "Ở {start}–{end}% của động tác, góc {segment} {dir} mức trung bình của chuyên gia {amount:.0f}°.",
    "hi": "गति के {start}–{end}% पर आपका {segment} कोण विशेषज्ञ औसत से {amount:.0f}° {dir} है।",
}
DIRECTION = {  # (below, above) the expert average
    "en": ("below", "above"), "ja": ("小さいです", "大きいです"),
    "vi": ("nhỏ hơn", "lớn hơn"), "hi": ("कम", "अधिक"),
}
METRIC_EVIDENCE = {
    "en": "{metric}: you {value}, experts {expert}.",
    "ja": "{metric}：あなた {value}、熟練者 {expert}。",
    "vi": "{metric}: bạn {value}, chuyên gia {expert}.",
    "hi": "{metric}: आप {value}, विशेषज्ञ {expert}।",
}
# rule -> (feature, format, metric label per language) for non-band evidence
METRICS = {
    "movement_fluidity": ("ldlj_shank_mean", "{:.1f}", {
        "en": "Smoothness score (higher is smoother)", "ja": "なめらかさスコア（高いほど良い）",
        "vi": "Điểm độ mượt (càng cao càng mượt)", "hi": "सहजता स्कोर (अधिक = अधिक सहज)"}),
    "tempo": ("dur_cv", "{:.0%}", {
        "en": "Variation in repetition time", "ja": "1回ごとの時間のばらつき",
        "vi": "Độ dao động thời gian mỗi lần", "hi": "हर दोहराव के समय में अंतर"}),
    "technique": ("knee_max_trend", "{:+.0%}", {
        "en": "Change in depth from first to last repetition", "ja": "最初から最後までの深さの変化",
        "vi": "Thay đổi độ sâu từ lần đầu đến lần cuối", "hi": "पहले से आखिरी दोहराव तक गहराई में बदलाव"}),
}
BAND_SEGMENTS = {"squat_depth": ["knee", "thigh", "shank"], "pushup_depth": ["uarm"]}


def validated_rules():
    res = pd.read_csv(OUT_DIR / "classifier_results.csv")
    ok = res[(res["model"] == "xgboost") & (res["auc_lo"] > 0.5)]
    return dict(zip(ok["rule"], ok["auc"]))


def band_evidence(rule, rep_scores, subject, exercise):
    """Worst stretch (most negative median z) among the rule's band curves."""
    r = rep_scores[(rep_scores["subject"] == subject) & (rep_scores["exercise"] == exercise)]
    segs = [s for s in BAND_SEGMENTS.get(rule, []) if f"z_{s}" in r and r[f"z_{s}"].notna().any()]
    if not segs:
        return None
    seg = min(segs, key=lambda s: r[f"z_{s}"].median())
    d = r.dropna(subset=[f"dev_{seg}_delta"])
    if d.empty:
        return None
    return {"type": "band", "segment": seg, "start": int(d[f"dev_{seg}_start"].median()),
            "end": int(d[f"dev_{seg}_end"].median()), "delta": float(d[f"dev_{seg}_delta"].median())}


def metric_evidence(rule, sets, labels, subject, exercise):
    if rule not in METRICS:
        return None
    feat, fmt, _ = METRICS[rule]
    row = sets[(sets["subject"] == subject) & (sets["exercise"] == exercise)]
    experts = labels[(labels["exercise"] == exercise) & labels["is_expert"]
                     & (labels["subject"] != subject)]["subject"]
    ref = sets[(sets["exercise"] == exercise) & sets["subject"].isin(experts)][feat]
    if row.empty or ref.empty or pd.isna(row[feat].iloc[0]):
        return None
    return {"type": "metric", "rule": rule, "value": fmt.format(row[feat].iloc[0]),
            "expert": fmt.format(ref.median())}


def render(evidence, lang):
    if evidence is None:
        return ""
    if evidence["type"] == "band":
        return BAND_EVIDENCE[lang].format(
            start=evidence["start"], end=evidence["end"], amount=abs(evidence["delta"]),
            dir=DIRECTION[lang][evidence["delta"] > 0],
            segment=SEGMENT_NAMES[evidence["segment"]][lang])
    label = METRICS[evidence["rule"]][2][lang]
    return METRIC_EVIDENCE[lang].format(metric=label, value=evidence["value"], expert=evidence["expert"])


LANG_NAMES = {"en": "English", "ja": "Japanese", "vi": "Vietnamese", "hi": "Hindi"}
SYSTEM_PROMPT = (
    "You rewrite workplace coaching cues so they sound natural and encouraging to a trainee. "
    "Rules: keep the exact meaning; keep every number exactly as written; add no new numbers, "
    "advice, steps or facts; write only in the requested language; reply with the rewritten "
    "cue only, no quotes or explanations.")


def signed_numbers(text):
    """Numbers with their sign, so "-17.0" and "17.0" differ (a dropped minus flips the meaning).

    A minus directly before a digit counts as a sign unless it follows a digit
    (as in the range "44-79").
    """
    text = text.replace("−", "-").replace("–", "-")
    return sorted(float(m) for m in re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?", text))


def llm_rephrase(text, lang, model):
    """Ask a local LLM to make the approved cue friendlier. Returns None if unusable.

    The guard rejects any output that drops a number or introduces a new one
    (in a first test, a model appended invented advice: "Maintain this angle for 6 seconds").
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Language: {LANG_NAMES[lang]}\nCue: {text}"}]
    try:
        r = requests.post(OLLAMA_CHAT_URL, timeout=120, json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.2}})
        out = r.json().get("message", {}).get("content", "").strip().strip('"「」')
    except (requests.RequestException, ValueError):
        return None
    numbers_in, numbers_out = signed_numbers(text), signed_numbers(out)
    if (not out or sorted(numbers_in) != sorted(numbers_out)
            or not 0.5 <= len(out) / max(len(text), 1) <= 1.8):
        return None
    return out


def build_report(llm_model=None):
    rules = validated_rules()
    oof = pd.read_csv(OUT_DIR / "oof_predictions.csv")
    rep_scores = pd.read_csv(OUT_DIR / "rep_scores.csv")
    sets = pd.read_csv(OUT_DIR / "set_features.csv")
    labels = pd.read_csv(OUT_DIR / "labels.csv")
    report = []
    for (s, ex), g in oof[oof["rule"].isin(rules)].groupby(["subject", "exercise"]):
        cues = []
        for _, p in g[g["p_xgb"] >= 0.5].sort_values("p_xgb", ascending=False).iterrows():
            code, texts = CUES[p["rule"]]
            ev = band_evidence(p["rule"], rep_scores, s, ex) or metric_evidence(
                p["rule"], sets, labels, s, ex)
            cue = {"code": code, "rule": p["rule"], "probability": round(float(p["p_xgb"]), 2),
                   "physio_says_error": bool(p["error"]), "evidence": ev, "text": {}, "source": {}}
            for lang in LANGS:
                approved = f"{texts[lang]} {render(ev, lang)}".strip()
                rephrased = llm_rephrase(approved, lang, llm_model) if llm_model else None
                cue["text"][lang] = rephrased or approved
                cue["source"][lang] = "llm" if rephrased else "library"
            cues.append(cue)
        report.append({"subject": int(s), "exercise": ex, "cues": cues})
    return report, rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None, help="Ollama model name to rephrase cues (optional)")
    args = ap.parse_args()
    report, rules = build_report(args.llm)
    (OUT_DIR / "coach_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    fired = [c for r in report for c in r["cues"]]
    print(f"validated rules (held-out AUC CI above 0.5): {rules}")
    print(f"sets checked: {len(report)}, cues fired: {len(fired)}, "
          f"of which the physiotherapist also marked an error: {sum(c['physio_says_error'] for c in fired)}")
    example = next((r for r in report if r["exercise"] == "Standard squat"
                    and any(c["code"] == "SQUAT_DEPTH_LOW" and c["physio_says_error"] for c in r["cues"])),
                   None)
    if example:
        print(f"\nexample - subject {example['subject']}, {example['exercise']} (never seen in training):")
        for c in example["cues"]:
            print(f"  {c['code']}  p={c['probability']}  physio error={c['physio_says_error']}")
            for lang in LANGS:
                print(f"    [{lang}/{c['source'][lang]}] {c['text'][lang]}")


if __name__ == "__main__":
    main()
