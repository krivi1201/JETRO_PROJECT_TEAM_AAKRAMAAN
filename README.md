# JETRO · SURPTech coaching dashboard

Sensor-based coaching for wood shaping and sanding. For every participant and
every task (box, cylinder, sphere, wooden corners) the dashboard shows **what
to improve and when in the recording it happened**, for example:

> Box task, 0:11.0–0:14.0 (sharpest at 0:12.6): pressure changed suddenly.
> Start and finish each stroke gently instead of pushing in or lifting off abruptly.

## Quick start (Windows)

1. Put the data in `data/01` … `data/21`; each folder holds `box.pts`,
   `cylinder.pts`, `sphere.pts` and `wooden_corners.pts`.
2. Double-click **install.bat** (first time only).
3. Double-click **start_dashboard.bat**. The browser opens at
   http://127.0.0.1:8000.
4. Press **Run analysis** on the first screen (about a minute per 84
   recordings; the log is shown live).
5. For AI feedback, start Ollama and install the model once:
   `ollama pull gemma3:4b`. Then press **Write feedback** for a participant.

Command-line equivalent:

```
pip install -r requirements.txt
python run_pipeline.py      # analysis
python app.py               # dashboard
python ollama_feedback.py 07   # feedback for one participant in the terminal
```

## What is new

| File | Change |
|---|---|
| `timing_analysis.py` | **New.** Finds when each problem happened (see below). |
| `ollama_feedback.py` | Gives the model the timed moments, asks for a "When it happened" section, and checks every time the model writes against the measured times (`outputs/ollama_feedback_meta.json`). Same model (`gemma3:4b`), same Ollama call; `num_predict` raised 220 → 700 for the longer answer. Accepts the participant on the command line or stdin. |
| `app.py` + `web/` | **New interface** (replaces the Streamlit `app2.py`): FastAPI server + custom web page. Coaching, session timeline, cohort, and method views; runs the pipeline and the feedback from buttons. Feedback is still saved to `outputs/saved_llm_answers.json` in the same `{"01": "answer"}` format without removing other entries. |
| `run_pipeline.py` | **New.** Runs every step in order. |
| `overall_expertise.py` | **Reconstructed** (the original was not supplied). Same K-means variability method as `clustering.py`, run per task; "Experienced across all four shapes (inferred)" only if experienced on all four. Replace it with your original if you have it. |
| `explain.py` | **Bug fix:** timing and bookkeeping columns (`start_time`, `end_time`, `duration_s`, `n_samples`, `start_sample`, sampling-rate columns) were used as model features and could appear in SHAP explanations. They are now excluded, matching the classifier scripts. |
| `xgboost_classification.py`, `compare_classifiers.py` | Same fix for `window_id`, `segment_id`, `start_sample`, `n_samples`; hard-coded `C:\Users\KRIVI ARORA\...` path replaced by the script's own folder. |
| `clustering.py` | Hard-coded path replaced. |
| `feature_extraction.py` | Output path made relative to the script; adds `start_time_rel_s` / `end_time_rel_s` (excluded from all models). |
| `data_loader.py`, `preprocessing.py` | Unchanged. |

## How the time of a mistake is found

1. Each recording is cleaned and cut into 2-second windows (1-second step),
   exactly as in `preprocessing.py`.
2. Seven behaviours are measured per window from the raw sensors: pressure
   level (|F|), pressure steadiness, sudden pressure changes (|d|F|/dt|),
   twisting steadiness (|T|), movement speed (X/Y/Z), smoothness of motion,
   and how much the piece is turned (from the RX/RY/RZ/RW quaternion).
3. Each window is compared with the same task's windows from participants
   labelled experienced on that task, **always excluding the person being
   analysed**, using a robust z-score (median / MAD).
4. A window is flagged at |z| ≥ 3.5 **and** ≥ 25 % difference. Neighbouring
   flagged windows become one moment with a start, an end, and the sharpest
   sample. Patterns covering most of a task are reported as "throughout the
   task" together with their worst stretch.
5. Times are minutes:seconds from the first valid sample of each recording.

Settings are at the top of `timing_analysis.py`.

**Read moments as "the sensors differed strongly from experienced
participants at this time", not as confirmed errors.** Experience labels are
inferred from sensor variability, not verified skill ratings.

## Outputs added

- `outputs/timing_moments.csv` – one row per timed moment
- `outputs/timing_window_metrics.csv` – every window with metrics and z-scores
- `outputs/timing_reference.csv` – the reference used for each case
- `outputs/timing_summary.json` – settings and counts
- `outputs/feedback_meta.json` – per participant: model, time written, which
  moments were given to the model, and any times it wrote that were not measured

## Tested

The whole pipeline and dashboard were run on a synthetic dataset in your
exact file format (21 × 4 recordings, 14 channels, offset `Time[s]`) with
mistakes injected at known times. Every injected mistake was found and its
sharpest point fell inside the injected period (e.g. injected pressure spikes
12–13 s → detected 0:11–0:14, sharpest 0:12.6). Real recordings will be
noisier; check a few moments against the session timeline.
