# JETRO — SURPTech Coaching Dashboard

A sensor-based coaching dashboard for the **SURPTech (SURP) dataset** of wood-shaping and sanding recordings. The project analyzes participant recordings across four tasks—**box, cylinder, sphere, and wooden corners**—to identify sensor patterns that differ from those of participants inferred to be experienced, and to present time-specific coaching feedback.

> Example feedback: **Box task, 0:11–0:14 (sharpest at 0:12.6):** pressure changed suddenly. Start and finish each stroke gently instead of pushing in or lifting off abruptly.

## Dataset

This project expects SURP/SURPTech recordings in the following layout:

```text
data/
├── 01/
│   ├── box.pts
│   ├── cylinder.pts
│   ├── sphere.pts
│   └── wooden_corners.pts
├── 02/
│   └── ...
└── 21/
    └── ...
```

Each participant folder should contain one `.pts` recording for each of the four tasks. The code expects the recording format described by the project’s data loader, including the `Time[s]` time column.

**Dataset availability:** The `data/` folder is not included in this repository. Place the authorized SURP/SURPTech dataset in the expected layout before running the analysis. Do not upload restricted, private, or personally identifiable recordings to a public repository.

**Testing note:** The README for this project reports that the pipeline was tested using a synthetic dataset matching the expected format (21 participants × 4 recordings, 14 channels, with an offset `Time[s]` column) and with mistakes injected at known times. This is a format/pipeline test; it should not be interpreted as a claim that the reported injected-error test was performed on real SURP recordings.

## Features

- Sensor-data preprocessing and feature extraction.
- XGBoost-based classification and comparison with other classifiers.
- Clustering-based analysis of participant experience, inferred from sensor variability.
- Time-localized detection of sensor patterns that differ from the experienced-participant reference.
- Coaching feedback generation using Ollama and `gemma3:4b`.
- Dashboard views for coaching feedback, session timeline, cohort, and methods.
- Outputs for timed moments, window metrics, reference statistics, and feedback metadata.

## How timing analysis works

1. Recordings are cleaned and divided into 2-second windows with a 1-second step, following `preprocessing.py`.
2. Seven behaviours are measured per window: pressure level, pressure steadiness, sudden pressure changes, twisting steadiness, movement speed, motion smoothness, and rotation of the piece.
3. Each window is compared with windows from participants inferred to be experienced on the same task. The participant being analyzed is excluded from their own reference.
4. A window is flagged when the robust z-score reaches the configured threshold (`|z| ≥ 3.5`) and the difference is at least 25%. Neighbouring flagged windows are combined into a timed moment.
5. Reported times are minutes and seconds from the first valid sample of the recording.

These moments indicate that sensor measurements differed strongly from the reference at that time; **they are not confirmed mistakes**. Experience labels are inferred from sensor variability and are not verified skill ratings. Real recordings may be noisier, so review flagged moments against the session timeline.

Settings can be adjusted near the top of `timing_analysis.py`.

## Requirements

- Python version compatible with the packages listed in `requirements.txt`.
- Ollama is optional and required only for LLM-generated coaching feedback.
- The Ollama model used by the project is `gemma3:4b`.

Install dependencies from the project root:

```powershell
pip install -r requirements.txt
```

## Quick start on Windows

1. Put the authorized dataset in `data/01` through `data/21`, using the structure shown above.
2. Double-click `install.bat` (first-time setup).
3. Double-click `start_dashboard.bat`.
4. Open `http://127.0.0.1:8000` in your browser.
5. Select **Run analysis** and wait for the pipeline to finish.
6. For AI feedback, install and start [Ollama](https://ollama.com/), then run:

   ```powershell
   ollama pull gemma3:4b
   ```

   Use **Write feedback** in the dashboard to generate feedback for a participant.

## Command-line usage

Run these commands from the project root:

```powershell
pip install -r requirements.txt
python run_pipeline.py
python app.py
python ollama_feedback.py 07
```

- `python run_pipeline.py` runs the analysis pipeline.
- `python app.py` starts the dashboard.
- `python ollama_feedback.py 07` generates feedback for participant `07` in the terminal (requires Ollama and the model).

## Important files

| File | Purpose |
|---|---|
| `app.py` | FastAPI dashboard server |
| `run_pipeline.py` | Runs the analysis pipeline |
| `data_loader.py` | Loads the sensor recordings |
| `preprocessing.py` | Cleans and prepares the recordings |
| `feature_extraction.py` | Extracts sensor features and relative timing columns |
| `xgboost_classification.py` | XGBoost classification |
| `compare_classifiers.py` | Compares classifiers |
| `clustering.py` | Clustering-based analysis |
| `overall_expertise.py` | Infers experience across the four tasks |
| `timing_analysis.py` | Finds and summarizes time-localized sensor deviations |
| `explain.py` | Model explanations |
| `ollama_feedback.py` | Generates time-aware coaching feedback |
| `web/` | Dashboard interface |
| `requirements.txt` | Python dependencies |

## Generated outputs

The pipeline may create these files under `outputs/`:

- `timing_moments.csv` — one row per timed moment.
- `timing_window_metrics.csv` — per-window metrics and z-scores.
- `timing_reference.csv` — reference statistics used for each case.
- `timing_summary.json` — timing-analysis settings and counts.
- `feedback_meta.json` — feedback metadata, including measured moments and any generated times that were not measured.
- `saved_llm_answers.json` — saved participant feedback.

Exact outputs depend on which pipeline steps have been run.

## Limitations and responsible interpretation

- Experience is inferred from sensor variability, not from independently verified skill ratings.
- A flagged moment is a strong deviation from the reference, not proof of an error.
- Validate results on real SURP/SURPTech recordings before making performance claims.
- Generated feedback should be reviewed for correctness and usefulness.
- Only share dataset files when you have permission to distribute them.

## Repository structure

```text
k_project/
├── app.py
├── clustering.py
├── compare_classifiers.py
├── data_loader.py
├── explain.py
├── feature_extraction.py
├── ollama_feedback.py
├── overall_expertise.py
├── preprocessing.py
├── run_pipeline.py
├── timing_analysis.py
├── xgboost_classification.py
├── requirements.txt
├── web/
├── outputs/
└── README.md
```

The `data/` folder is intentionally omitted from this example because the dataset is not included in the repository.
