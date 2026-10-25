"""
run_pipeline.py

Runs the whole analysis from the raw data in data/01 ... data/21:

    1. feature_extraction.py      window features      (required)
    2. overall_expertise.py       inferred labels      (required)
    3. clustering.py              participant clusters (optional)
    4. xgboost_classification.py  LOPO XGBoost         (optional)
    5. compare_classifiers.py     model comparison     (optional)
    6. explain.py                 SHAP explanations    (required)
    7. timing_analysis.py         timed moments        (required)

Optional steps may fail (for example compare_classifiers.py needs at
least five participants in each class); the run continues and reports it.

Usage:  python run_pipeline.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("feature_extraction.py", True),
    ("overall_expertise.py", True),
    ("clustering.py", False),
    ("xgboost_classification.py", False),
    ("compare_classifiers.py", False),
    ("explain.py", True),
    ("timing_analysis.py", True),
]


def main():
    started = time.time()
    results = []
    print("=" * 65, flush=True)
    print("JETRO ANALYSIS PIPELINE", flush=True)
    print("=" * 65, flush=True)

    for index, (script, required) in enumerate(STEPS, start=1):
        print(f"\n>>> STEP {index}/{len(STEPS)}: {script}"
              f"{'' if required else ' (optional)'}", flush=True)
        step_start = time.time()
        completed = subprocess.run(
            [sys.executable, "-u", str(BASE_DIR / script)],
            cwd=str(BASE_DIR),
            env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg"),
        )
        ok = completed.returncode == 0
        results.append((script, required, ok, time.time() - step_start))
        print(f"<<< {script}: {'OK' if ok else 'FAILED'} "
              f"({time.time() - step_start:.1f} s)", flush=True)
        if not ok and required:
            print(f"\nPIPELINE STOPPED: required step {script} failed.", flush=True)
            sys.exit(1)

    print("\n" + "=" * 65, flush=True)
    print("PIPELINE SUMMARY", flush=True)
    print("=" * 65, flush=True)
    for script, required, ok, seconds in results:
        status = "OK" if ok else ("FAILED (optional, skipped)" if not required else "FAILED")
        print(f"  {script:<28} {status:<28} {seconds:6.1f} s", flush=True)
    print(f"\nFinished in {time.time() - started:.1f} s.", flush=True)
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
