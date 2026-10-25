
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# --------------------------------------------------
# PATHS
# --------------------------------------------------

# Resolved relative to this file so the project works from any folder.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES_FILE = OUTPUT_DIR / "window_features.csv"
LABELS_FILE = OUTPUT_DIR / "overall_expertise.csv"

RESULTS_FILE = OUTPUT_DIR / "classifier_comparison.csv"


# --------------------------------------------------
# LOAD EXISTING CSV FILES
# --------------------------------------------------

features = pd.read_csv(FEATURES_FILE)
labels = pd.read_csv(LABELS_FILE)

for frame in (features, labels):
    frame["participant_folder"] = (
        frame["participant_folder"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

target_col = "overall_inferred_status"

if target_col not in labels.columns:
    raise ValueError(
        f"Missing target column: {target_col}"
    )

if labels["participant_folder"].duplicated().any():
    raise ValueError(
        "Labels must contain one row per participant."
    )


# --------------------------------------------------
# SELECT SENSOR FEATURES
# Exclude IDs, metadata, timing diagnostics and
# spectral columns, matching previous experiments.
# --------------------------------------------------

excluded_exact = {
    "participant_folder",
    "recording_id",
    "object_type",
    "cluster",
    "cluster_name",
    "variability_score",
    "PCA_1",
    "PCA_2",
    # Window bookkeeping, not sensor behaviour.
    "window_id",
    "segment_id",
    "start_sample",
    "n_samples",
}

excluded_parts = (
    "sampling_rate",
    "median_interval",
    "max_interval",
    "interval_ratio",
    "duration",
    "window_start",
    "window_end",
    "window_index",
    "time",
)

excluded_suffixes = (
    "_dominant_frequency_hz",
    "_spectral_centroid_hz",
    "_spectral_power",
)

feature_cols = [
    col for col in features.columns
    if col not in excluded_exact
    and pd.api.types.is_numeric_dtype(features[col])
    and not any(
        part in col.lower() for part in excluded_parts
    )
    and not col.endswith(excluded_suffixes)
]

if not feature_cols:
    raise ValueError("No numeric sensor features found.")


# --------------------------------------------------
# AGGREGATE TO ONE ROW PER PARTICIPANT
# --------------------------------------------------

participant_features = (
    features.groupby("participant_folder")[feature_cols]
    .median()
    .reset_index()
)

data = participant_features.merge(
    labels[["participant_folder", target_col]],
    on="participant_folder",
    how="inner",
    validate="one_to_one",
)

data = data.dropna(subset=[target_col]).reset_index(drop=True)

data["target"] = (
    data[target_col]
    .eq("Experienced across all four shapes (inferred)")
    .astype(int)
)

X = data[feature_cols]
y = data["target"]

if y.value_counts().min() < 5:
    raise ValueError(
        "At least five participants are needed in each class "
        "for the planned five-fold comparison."
    )

print("=" * 65)
print("CLASSIFIER COMPARISON")
print("=" * 65)
print(f"Participants: {len(data)}")
print(f"Features: {len(feature_cols)}")
print(f"Class 0 (not expert across all shapes): {(y == 0).sum()}")
print(f"Class 1 (inferred expert across all shapes): {(y == 1).sum()}")


# --------------------------------------------------
# DEFINE MODELS
# Imputation is fitted within each validation fold.
# --------------------------------------------------

models = {
    "Majority baseline": DummyClassifier(
        strategy="most_frequent"
    ),

    "Balanced logistic regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            C=0.1,
            max_iter=5000,
            random_state=42,
        )),
    ]),

    "Balanced random forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )),
    ]),

    "Balanced XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=50,
            max_depth=1,
            learning_rate=0.03,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=10,
            scale_pos_weight=9 / 12,
            random_state=42,
            n_jobs=1,
        )),
    ]),
}


# --------------------------------------------------
# REPEATED STRATIFIED CROSS-VALIDATION
# All models use the same folds.
# --------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro",
}

results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    row = {"model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]

        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    results.append(row)

    print(f"\n{name}")
    print(
        "  Balanced accuracy: "
        f"{row['balanced_accuracy_mean']:.3f} "
        f"+/- {row['balanced_accuracy_std']:.3f}"
    )
    print(
        "  Macro F1: "
        f"{row['macro_f1_mean']:.3f} "
        f"+/- {row['macro_f1_std']:.3f}"
    )
    print(
        "  Accuracy: "
        f"{row['accuracy_mean']:.3f} "
        f"+/- {row['accuracy_std']:.3f}"
    )


# --------------------------------------------------
# SAVE COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    "balanced_accuracy_mean",
    ascending=False,
)

results_df.to_csv(RESULTS_FILE, index=False)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

print(f"\nSaved results to:\n{RESULTS_FILE}")

print(
    "\nCAUTION: Repeated cross-validation folds overlap, so the "
    "reported standard deviations are descriptive, not formal "
    "confidence intervals. Results measure prediction of "
    "inferred cluster-derived labels, not verified expertise."
)