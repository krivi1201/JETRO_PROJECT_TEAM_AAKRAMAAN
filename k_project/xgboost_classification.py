
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from xgboost import XGBClassifier


# --------------------------------------------------
# PATHS
# --------------------------------------------------

# Resolved relative to this file so the project works from any folder.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES_FILE = OUTPUT_DIR / "window_features.csv"
LABELS_FILE = OUTPUT_DIR / "overall_expertise.csv"

PREDICTIONS_FILE = OUTPUT_DIR / "xgboost_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "xgboost_metrics.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "xgboost_feature_importance.csv"


# --------------------------------------------------
# LOAD EXISTING OUTPUTS
# --------------------------------------------------

for file_path in [FEATURES_FILE, LABELS_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find {file_path}")

features = pd.read_csv(FEATURES_FILE)
labels = pd.read_csv(LABELS_FILE)

features["participant_folder"] = (
    features["participant_folder"]
    .astype(str).str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(2)
)

labels["participant_folder"] = (
    labels["participant_folder"]
    .astype(str).str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(2)
)

target_col = "overall_inferred_status"

if target_col not in labels.columns:
    raise ValueError(
        f"Expected target column '{target_col}' in "
        "overall_expertise.csv"
    )

if labels["participant_folder"].duplicated().any():
    raise ValueError(
        "Expected exactly one label row per participant."
    )


# --------------------------------------------------
# CREATE PARTICIPANT-LEVEL FEATURES
# Each participant contributes one observation.
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

feature_cols = []

for col in features.columns:
    if col in excluded_exact:
        continue

    if not pd.api.types.is_numeric_dtype(features[col]):
        continue

    if any(part in col.lower() for part in excluded_parts):
        continue

    if col.endswith(excluded_suffixes):
        continue

    feature_cols.append(col)

if not feature_cols:
    raise ValueError("No usable numeric features were found.")

# Aggregate windows into one row per participant.
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

if data["participant_folder"].nunique() < 6:
    raise ValueError(
        "Too few participants for meaningful evaluation."
    )

# Use binary labels: 1 = experienced across all shapes;
# 0 = does not meet the strict all-shapes criterion.
data["target"] = (
    data[target_col]
    .eq("Experienced across all four shapes (inferred)")
    .astype(int)
)

X = data[feature_cols]
y = data["target"]

class_counts = y.value_counts().sort_index()

if len(class_counts) != 2:
    raise ValueError(
        "Both target classes must be present."
    )

print("=" * 65)
print("XGBOOST: PARTICIPANT-LEVEL CLASSIFICATION")
print("=" * 65)
print(f"Participants: {len(data)}")
print(f"Numeric features: {len(feature_cols)}")
print(f"Inferred non-expert group (0): {int((y == 0).sum())}")
print(f"Inferred all-shapes expert group (1): {int((y == 1).sum())}")


# --------------------------------------------------
# MODEL
# No scaling is required for XGBoost, but median
# imputation is fitted separately inside each fold.
# --------------------------------------------------

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("xgb", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        random_state=42,
        n_jobs=1,
    )),
])


# --------------------------------------------------
# LEAVE-ONE-PARTICIPANT-OUT EVALUATION
# Every participant is held out once.
# --------------------------------------------------

loo = LeaveOneOut()

predicted = cross_val_predict(
    model,
    X,
    y,
    cv=loo,
    method="predict",
    n_jobs=1,
)

# Cross-validated metrics on participant-level predictions.
accuracy = accuracy_score(y, predicted)
balanced_accuracy = balanced_accuracy_score(y, predicted)
f1 = f1_score(y, predicted, zero_division=0)

cm = confusion_matrix(y, predicted, labels=[0, 1])

print("\nCross-validated results:")
print(f"Accuracy:          {accuracy:.3f}")
print(f"Balanced accuracy: {balanced_accuracy:.3f}")
print(f"F1-score:          {f1:.3f}")

print("\nConfusion matrix (rows=true, columns=predicted):")
print("Labels: 0 = not expert across all shapes; 1 = inferred expert")
print(cm)

print("\nClassification report:")
print(classification_report(
    y,
    predicted,
    labels=[0, 1],
    target_names=[
        "Not expert across all shapes (inferred)",
        "Experienced across all shapes (inferred)",
    ],
    zero_division=0,
))


# --------------------------------------------------
# SAVE CROSS-VALIDATED PREDICTIONS
# --------------------------------------------------

predictions = data[
    ["participant_folder", target_col, "target"]
].copy()

predictions["predicted_target"] = predicted
predictions["predicted_status"] = np.where(
    predicted == 1,
    "Experienced across all four shapes (inferred)",
    "Not expert across all four shapes",
)

predictions.to_csv(PREDICTIONS_FILE, index=False)


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

metrics = pd.DataFrame([
    {"metric": "participants", "value": len(data)},
    {"metric": "features", "value": len(feature_cols)},
    {"metric": "accuracy", "value": accuracy},
    {"metric": "balanced_accuracy", "value": balanced_accuracy},
    {"metric": "f1_score", "value": f1},
])

metrics.to_csv(METRICS_FILE, index=False)


# --------------------------------------------------
# FIT FINAL MODEL FOR DESCRIPTIVE FEATURE IMPORTANCE
# This is not the cross-validation evaluation model.
# --------------------------------------------------

model.fit(X, y)

xgb_model = model.named_steps["xgb"]

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv(IMPORTANCE_FILE, index=False)

print("\nTop 15 features from the full-data fit:")
print(importance.head(15).to_string(index=False))

print("\nSaved files:")
print(PREDICTIONS_FILE)
print(METRICS_FILE)
print(IMPORTANCE_FILE)

print(
    "\nIMPORTANT: The target labels are inferred from clustering. "
    "These metrics measure reproduction of those labels, "
    "not validated real-world expertise."
)