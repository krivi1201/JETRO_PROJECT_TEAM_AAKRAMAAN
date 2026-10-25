"""
overall_expertise.py

Creates outputs/overall_expertise.csv, the label file used by
explain.py, xgboost_classification.py and compare_classifiers.py.

NOTE: this file was reconstructed because the original
overall_expertise.py was not part of the project files supplied. It uses
the same method as clustering.py (two-cluster K-means on participant-level
median *_std sensor features; the lower-variability cluster is labelled
"Experienced (inferred)"), but runs it separately for each shape:

    box, cylinder, sphere, wooden_corners

A participant is "Experienced across all four shapes (inferred)" only if
they fall in the experienced cluster for every shape. These are inferred
labels from sensor variability, not verified skill labels.

If you still have your original overall_expertise.py, you can use it
instead; downstream scripts only need the columns
participant_folder and overall_inferred_status.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES_FILE = OUTPUT_DIR / "window_features.csv"
LABELS_FILE = OUTPUT_DIR / "overall_expertise.csv"

SHAPES = ["box", "cylinder", "sphere", "wooden_corners"]

EXPERIENCED = "Experienced (inferred)"
NOVICE = "Novice (inferred)"
OVERALL_EXPERT = "Experienced across all four shapes (inferred)"
OVERALL_NOT = "Not expert across all four shapes"


def std_feature_columns(df):
    return [
        col for col in df.columns
        if col.endswith("_std")
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def cluster_one_shape(shape_windows, std_features):
    """Return a DataFrame: participant_folder, status, variability_score."""
    profiles = (
        shape_windows.groupby("participant_folder")[std_features]
        .median()
        .sort_index()
    )
    # Drop features that are missing for every participant of this shape.
    profiles = profiles.loc[:, profiles.notna().any(axis=0)]

    if len(profiles) < 2 or profiles.shape[1] == 0:
        return pd.DataFrame({
            "participant_folder": profiles.index,
            "status": np.nan,
            "variability_score": np.nan,
        })

    X = SimpleImputer(strategy="median").fit_transform(profiles)
    X = StandardScaler().fit_transform(X)
    score = X.mean(axis=1)

    labels = KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(X)
    cluster_scores = pd.Series(score).groupby(labels).mean()
    experienced_cluster = int(cluster_scores.idxmin())

    return pd.DataFrame({
        "participant_folder": profiles.index,
        "status": [
            EXPERIENCED if int(lab) == experienced_cluster else NOVICE
            for lab in labels
        ],
        "variability_score": score,
    })


def main():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_FILE}. Run feature_extraction.py first."
        )

    df = pd.read_csv(FEATURES_FILE)
    df["participant_folder"] = (
        df["participant_folder"].astype(str).str.strip()
        .str.replace(r"\.0$", "", regex=True).str.zfill(2)
    )

    std_features = std_feature_columns(df)
    if len(std_features) < 3:
        raise ValueError("Not enough *_std features in window_features.csv.")

    participants = sorted(df["participant_folder"].unique())
    result = pd.DataFrame({"participant_folder": participants})

    print("=" * 60)
    print("PER-SHAPE INFERRED EXPERTISE")
    print("=" * 60)

    for shape in SHAPES:
        shape_windows = df[df["object_type"] == shape]
        if shape_windows.empty:
            print(f"{shape}: no windows found")
            result[f"{shape}_status"] = np.nan
            result[f"{shape}_variability_score"] = np.nan
            continue

        shape_result = cluster_one_shape(shape_windows, std_features)
        shape_result = shape_result.rename(columns={
            "status": f"{shape}_status",
            "variability_score": f"{shape}_variability_score",
        })
        result = result.merge(shape_result, on="participant_folder", how="left")

        counts = shape_result[f"{shape}_status"].value_counts().to_dict()
        print(f"{shape}: {counts}")

    status_cols = [f"{shape}_status" for shape in SHAPES]
    result["n_shapes_experienced"] = (
        result[status_cols].eq(EXPERIENCED).sum(axis=1)
    )
    result["all_four_shapes_experienced"] = result["n_shapes_experienced"] == len(SHAPES)
    result["overall_inferred_status"] = np.where(
        result["all_four_shapes_experienced"], OVERALL_EXPERT, OVERALL_NOT
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(LABELS_FILE, index=False)

    print("\nOverall:")
    print(result["overall_inferred_status"].value_counts().to_string())
    print(f"\nSaved: {LABELS_FILE}")
    print(
        "\nIMPORTANT: labels are inferred from sensor variability, "
        "not verified against real skill labels."
    )


if __name__ == "__main__":
    main()
