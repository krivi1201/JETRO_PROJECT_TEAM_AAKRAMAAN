
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# PATHS
# --------------------------------------------------

# Resolved relative to this file so the project works from any folder.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES_FILE = OUTPUT_DIR / "window_features.csv"

PARTICIPANT_OUTPUT = OUTPUT_DIR / "participant_clusters.csv"
WINDOW_OUTPUT = OUTPUT_DIR / "window_clusters.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "cluster_summary.csv"


# --------------------------------------------------
# LOAD EXISTING FEATURES ONLY
# --------------------------------------------------

df = pd.read_csv(FEATURES_FILE)

required = {"participant_folder", "object_type"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["participant_folder"] = (
    df["participant_folder"].astype(str).str.zfill(2)
)

# Use sensor standard-deviation features as a proxy
# for movement and force/torque variability.
std_features = [
    col for col in df.columns
    if col.endswith("_std")
    and pd.api.types.is_numeric_dtype(df[col])
]

if len(std_features) < 3:
    raise ValueError(
        "Not enough numeric *_std features were found. "
        "Check the columns in window_features.csv."
    )

print(f"Windows loaded: {len(df)}")
print(f"Variability features selected: {len(std_features)}")


# --------------------------------------------------
# PARTICIPANT-LEVEL PROFILES
# Every participant contributes one row.
# --------------------------------------------------

participant_features = (
    df.groupby("participant_folder")[std_features]
    .median()
    .sort_index()
)

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(participant_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# --------------------------------------------------
# TWO-CLUSTER K-MEANS
# --------------------------------------------------

model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=50
)

raw_labels = model.fit_predict(X_scaled)

# A transparent variability score:
# average standardized variability across selected features.
variability_score = X_scaled.mean(axis=1)

cluster_scores = pd.DataFrame({
    "cluster": raw_labels,
    "variability_score": variability_score
}).groupby("cluster")["variability_score"].mean()

experienced_cluster = int(cluster_scores.idxmin())
novice_cluster = int(cluster_scores.idxmax())

# Explicitly label the clusters using the stated heuristic.
label_map = {
    experienced_cluster: "Experienced (inferred)",
    novice_cluster: "Novice (inferred)"
}

participants = participant_features.copy()
participants["cluster"] = raw_labels
participants["cluster_name"] = [
    label_map[int(label)] for label in raw_labels
]
participants["variability_score"] = variability_score

participants = participants.reset_index()


# --------------------------------------------------
# ADD PCA COORDINATES FOR VISUALIZATION
# --------------------------------------------------

from sklearn.decomposition import PCA

if X_scaled.shape[1] >= 2 and len(X_scaled) >= 2:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    participants["PCA_1"] = coords[:, 0]
    participants["PCA_2"] = coords[:, 1]

    explained = pca.explained_variance_ratio_.sum()
else:
    participants["PCA_1"] = np.nan
    participants["PCA_2"] = np.nan
    explained = np.nan


# --------------------------------------------------
# MAP PARTICIPANT LABELS BACK TO EACH WINDOW
# --------------------------------------------------

label_lookup = participants[
    ["participant_folder", "cluster", "cluster_name",
     "variability_score"]
]

window_results = df.merge(
    label_lookup,
    on="participant_folder",
    how="left",
    validate="many_to_one"
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

summary = (
    participants.groupby(["cluster", "cluster_name"])
    .agg(
        participants=("participant_folder", "count"),
        mean_variability_score=("variability_score", "mean"),
        median_variability_score=("variability_score", "median")
    )
    .reset_index()
)

# Save results.
participants.to_csv(PARTICIPANT_OUTPUT, index=False)
window_results.to_csv(WINDOW_OUTPUT, index=False)
summary.to_csv(SUMMARY_OUTPUT, index=False)


# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("INFERRED NOVICE / EXPERIENCED CLUSTERING")
print("=" * 60)

print(f"\nParticipants: {len(participants)}")
print(f"Features used: {len(std_features)}")
print(f"PCA variance explained by first two components: "
      f"{explained:.2%}" if not np.isnan(explained)
      else "PCA variance explained: unavailable")

print("\nCluster summary:")
print(summary.to_string(index=False))

print("\nParticipant assignments:")
print(
    participants[
        ["participant_folder", "cluster",
         "cluster_name", "variability_score"]
    ]
    .sort_values(["cluster_name", "participant_folder"])
    .to_string(index=False)
)

print("\nFiles saved:")
print(PARTICIPANT_OUTPUT)
print(WINDOW_OUTPUT)
print(SUMMARY_OUTPUT)

print(
    "\nIMPORTANT: These experience labels are inferred from "
    "sensor variability, not verified against actual skill labels."
)