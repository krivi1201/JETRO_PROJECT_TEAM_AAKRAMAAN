
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES_FILE = OUTPUT_DIR / "window_features.csv"
LABELS_FILE = OUTPUT_DIR / "overall_expertise.csv"

SHAP_VALUES_FILE = OUTPUT_DIR / "explain_shap_oof_values.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "explain_shap_feature_importance.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "explain_shap_predictions.csv"

SUMMARY_PLOT_FILE = OUTPUT_DIR / "explain_shap_summary.png"
BAR_PLOT_FILE = OUTPUT_DIR / "explain_shap_bar.png"

TOP_N_FEATURES = 15
RANDOM_STATE = 42


# --------------------------------------------------
# 2. LOAD EXISTING FEATURES AND LABELS
# --------------------------------------------------


def load_data():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_FILE}\n"
            "Run feature_extraction.py first."
        )

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Label file not found: {LABELS_FILE}\n"
            "Run overall_expertise.py first."
        )

    features_df = pd.read_csv(FEATURES_FILE)
    labels_df = pd.read_csv(LABELS_FILE)

    # Required columns in your existing CSV files.
    if "participant_folder" not in features_df.columns:
        raise ValueError(
            "window_features.csv is missing participant_folder."
        )

    if "participant_folder" not in labels_df.columns:
        raise ValueError(
            "overall_expertise.csv is missing participant_folder."
        )

    # Handle the exact label column in your CSV.
    status_column = "**overall_inferred_status"

    if status_column not in labels_df.columns:
        # Fallback in case the CSV has a different number of leading stars.
        matching_columns = [
            col for col in labels_df.columns
            if "overall_inferred_status" in col
        ]

        if matching_columns:
            status_column = matching_columns[0]
        elif "all_four_shapes_experienced" in labels_df.columns:
            status_column = "all_four_shapes_experienced"
        else:
            raise ValueError(
                "Could not find the overall expertise label column. "
                f"Available columns: {labels_df.columns.tolist()}"
            )

    # Select numeric features and exclude metadata and diagnostic columns.
    excluded_exact = {
        "recording_id",
        "participant_folder",
        "object_type",
        "window_id",
        "segment_id",
        "window_start",
        "window_end",
        "window_duration",
        "sampling_rate",
        "median_interval",
        "max_interval",
        "interval_ratio",
        "overall_expertise",
        "cluster",
        "label",
        "PCA_1",
        "PCA_2",
        # Window bookkeeping produced by feature_extraction.py.
        "start_sample",
        "n_samples",
    }

    # Same timing/diagnostic exclusions as xgboost_classification.py and
    # compare_classifiers.py. Without these, columns such as start_time,
    # end_time, duration_s and estimated_sampling_rate_hz were used as
    # "skill" features and could appear in the SHAP explanations.
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

    numeric_columns = features_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_columns = [
        col for col in numeric_columns
        if col not in excluded_exact
        and not any(part in col.lower() for part in excluded_parts)
        and not any(
            term in col.lower()
            for term in [
                "dominant_frequency",
                "spectral_power",
                "spectral_entropy",
                "spectral_centroid",
                "spectral_energy",
                "spectral",
            ]
        )
    ]

    if not feature_columns:
        raise ValueError("No usable numeric features were found.")

    # Aggregate windows into one median feature vector per participant.
    participant_features = (
        features_df.groupby("participant_folder")[feature_columns]
        .median()
        .reset_index()
    )

    # Convert the existing all-four-shapes criterion into binary labels.
    labels_df = labels_df.copy()
    raw_labels = labels_df[status_column]

    if status_column == "all_four_shapes_experienced":
        if pd.api.types.is_bool_dtype(raw_labels):
            labels_df["target"] = raw_labels.astype(int)
        else:
            normalized = raw_labels.astype(str).str.strip().str.lower()
            labels_df["target"] = normalized.map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
            })
    else:
        normalized = raw_labels.astype(str).str.strip().str.lower()

        labels_df["target"] = normalized.map({
            "experienced across all four shapes (inferred)": 1,
            "experienced across all four shapes": 1,
            "expert": 1,
            "not expert across all four shapes": 0,
            "not expert": 0,
            "not experienced across all four shapes": 0,
            "non-expert": 0,
            "non expert": 0,
        })

        # Accommodate reasonable variations in the status wording.
        for idx in labels_df.index[labels_df["target"].isna()]:
            status = normalized.loc[idx]

            if "not expert" in status or "not experienced" in status:
                labels_df.loc[idx, "target"] = 0
            elif (
                "experienced across all four shapes" in status
                or status == "expert"
            ):
                labels_df.loc[idx, "target"] = 1

    if labels_df["target"].isna().any():
        raise ValueError(
            f"Could not interpret values in '{status_column}'. "
            f"Unique values: {raw_labels.unique().tolist()}"
        )

    if labels_df["participant_folder"].duplicated().any():
        raise ValueError(
            "overall_expertise.csv contains duplicate participants."
        )

    data = participant_features.merge(
        labels_df[["participant_folder", "target"]],
        on="participant_folder",
        how="inner",
        validate="one_to_one",
    )

    data = data.sort_values(
        "participant_folder"
    ).reset_index(drop=True)

    if len(data) < 5:
        raise ValueError(
            f"Only {len(data)} participants matched between the files."
        )

    X = data[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    y = data["target"].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "Both inferred classes must be present in the labels."
        )

    # Drop features that are entirely missing.
    valid_features = X.columns[X.notna().any(axis=0)].tolist()
    X = X[valid_features]

    print(f"\nLabel column used: {status_column}")
    print(f"Participants matched: {len(data)}")
    print(f"Features selected: {len(valid_features)}")

    return data, X, y, valid_features


# --------------------------------------------------
# 3. EXTRACT CLASS-1 SHAP VALUES
# --------------------------------------------------

def get_expert_shap_values(explainer, X_test):
    """
    Return SHAP values for class 1 (inferred expert).

    Supports common SHAP output formats across versions.
    """
    raw_values = explainer.shap_values(X_test)
    values = np.asarray(raw_values) if not isinstance(
        raw_values, list
    ) else None

    if isinstance(raw_values, list):
        # Older SHAP versions return one array per class.
        if len(raw_values) > 1:
            result = np.asarray(raw_values[1])
        else:
            result = np.asarray(raw_values[0])

    elif values.ndim == 3:
        # Common newer format: samples x features x classes.
        if values.shape[-1] == 2:
            result = values[:, :, 1]
        # Alternative format: classes x samples x features.
        elif values.shape[0] == 2:
            result = values[1]
        else:
            raise ValueError(
                f"Unexpected SHAP array shape: {values.shape}"
            )

    elif values.ndim == 2:
        result = values

    else:
        raise ValueError(
            f"Unexpected SHAP output dimensions: {values.shape}"
        )

    result = np.asarray(result)

    if result.ndim == 1:
        result = result.reshape(1, -1)

    if result.shape[0] != X_test.shape[0]:
        raise ValueError(
            "SHAP output sample count does not match input sample count."
        )

    if result.shape[1] != X_test.shape[1]:
        raise ValueError(
            "SHAP output feature count does not match input feature count."
        )

    return result


# --------------------------------------------------
# 4. LEAVE-ONE-PARTICIPANT-OUT SHAP
# --------------------------------------------------

def run_oof_explanations(data, X, y, feature_columns):
    n_participants = len(data)
    n_features = len(feature_columns)

    shap_matrix = np.zeros((n_participants, n_features))
    predicted_probabilities = np.zeros(n_participants)
    predicted_classes = np.zeros(n_participants, dtype=int)

    print("\nRunning leave-one-participant-out SHAP...")
    print(f"Participants: {n_participants}")
    print(f"Features: {n_features}")

    for test_idx in range(n_participants):
        train_mask = np.ones(n_participants, dtype=bool)
        train_mask[test_idx] = False

        X_train = X.iloc[train_mask]
        X_test = X.iloc[[test_idx]]

        y_train = y.iloc[train_mask]

        # Fit imputation on training participants only.
        imputer = SimpleImputer(strategy="median")

        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X_train_imputed, y_train)

        probabilities = model.predict_proba(X_test_imputed)[0]

        # Find the probability column corresponding to class 1.
        expert_column = list(model.classes_).index(1)
        predicted_probabilities[test_idx] = probabilities[expert_column]

        predicted_classes[test_idx] = model.predict(
            X_test_imputed
        )[0]

        # Explain this participant using the model that excluded them.
        explainer = shap.TreeExplainer(model)

        participant_shap = get_expert_shap_values(
            explainer,
            X_test_imputed
        )

        shap_matrix[test_idx, :] = participant_shap[0]

        print(
            f"[{test_idx + 1:02d}/{n_participants}] "
            f"Participant {data.loc[test_idx, 'participant_folder']} "
            f"completed"
        )

    return shap_matrix, predicted_probabilities, predicted_classes


# --------------------------------------------------
# 5. SAVE EXPLANATIONS AND PREDICTIONS
# --------------------------------------------------

def save_results(
    data,
    X,
    y,
    feature_columns,
    shap_matrix,
    predicted_probabilities,
    predicted_classes
):
    # Global importance: mean absolute SHAP value.
    importance = pd.DataFrame({
        "feature": feature_columns,
        "mean_absolute_shap": np.mean(
            np.abs(shap_matrix), axis=0
        ),
        "mean_shap": np.mean(shap_matrix, axis=0),
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False
    ).reset_index(drop=True)

    importance.to_csv(IMPORTANCE_FILE, index=False)

    # Per-participant signed SHAP values.
    shap_df = pd.DataFrame(
        shap_matrix,
        columns=[f"SHAP_{col}" for col in feature_columns]
    )

    shap_df.insert(
        0,
        "participant_folder",
        data["participant_folder"].astype(str).values
    )

    shap_df.insert(1, "true_inferred_label", y.values)
    shap_df.insert(
        2,
        "predicted_label",
        predicted_classes
    )
    shap_df.insert(
        3,
        "predicted_probability_expert",
        predicted_probabilities
    )

    shap_df.to_csv(SHAP_VALUES_FILE, index=False)

    # Predictions in a separate, easy-to-read file.
    predictions = pd.DataFrame({
        "participant_folder": data["participant_folder"].astype(str),
        "true_inferred_label": y.values,
        "predicted_label": predicted_classes,
        "predicted_probability_expert": predicted_probabilities,
        "correct": predicted_classes == y.values,
    })

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    print("\nSaved:")
    print(f"  {IMPORTANCE_FILE}")
    print(f"  {SHAP_VALUES_FILE}")
    print(f"  {PREDICTIONS_FILE}")

    return importance


# --------------------------------------------------
# 6. CREATE SHAP PLOTS
# --------------------------------------------------

def create_plots(X, shap_matrix, importance, feature_columns):
    top_features = importance.head(TOP_N_FEATURES)["feature"].tolist()

    feature_indices = [
        feature_columns.index(feature)
        for feature in top_features
    ]

    # Use participant feature values for plotting.
    # Fill missing display values with feature medians.
    X_display = X.copy()
    X_display = X_display.fillna(X_display.median())

    # SHAP summary beeswarm plot.
    plt.figure()

    shap.summary_plot(
        shap_matrix,
        X_display,
        feature_names=feature_columns,
        max_display=TOP_N_FEATURES,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        SUMMARY_PLOT_FILE,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close("all")

    # Mean absolute SHAP bar plot.
    plot_data = importance.head(TOP_N_FEATURES).sort_values(
        "mean_absolute_shap",
        ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_data["feature"],
        plot_data["mean_absolute_shap"]
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Top Features Explaining Inferred Expertise")
    plt.tight_layout()
    plt.savefig(
        BAR_PLOT_FILE,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close("all")

    print(f"  {SUMMARY_PLOT_FILE}")
    print(f"  {BAR_PLOT_FILE}")


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SHAP EXPLANATION - BALANCED RANDOM FOREST")
    print("=" * 60)

    data, X, y, feature_columns = load_data()

    print(f"\nParticipants: {len(data)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Not expert across all shapes: {(y == 0).sum()}")
    print(f"Experienced across all shapes (inferred): {(y == 1).sum()}")

    shap_matrix, probabilities, predictions = run_oof_explanations(
        data,
        X,
        y,
        feature_columns
    )

    importance = save_results(
        data,
        X,
        y,
        feature_columns,
        shap_matrix,
        probabilities,
        predictions
    )

    create_plots(
        X,
        shap_matrix,
        importance,
        feature_columns
    )

    print("\nTop 15 features by mean absolute SHAP:")
    print(importance.head(TOP_N_FEATURES).to_string(index=False))

    accuracy = np.mean(predictions == y.values)

    print(f"\nLOOCV accuracy: {accuracy:.3f}")
    print("\nexplain.py completed successfully.")


if __name__ == "__main__":
    main()