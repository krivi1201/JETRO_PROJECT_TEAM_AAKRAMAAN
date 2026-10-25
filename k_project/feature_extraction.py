
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_all_recordings
from preprocessing import preprocess_all_recordings


# Resolved relative to this file so the script works from any folder.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COLUMN = "Time[s]"

SENSOR_COLUMNS = [
    "X", "Y", "Z",
    "RX", "RY", "RZ", "RW",
    "FX", "FY", "FZ",
    "TX", "TY", "TZ",
]

# FFT features require a sufficiently regular sampling interval.
MIN_SAMPLES_FOR_SPECTRAL = 32
MAX_RELATIVE_INTERVAL_DEVIATION = 0.10


def safe_float(value):
    """Convert finite numeric results to floats; otherwise return NaN."""
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def statistical_features(values, prefix):
    """Calculate descriptive features for one sensor channel."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {}

    diff = np.diff(x)
    mean = np.mean(x)
    std = np.std(x)

    q25, q75 = np.percentile(x, [25, 75])

    centered = x - mean
    if std > 0:
        skewness = np.mean((centered / std) ** 3)
        kurtosis = np.mean((centered / std) ** 4) - 3.0
    else:
        skewness = 0.0
        kurtosis = 0.0

    return {
        f"{prefix}_mean": safe_float(mean),
        f"{prefix}_std": safe_float(std),
        f"{prefix}_variance": safe_float(np.var(x)),
        f"{prefix}_min": safe_float(np.min(x)),
        f"{prefix}_max": safe_float(np.max(x)),
        f"{prefix}_range": safe_float(np.max(x) - np.min(x)),
        f"{prefix}_median": safe_float(np.median(x)),
        f"{prefix}_iqr": safe_float(q75 - q25),
        f"{prefix}_mean_absolute": safe_float(np.mean(np.abs(x))),
        f"{prefix}_rms": safe_float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_skewness": safe_float(skewness),
        f"{prefix}_kurtosis": safe_float(kurtosis),
        f"{prefix}_mean_absolute_difference": (
            safe_float(np.mean(np.abs(diff))) if len(diff) else np.nan
        ),
        f"{prefix}_zero_crossing_rate": (
            safe_float(np.mean(np.diff(np.signbit(x)) != 0))
            if len(x) > 1 else np.nan
        ),
    }


def spectral_features(values, sampling_rate, prefix, timestamps):
    """
    Calculate FFT-based features only when the timestamps are
    sufficiently regular for a conventional FFT.
    """
    x = np.asarray(values, dtype=float)
    t = np.asarray(timestamps, dtype=float)

    result = {
        f"{prefix}_dominant_frequency_hz": np.nan,
        f"{prefix}_spectral_centroid_hz": np.nan,
        f"{prefix}_spectral_power": np.nan,
    }

    if len(x) < MIN_SAMPLES_FOR_SPECTRAL:
        return result

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)):
        return result

    dt = np.diff(t)
    if len(dt) == 0 or np.any(dt <= 0):
        return result

    median_dt = np.median(dt)

    if median_dt <= 0:
        return result

    # Reject FFT features if sampling intervals vary by more than 10%
    # relative to the median interval.
    relative_deviation = np.abs(dt - median_dt) / median_dt

    # Allow moderate timestamp jitter, but reject windows
    # containing substantial gaps.
    if np.max(dt) > max(10.0 * median_dt, 0.100):
        return result

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        return result

    centered = x - np.mean(x)
    tapered = centered * np.hanning(len(centered))

    spectrum = np.abs(np.fft.rfft(tapered)) ** 2
    frequencies = np.fft.rfftfreq(
        len(tapered),
        d=1.0 / sampling_rate,
    )

    # Exclude the zero-frequency component.
    if len(spectrum) <= 1:
        return result

    spectrum = spectrum[1:]
    frequencies = frequencies[1:]

    total_power = np.sum(spectrum)
    if total_power <= 0 or not np.isfinite(total_power):
        return result

    dominant_index = int(np.argmax(spectrum))

    result.update({
        f"{prefix}_dominant_frequency_hz": safe_float(
            frequencies[dominant_index]
        ),
        f"{prefix}_spectral_centroid_hz": safe_float(
            np.sum(frequencies * spectrum) / total_power
        ),
        f"{prefix}_spectral_power": safe_float(total_power),
    })

    return result


def extract_window_features(window):
    """Extract features from one preprocessed window dictionary."""
    df = window["data"].copy()

    times = df[TIME_COLUMN].to_numpy(dtype=float)
    dt = np.diff(times)
    positive_dt = dt[dt > 0]

    if len(times) < 2 or len(positive_dt) == 0:
        return None

    median_dt = float(np.median(positive_dt))
    sampling_rate = 1.0 / median_dt

    features = {
        "window_id": window["window_id"],
        "segment_id": window.get("segment_id", 0),
        "start_sample": window["start_sample"],
        "start_time": window["start_time"],
        "end_time": window["end_time"],
        "duration_s": float(times[-1] - times[0]),
        "n_samples": len(df),
        "estimated_sampling_rate_hz": sampling_rate,
        "median_interval_ms": median_dt * 1000.0,
        "max_interval_ms": (
            float(np.max(positive_dt)) * 1000.0
        ),
    }

    # A window should not contain a gap that escaped preprocessing.
    features["max_interval_to_median_ratio"] = (
        float(np.max(positive_dt) / median_dt)
    )

    for column in SENSOR_COLUMNS:
        if column not in df.columns:
            continue

        values = df[column].to_numpy(dtype=float)

        features.update(
            statistical_features(values, column)
        )

        features.update(
            spectral_features(
                values,
                sampling_rate,
                column,
                times,
            )
        )

    return features


def extract_all_features(processed):
    """Build a feature table from already-preprocessed recordings."""
    rows = []

    for recording_id, result in processed.items():
        cleaned_data = result["cleaned_data"]

        # Recover metadata from the loaded recording.
        participant_folder = str(
            cleaned_data["participant_folder"].iloc[0]
            if "participant_folder" in cleaned_data.columns
            else recording_id.split("_")[0]
        )

        object_type = str(
            cleaned_data["object_type"].iloc[0]
            if "object_type" in cleaned_data.columns
            else recording_id.split("_", 1)[-1]
        )

        # First valid timestamp of the cleaned recording. Used to express
        # window times as "seconds since the recording started".
        recording_t0 = float(cleaned_data["Time[s]"].iloc[0])

        for window in result["windows"]:
            row = extract_window_features(window)

            if row is None:
                continue

            row.update({
                "recording_id": recording_id,
                "participant_folder": participant_folder,
                "object_type": object_type,
                # Elapsed-time columns contain "time", so every model
                # script excludes them from the feature set.
                "start_time_rel_s": row["start_time"] - recording_t0,
                "end_time_rel_s": row["end_time"] - recording_t0,
            })

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    # Load each recording once for this run.
    recordings = load_all_recordings()

    # Preprocess the in-memory recordings; no second loader call.
    processed, preprocessing_summary = preprocess_all_recordings(
        recordings
    )

    features = extract_all_features(processed)

    output_path = OUTPUT_DIR / "window_features.csv"
    features.to_csv(output_path, index=False)

    preprocessing_summary.to_csv(
        OUTPUT_DIR / "preprocessing_summary.csv",
        index=False,
    )

    print("\nFEATURE EXTRACTION SUMMARY")
    print(f"Recordings: {features['recording_id'].nunique()}")
    print(f"Windows with features: {len(features):,}")
    print(f"Feature table shape: {features.shape}")
    print(f"Unique object types: {features['object_type'].nunique()}")

    numeric_columns = features.select_dtypes(
        include=[np.number]
    ).columns

    missing_by_column = features[numeric_columns].isna().sum()
    missing_by_column = missing_by_column[
        missing_by_column > 0
    ]

    print(
        "Rows containing missing numeric features:",
        int(features[numeric_columns].isna().any(axis=1).sum()),
    )

    if not missing_by_column.empty:
        print("\nColumns with missing values:")
        print(missing_by_column.to_string())

    print(f"\nSaved feature table: {output_path}")


if __name__ == "__main__":
    main()