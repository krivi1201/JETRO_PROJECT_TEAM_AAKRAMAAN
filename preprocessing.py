import numpy as np
import pandas as pd


# A gap larger than this multiple of the typical interval starts a new segment.
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 1.0
GAP_THRESHOLD_MULTIPLIER = 5.0

# Avoid windows that are too small to support reliable time-series features.
MIN_SAMPLES_PER_WINDOW = 32


def clean_recording(df):
    """Clean one recording without joining data across timestamp gaps."""
    df = df.copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Time[s]"])
    df = df.sort_values("Time[s]").drop_duplicates(
        subset=["Time[s]"], keep="first"
    )

    sensor_columns = [
        c for c in df.columns
        if c not in {
            "Time[s]", "recording_id",
            "participant_folder", "object_type"
        }
    ]

    df[sensor_columns] = df[sensor_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=sensor_columns).reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("Recording has too few valid samples.")

    return df


def split_at_time_gaps(df):
    """Split only at substantial timestamp gaps."""
    times = df["Time[s]"].to_numpy(dtype=float)
    intervals = np.diff(times)

    positive_intervals = intervals[intervals > 0]

    if len(positive_intervals) == 0:
        return [df.copy()]

    median_interval = float(np.median(positive_intervals))

    # Require BOTH a relative and an absolute gap threshold.
    relative_threshold = 10.0 * median_interval
    absolute_threshold = 0.100  # 100 milliseconds

    gap_threshold = max(relative_threshold, absolute_threshold)

    split_positions = np.where(intervals > gap_threshold)[0] + 1

    boundaries = np.concatenate(
        ([0], split_positions, [len(df)])
    )

    segments = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        segment = df.iloc[start:end].copy().reset_index(drop=True)

        if len(segment) >= 2:
            segments.append(segment)

    return segments


def make_windows(df, window_seconds=None, stride_seconds=None):
    """
    Create duration-based windows within one continuous segment.
    Returns dictionaries containing each window's metadata and data.
    """
    if window_seconds is None:
        window_seconds = WINDOW_SECONDS

    if stride_seconds is None:
        stride_seconds = STRIDE_SECONDS

    if len(df) < MIN_SAMPLES_PER_WINDOW:
        return []

    times = df["Time[s]"].to_numpy(dtype=float)
    intervals = np.diff(times)
    intervals = intervals[intervals > 0]

    if len(intervals) == 0:
        return []

    median_interval = float(np.median(intervals))
    estimated_rate = 1.0 / median_interval

    # Convert durations to approximate sample counts for this segment.
    window_samples = max(
        MIN_SAMPLES_PER_WINDOW,
        int(round(window_seconds * estimated_rate))
    )
    stride_samples = max(
        1,
        int(round(stride_seconds * estimated_rate))
    )

    # Do not produce a window whose actual duration is too short.
    windows = []
    window_number = 0

    for start in range(
        0, len(df) - window_samples + 1, stride_samples
    ):
        win = df.iloc[start:start + window_samples].copy()
        actual_duration = float(
            win["Time[s]"].iloc[-1] - win["Time[s]"].iloc[0]
        )

        # Duration-based validation, with a small tolerance for sampling.
        if actual_duration < window_seconds * 0.90:
            continue

        windows.append({
            "window_id": window_number,
            "start_sample": start,
            "start_time": float(win["Time[s]"].iloc[0]),
            "end_time": float(win["Time[s]"].iloc[-1]),
            "n_samples": len(win),
            "estimated_sampling_rate_hz": estimated_rate,
            "data": win.reset_index(drop=True),
        })
        window_number += 1

    return windows


def preprocess_recording(df):
    """Clean a recording, split gaps, and window each segment."""
    cleaned = clean_recording(df)
    segments = split_at_time_gaps(cleaned)

    windows = []
    for segment_id, segment in enumerate(segments):
        segment_windows = make_windows(segment)

        for window in segment_windows:
            window["segment_id"] = segment_id
            windows.append(window)

    return {
        "cleaned_data": cleaned,
        "segments": segments,
        "windows": windows,
    }


def preprocess_all_recordings(recordings):
    """
    Accept recordings already loaded by data_loader.py.
    No data-loader call is made inside this function.
    """
    processed = {}
    summary = []

    for recording_id, df in recordings.items():
        result = preprocess_recording(df)
        processed[recording_id] = result

        summary.append({
            "recording_id": recording_id,
            "cleaned_samples": len(result["cleaned_data"]),
            "continuous_segments": len(result["segments"]),
            "windows": len(result["windows"]),
        })

    summary_df = pd.DataFrame(summary)

    print("\nPREPROCESSING SUMMARY")
    print(f"Recordings processed: {len(processed)}")
    print(f"Continuous segments: {summary_df['continuous_segments'].sum()}")
    print(f"Windows created: {summary_df['windows'].sum()}")
    print(
        "Recordings with zero windows:",
        int((summary_df["windows"] == 0).sum())
    )

    return processed, summary_df

if __name__ == "__main__":
    from data_loader import load_all_recordings

    recordings = load_all_recordings()
    processed, summary = preprocess_all_recordings(recordings)

    from pathlib import Path

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "preprocessing_summary.csv",
        index=False
    )

    print("\nSummary saved to outputs/preprocessing_summary.csv")
    print("\nFirst 10 recordings:")
    print(summary.head(10).to_string(index=False))