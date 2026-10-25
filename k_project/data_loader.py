
"""
data_loader.py

Loads SURP .pts recordings from numbered folders:
data/01, data/02, ..., data/21.

Each numbered folder may contain:
    box.pts
    cylinder.pts
    sphere.pts
    wooden_corners.pts

Each recording is kept separate. We do not combine the
time-series rows from different recordings at this stage.
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

EXPECTED_COLUMNS = [
    "X", "Y", "Z", "Time[s]",
    "RX", "RY", "RZ", "RW",
    "FX", "FY", "FZ",
    "TX", "TY", "TZ"
]

RECORDING_FILES = [
    "box.pts",
    "cylinder.pts",
    "sphere.pts",
    "wooden_corners.pts",
]


# ---------------------------------------------------------
# 2. Load one .pts recording
# ---------------------------------------------------------

def load_pts(file_path):
    """Read one .pts file and validate its 14 numeric columns."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    records = []
    skipped_lines = 0

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Ignore blank lines and common comment lines.
            if not line or line.startswith(("#", "//", "%")):
                skipped_lines += 1
                continue

            # Support whitespace- or comma-separated values.
            parts = line.replace(",", " ").split()

            try:
                values = [float(value) for value in parts]
            except ValueError:
                # Skip textual headers; report them in the summary.
                skipped_lines += 1
                continue

            # Numeric rows with the wrong width indicate a format issue.
            if len(values) != len(EXPECTED_COLUMNS):
                raise ValueError(
                    f"Malformed numeric row in {file_path}\n"
                    f"Line {line_number}: expected "
                    f"{len(EXPECTED_COLUMNS)} values, "
                    f"found {len(values)}."
                )

            records.append(values)

    if not records:
        raise ValueError(f"No numeric data found in {file_path}")

    df = pd.DataFrame(records, columns=EXPECTED_COLUMNS)

    # Store metadata separately from the 14 sensor/time channels.
    # Use a unique ID that combines the folder and recording name.
    # Example: 01_box, 02_box, 01_sphere.
    df["recording_id"] = (
        f"{file_path.parent.name}_{file_path.stem}"
    )

    # Preserve the folder name separately.
    df["participant_folder"] = file_path.parent.name

    # Identify which object/task recording this file represents.
    df["object_type"] = file_path.stem

    print(
        f"Loaded {file_path.parent.name}/{file_path.name}: "
        f"{len(df):,} rows"
    )

    if skipped_lines:
        print(f"  Non-data/blank lines skipped: {skipped_lines}")

    return df


# ---------------------------------------------------------
# 3. Discover all recordings in folders 01 through 21
# ---------------------------------------------------------

def find_all_recordings(data_dir=DATA_DIR):
    """
    Find all expected files in numbered subfolders.

    Returns a list of file paths. Missing files are reported,
    but do not prevent other recordings from loading.
    """

    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Data directory not found: {data_dir.resolve()}"
        )

    recording_paths = []
    missing_files = []

    # Explicitly inspect the numbered folders 01 through 21.
    for folder_number in range(1, 22):
        folder_name = f"{folder_number:02d}"
        folder_path = data_dir / folder_name

        if not folder_path.is_dir():
            print(f"WARNING: Folder not found: {folder_path}")
            continue

        for filename in RECORDING_FILES:
            file_path = folder_path / filename

            if file_path.is_file():
                recording_paths.append(file_path)
            else:
                missing_files.append(
                    f"{folder_name}/{filename}"
                )

    return recording_paths, missing_files


# ---------------------------------------------------------
# 4. Load all recordings
# ---------------------------------------------------------

def load_all_recordings(data_dir=DATA_DIR):
    """Load each discovered recording into a dictionary."""

    paths, missing_files = find_all_recordings(data_dir)

    if not paths:
        raise FileNotFoundError(
            f"No expected .pts files found under {data_dir.resolve()}"
        )

    recordings = {}

    print("=" * 65)
    print("SURP DATA LOADING")
    print("=" * 65)
    print(f"Data directory: {Path(data_dir).resolve()}")
    print(f"Recordings found: {len(paths)}")

    for path in paths:
        folder_name = path.parent.name
        object_name = path.stem

        # Unique key: e.g. 01_box, 02_box, 01_sphere.
        recording_id = f"{folder_name}_{object_name}"

        if recording_id in recordings:
            raise ValueError(
                f"Duplicate recording identifier: {recording_id}"
            )

        recordings[recording_id] = load_pts(path)

    if missing_files:
        print("\nWARNING: Some expected files were missing:")
        for filename in missing_files:
            print(f"  - {filename}")
    else:
        print("\nAll four expected files exist in all 21 folders.")

    return recordings


# ---------------------------------------------------------
# 5. Run verification
# ---------------------------------------------------------

if __name__ == "__main__":

    recordings = load_all_recordings()

    print("\n" + "=" * 65)
    print("DATA LOADER VERIFICATION")
    print("=" * 65)

    total_rows = 0

    for recording_id, df in recordings.items():
        total_rows += len(df)

        print(f"\nRecording ID: {recording_id}")
        print(f"Shape: {df.shape}")
        print(f"Sensor columns: {EXPECTED_COLUMNS}")
        print(f"First 2 rows:\n{df.head(2).to_string(index=False)}")

        # Report missing values in the sensor/time columns.
        missing = df[EXPECTED_COLUMNS].isna().sum()
        if missing.any():
            print("WARNING: Missing values found:")
            print(missing[missing > 0].to_string())

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"Successfully loaded recordings: {len(recordings)}")
    print(f"Total numeric rows across recordings: {total_rows:,}")
    print("Data loader completed.")
