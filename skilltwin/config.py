"""Shared constants for the Skill Twin pipeline."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR.parent / "out"

FS = 200                  # Hz, rate of the Segmented/ files (IMU is 100 Hz, sample-and-hold)
EXPERT_MIN_SCORE = 8      # a set is "expert" if overall quality >= this and it has zero errors

# Muscle key -> (column prefix in the Mendeley CSVs, body segment its IMU sits on).
# Biceps and triceps IMUs both sit on the upper arm (quaternion corr up to 0.99),
# so the sensors cover three segments: upper arm, thigh, shank.
SENSORS = {
    "biceps":  ("Biceps", "upper_arm"),
    "triceps": ("Triceps", "upper_arm"),
    "rectus":  ("Rectus", "thigh"),
    "gastroc": ("Gastrocnemious", "shank"),
}
QUAT = ("w", "x", "y", "z")   # Xsens order: Quat_q0 is the scalar part

# Canonical exercise name (as in ExerciseQuality/) -> (family, mode, side).
# Side of "Side plank hold" is assumed: it pairs with the "(R side)" file.
EXERCISES = {
    "Standard squat":            ("squat", "dynamic", None),
    "Wide squat":                ("squat", "dynamic", None),
    "Jump squats":               ("squat", "dynamic", None),
    "Squat hold":                ("squat", "hold", None),
    "Wide squat hold":           ("squat", "hold", None),
    "Lunge (left leg forward)":  ("lunge", "dynamic", "L"),
    "Lunge (right leg forward)": ("lunge", "dynamic", "R"),
    "Lunge hold (left leg)":     ("lunge", "hold", "L"),
    "Lunge hold (right leg)":    ("lunge", "hold", "R"),
    "Push-ups":                  ("pushup", "dynamic", None),
    "Triceps push-ups":          ("pushup", "dynamic", None),
    "Extended arm plank hold":   ("plank", "hold", None),
    "Forearm plank hold":        ("plank", "hold", None),
    "Plank walkout":             ("plank", "dynamic", None),
    "Mountain climbers":         ("plank", "dynamic", None),
    "Side plank hold":           ("side_plank", "hold", "L"),
    "Side plank hold (R side)":  ("side_plank", "hold", "R"),
    "Side plank rot.(L side)":   ("side_plank", "dynamic", "L"),
    "Side plank rot.(R side)":   ("side_plank", "dynamic", "R"),
    "Bridging (lying on back)":  ("bridge", "dynamic", None),
    "Bridging hold":             ("bridge", "hold", None),
    "Superman":                  ("superman", "dynamic", None),
    "Superman hold":             ("superman", "hold", None),
    "Burpees":                   ("burpee", "dynamic", None),
}

# Segmented file names that differ from the label name beyond spacing/case.
FILE_ALIASES = {
    "bridginghold(lyingonback)": "Bridging hold",
}

# Single-muscle recordings in NonRepetitive/ (probably reference contractions; unconfirmed).
REFERENCE_FILES = {
    "biceps": "ref:biceps",
    "triceps": "ref:triceps",
    "rectus": "ref:rectus",
    "gastrocnemius": "ref:gastroc",
}

# Can the sensors (upper arm, thigh, shank IMU + 4 EMG) judge this rule?
#   yes     - directly measurable
#   partial - only a proxy is measurable; try it, report it separately
#   no      - needs head / trunk / feet / hands, which no sensor covers
OBSERVABILITY = {
    "squat_depth":              ("yes", "knee angle from thigh vs shank orientation"),
    "pushup_depth":             ("yes", "upper-arm pitch range"),
    "full_range_of_motion":     ("yes", "upper-arm / thigh range of motion"),
    "range_of_motion":          ("yes", "upper-arm rotation range"),
    "full_hip_extension":       ("yes", "thigh pitch at top of bridge"),
    "leg_position":             ("yes", "knee angle (leg straight or bent)"),
    "leg_placement":            ("yes", "knee flexion during walkout"),
    "hip_joint_angle":          ("yes", "thigh pitch when knee is drawn in"),
    "limb_movement_uniformity": ("yes", "timing of upper-arm vs thigh lift"),
    "movement_fluidity":        ("yes", "jerk / smoothness of segment rotation"),
    "tempo":                    ("yes", "repetition duration variability"),
    "hold_and_tempo":           ("yes", "repetition duration variability"),
    "technique":                ("yes", "drift of features from first to last repetition"),
    "stability":                ("yes", "tremor and sway of segment orientation"),
    "knee_joint_axis":          ("partial", "frontal-plane thigh/shank angle; sensor-to-bone alignment unknown"),
    "ankle_joint_axis":         ("partial", "shank roll only"),
    "hip_angle":                ("partial", "thigh vs upper-arm pitch, no trunk sensor"),
    "body_line":                ("partial", "thigh vs upper-arm alignment, no trunk sensor"),
    "plank_position":           ("partial", "thigh vs upper-arm alignment, no trunk sensor"),
    "landing":                  ("partial", "gastrocnemius EMG burst, no accelerometer"),
    "jump_height":              ("partial", "thigh/shank angular speed at push-off"),
    "body_down_to_floor":       ("partial", "upper-arm angular speed during descent"),
    "transition_to_squat_I":    ("partial", "segment angular speed in transition"),
    "transition_to_squat_II":   ("partial", "segment angular speed in transition"),
    "transition_to_plank_I":    ("partial", "segment angular speed in transition"),
    "transition_to_plank_II":   ("partial", "segment angular speed in transition"),
    "core_stability":           ("partial", "limb sway as a proxy, no trunk sensor"),
    "core_stability_I":         ("partial", "limb sway as a proxy, no trunk sensor"),
    "core_stability_II":        ("partial", "limb sway as a proxy, no trunk sensor"),
    "elbow_extension":          ("no", "no forearm sensor"),
    "elbow_position":           ("no", "no forearm / trunk sensor"),
    "support_angle":            ("no", "needs shoulder-over-hand position"),
    "spine_alignment":          ("no", "no trunk sensor"),
    "head_axis":                ("no", "no head sensor"),
    "foot_position":            ("no", "no foot sensor"),
    "toe_direction":            ("no", "no foot sensor"),
    "feet":                     ("no", "no foot sensor"),
    "lower_limb_position":      ("no", "needs stance width"),
    "center_of_gravity":        ("no", "needs force plate"),
    "weight_distribution":      ("no", "needs force plate"),
    "hand_position":            ("no", "no hand sensor"),
    "finger_direction":         ("no", "no hand sensor"),
    "arm_position":             ("no", "needs arm vs trunk position"),
    "movement_symmetry":        ("no", "sensors on one side only"),
    "plank_alignment":          ("no", "no trunk / head sensor"),
}
