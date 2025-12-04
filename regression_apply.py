import cv2
import math
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import os

# =====================================
# CONFIG
# =====================================
TOP_N = 3          # show top N corrections
MIN_DIFF = 12.0    # only show corrections with |diff| >= 12°

# =====================================
# LOAD CLASSIFIER + SCALER
# =====================================
POSE_CLS_PATH = "Results/Classifier/pose_name_classifier.pkl"
POSE_SCALER_PATH = "Results/Classifier/pose_scaler.pkl"

pose_clf = joblib.load(POSE_CLS_PATH)
pose_scaler = joblib.load(POSE_SCALER_PATH)

# =====================================
# MEDIAPIPE INIT
# =====================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# =====================================
# ANGLE COLUMNS (must match training)
# =====================================
angle_cols = [
    "left_elbow_angle", "right_elbow_angle",
    "left_shoulder_angle", "right_shoulder_angle",
    "left_knee_angle", "right_knee_angle",
    "angle_for_ardhaChandrasana1", "angle_for_ardhaChandrasana2",
    "hand_angle", "left_hip_angle", "right_hip_angle",
    "neck_angle_uk", "left_wrist_angle_bk", "right_wrist_angle_bk"
]

# =====================================
# POSE → RELEVANT ANGLES MAP
#   (angles we are allowed to correct per pose)
# =====================================
POSE_ANGLE_MAP = {
    "Downward_dog": [
        "left_elbow_angle", "right_elbow_angle",
        "left_shoulder_angle", "right_shoulder_angle",
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "neck_angle_uk",
        "left_wrist_angle_bk", "right_wrist_angle_bk",
        "hand_angle"
    ],

    "ArdhaChandrasana": [
        "left_hip_angle", "right_hip_angle",
        "angle_for_ardhaChandrasana1", "angle_for_ardhaChandrasana2",
        "left_knee_angle", "right_knee_angle"
    ],

    "Triangle": [
        "left_hip_angle", "right_hip_angle",
        "left_shoulder_angle", "right_shoulder_angle",
        "neck_angle_uk"
    ],

    "Veerabhadrasana": [
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "left_shoulder_angle", "right_shoulder_angle"
    ],

    "Natarajasana": [
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "left_shoulder_angle", "right_shoulder_angle"
    ],

    "UtkataKonasana": [
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "left_shoulder_angle", "right_shoulder_angle"
    ],

    "Vrukshasana": [
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "neck_angle_uk"
    ],

    "BaddhaKonasana": [
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle"
    ]
}

# =====================================
# JOINT MAP: angle → landmark index
#   Store ints so we can index landmarks directly
# =====================================
joint_map = {
    "left_elbow_angle": int(mp_pose.PoseLandmark.LEFT_ELBOW),
    "right_elbow_angle": int(mp_pose.PoseLandmark.RIGHT_ELBOW),
    "left_shoulder_angle": int(mp_pose.PoseLandmark.LEFT_SHOULDER),
    "right_shoulder_angle": int(mp_pose.PoseLandmark.RIGHT_SHOULDER),
    "left_knee_angle": int(mp_pose.PoseLandmark.LEFT_KNEE),
    "right_knee_angle": int(mp_pose.PoseLandmark.RIGHT_KNEE),
    "left_hip_angle": int(mp_pose.PoseLandmark.LEFT_HIP),
    "right_hip_angle": int(mp_pose.PoseLandmark.RIGHT_HIP),
    "neck_angle_uk": int(mp_pose.PoseLandmark.NOSE),
    "left_wrist_angle_bk": int(mp_pose.PoseLandmark.LEFT_WRIST),
    "right_wrist_angle_bk": int(mp_pose.PoseLandmark.RIGHT_WRIST),
    "hand_angle": int(mp_pose.PoseLandmark.RIGHT_WRIST),
    "angle_for_ardhaChandrasana1": int(mp_pose.PoseLandmark.RIGHT_HIP),
    "angle_for_ardhaChandrasana2": int(mp_pose.PoseLandmark.LEFT_HIP),
}

# =====================================
# BASIC ANGLE + EXTRACTION FUNCTIONS
# =====================================
def calculate_angle(a, b, c):
    """Return angle at point b given 3 landmarks (x,y,z)."""
    x1, y1, _ = a
    x2, y2, _ = b
    x3, y3, _ = c

    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) -
        math.atan2(y1 - y2, x1 - x2)
    )
    if angle < 0:
        angle += 360
    return angle


def extract_angles(landmarks):
    """Compute all angles used by the model from Mediapipe landmarks."""
    angles = {}

    angles["left_elbow_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.LEFT_SHOULDER)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_ELBOW)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_WRIST)],
    )

    angles["right_elbow_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.RIGHT_SHOULDER)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_ELBOW)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_WRIST)],
    )

    angles["left_shoulder_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.LEFT_ELBOW)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_SHOULDER)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_HIP)],
    )

    angles["right_shoulder_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.RIGHT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_SHOULDER)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_ELBOW)],
    )

    angles["left_knee_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.LEFT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_KNEE)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_ANKLE)],
    )

    angles["right_knee_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.RIGHT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_KNEE)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_ANKLE)],
    )

    angles["angle_for_ardhaChandrasana1"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.RIGHT_ANKLE)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_ANKLE)],
    )

    angles["angle_for_ardhaChandrasana2"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.LEFT_ANKLE)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_ANKLE)],
    )

    angles["hand_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.LEFT_ELBOW)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_SHOULDER)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_ELBOW)],
    )

    angles["left_hip_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.LEFT_SHOULDER)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_KNEE)],
    )

    angles["right_hip_angle"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.RIGHT_SHOULDER)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_KNEE)],
    )

    angles["neck_angle_uk"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.NOSE)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_SHOULDER)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_SHOULDER)],
    )

    angles["left_wrist_angle_bk"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.LEFT_WRIST)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.LEFT_ANKLE)],
    )

    angles["right_wrist_angle_bk"] = calculate_angle(
        landmarks[int(mp_pose.PoseLandmark.RIGHT_WRIST)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_HIP)],
        landmarks[int(mp_pose.PoseLandmark.RIGHT_ANKLE)],
    )

    return angles


def get_image_landmarks_angles(image_path):
    """Load image, run Mediapipe, return (img, landmarks, angles dict)."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot load image: {image_path}")

    h, w, _ = img.shape
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        raise RuntimeError("No pose detected in image.")

    landmarks = [
        (lm.x * w, lm.y * h, lm.z * w)
        for lm in results.pose_landmarks.landmark
    ]
    angles = extract_angles(landmarks)
    return img, landmarks, angles


# =====================================
# DRAWING HELPERS
# =====================================
def draw_skeleton(img, landmarks, color=(255, 255, 255), thickness=2):
    """Draw a simple skeleton from Mediapipe connections."""
    out = img.copy()
    for i1, i2 in mp_pose.POSE_CONNECTIONS:
        x1, y1, _ = landmarks[int(i1)]
        x2, y2, _ = landmarks[int(i2)]
        cv2.line(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness,
        )
    return out


def highlight_joints(img, landmarks, corrections):
    """
    corrections: list of (angle_name, diff, joint_index)
    """
    out = img.copy()
    for angle_name, diff, j_idx in corrections:
        x, y, _ = landmarks[int(j_idx)]
        cv2.circle(out, (int(x), int(y)), 10, (0, 0, 255), -1)
    return out


# =====================================
# MAIN ANALYSIS PIPELINE
# =====================================
def analyze_pose(image_path: str):
    print(f"\n🔍 Analyzing: {image_path}")

    img, landmarks, angles = get_image_landmarks_angles(image_path)
    df = pd.DataFrame([angles])[angle_cols]

    # ---------- Pose classification ----------
    X_scaled = pose_scaler.transform(df)
    pose_name = pose_clf.predict(X_scaled)[0]
    print(f"\n🧘 Detected pose: {pose_name}")

    # ---------- Load pose-specific regressor ----------
    regressor_path = f"Results/Regressors/Regressor_{pose_name}.pkl"
    if not os.path.exists(regressor_path):
        raise RuntimeError(f"Regressor not found for pose {pose_name}: {regressor_path}")

    regressor = joblib.load(regressor_path)

    # ---------- Predict ideal angles ----------
    ideal = regressor.predict(df)[0]

    # ---------- Compute corrections (pose-specific) ----------
    relevant_angles = POSE_ANGLE_MAP.get(pose_name, angle_cols)

    diffs = []
    for i, col in enumerate(angle_cols):
        if col not in relevant_angles:
            continue

        user_val = df[col].iloc[0]
        ideal_val = ideal[i]
        diff = ideal_val - user_val

        if abs(diff) >= MIN_DIFF:
            diffs.append((col, diff))

    if not diffs:
        print(f"\n✅ No corrections larger than {MIN_DIFF}° needed for this pose.")
        os.makedirs("Results", exist_ok=True)
        skel = draw_skeleton(img, landmarks)
        out_path = "Results/pose_feedback.png"
        cv2.imwrite(out_path, skel)
        print(f"🖼 Feedback image saved to: {out_path}")
        cv2.imshow("Pose Feedback", skel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Sort by magnitude and keep top N
    diffs = sorted(diffs, key=lambda x: abs(x[1]), reverse=True)
    top_diffs = diffs[:TOP_N]

    print(f"\n❌ Top corrections (>|{MIN_DIFF}°|, max {TOP_N}):")
    corrections_for_drawing = []
    text_lines = []

    for col, diff in top_diffs:
        direction = "increase" if diff > 0 else "decrease"
        joint_idx = joint_map[col]
        corrections_for_drawing.append((col, diff, joint_idx))
        line = f"{col}: {direction} by {abs(diff):.1f}°"
        text_lines.append(line)
        print("•", line)

    # ---------- Draw output ----------
    skel = draw_skeleton(img, landmarks)
    skel = highlight_joints(skel, landmarks, corrections_for_drawing)

    # Put text lines
    y0 = 30
    for i, line in enumerate(text_lines):
        y = y0 + i * 25
        cv2.putText(
            skel,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    os.makedirs("Results", exist_ok=True)

    # Save with same name as input image
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)

    out_path = f"Results/{name}_feedback.png"
    cv2.imwrite(out_path, skel)

    print(f"🖼 Saved to: {out_path}")

  

# =====================================
# ENTRY POINT
# =====================================
if __name__ == "__main__":
    folder = "ima"
    valid_ext = (".jpg", ".jpeg", ".png")

    for fname in os.listdir(folder):
        if fname.lower().endswith(valid_ext):
            path = os.path.join(folder, fname)
            try:
                analyze_pose(path)
            except Exception as e:
                print(f"Error with {path}: {e}")
