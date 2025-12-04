# universal_pose_corrector.py

import cv2
import math
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import os

# ============================
# 0. CONFIG
# ============================
TOP_N = 3        # top N biggest corrections
MIN_DIFF = 12.0  # show only corrections > 12°

# ============================
# 1. LOAD CLASSIFIER & SCALER
# ============================
pose_clf = joblib.load("Results/Classifier/pose_name_classifier.pkl")
pose_scaler = joblib.load("Results/Classifier/pose_scaler.pkl")

# ============================
# 2. MEDIAPIPE INIT
# ============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Angle columns (must match training)
angle_cols = [
    "left_elbow_angle","right_elbow_angle",
    "left_shoulder_angle","right_shoulder_angle",
    "left_knee_angle","right_knee_angle",
    "angle_for_ardhaChandrasana1","angle_for_ardhaChandrasana2",
    "hand_angle","left_hip_angle","right_hip_angle",
    "neck_angle_uk","left_wrist_angle_bk","right_wrist_angle_bk"
]


# ============================
# 3. ANGLE FUNCTIONS
# ============================
def calculate_angle(a, b, c):
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
    angles = {}

    angles["left_elbow_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
    )

    angles["right_elbow_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
    )

    angles["left_shoulder_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    )

    angles["right_shoulder_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    )

    angles["left_knee_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    )

    angles["right_knee_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    )

    angles["angle_for_ardhaChandrasana1"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    )

    angles["angle_for_ardhaChandrasana2"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    )

    angles["hand_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    )

    angles["left_hip_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
    )

    angles["right_hip_angle"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
    )

    angles["neck_angle_uk"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.NOSE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    )

    angles["left_wrist_angle_bk"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    )

    angles["right_wrist_angle_bk"] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    )

    return angles


# ============================
# 4. IMAGE → LANDMARKS + ANGLES
# ============================
def get_landmarks_and_angles(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not load image: {image_path}")

    h, w, _ = img.shape
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        raise Exception("No pose detected by Mediapipe.")

    landmarks = [
        (lm.x * w, lm.y * h, lm.z * w)
        for lm in results.pose_landmarks.landmark
    ]

    angles = extract_angles(landmarks)
    return img, landmarks, angles


# ============================
# 5. DRAW SKELETON + HIGHLIGHTS
# ============================
def draw_skeleton(img, landmarks, color=(255, 255, 255), thickness=2):
    output = img.copy()
    for conn in mp_pose.POSE_CONNECTIONS:
        i1 = conn[0]
        i2 = conn[1]
        x1, y1, _ = landmarks[i1]
        x2, y2, _ = landmarks[i2]
        cv2.line(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness
        )
    return output


def highlight_joints(img, landmarks, corrections):
    """
    corrections: list of (angle_name, diff, joint_index)
    """
    output = img.copy()

    for angle_name, diff, joint_index in corrections:
        x, y, _ = landmarks[joint_index]
        x, y = int(x), int(y)

        cv2.circle(output, (x, y), 10, (0, 0, 255), -1)  # red circle

    return output


# ============================
# 6. MAIN ANALYSIS PIPELINE
# ============================
def analyze_pose(image_path):
    print(f"\n🔍 Analyzing: {image_path}")

    img, landmarks, angles = get_landmarks_and_angles(image_path)
    df_angles = pd.DataFrame([angles])[angle_cols]

    # -------------------------
    # Pose classification
    # -------------------------
    X_scaled = pose_scaler.transform(df_angles)
    pose_name = pose_clf.predict(X_scaled)[0]

    print(f"\n🧘 Detected pose: {pose_name}")

    # -------------------------
    # Load regressor for this pose
    # -------------------------
    regressor_path = f"Results/Regressors/Regressor_{pose_name}.pkl"
    if not os.path.exists(regressor_path):
        raise Exception(f"Regressor not found for pose {pose_name}: {regressor_path}")

    regressor = joblib.load(regressor_path)

    # -------------------------
    # Predict ideal angles
    # -------------------------
    ideal = regressor.predict(df_angles)[0]

    # -------------------------
    # Compute differences
    # -------------------------
    diffs = []
    for i, col in enumerate(angle_cols):
        user_val = df_angles[col].iloc[0]
        ideal_val = ideal[i]
        diff = ideal_val - user_val

        if abs(diff) > MIN_DIFF:
            diffs.append((col, diff))

    if not diffs:
        print("\n✅ Pose is already very close to ideal (no corrections > "
              f"{MIN_DIFF}°).")
        # Still draw user skeleton
        os.makedirs("Results", exist_ok=True)
        skel = draw_skeleton(img, landmarks, (255, 255, 255), 2)
        out_path = "Results/pose_feedback.png"
        cv2.imwrite(out_path, skel)
        print(f"🖼 Feedback image saved to: {out_path}")
        cv2.imshow("Pose Feedback", skel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Sort by absolute difference, descending, take top N
    diffs = sorted(diffs, key=lambda x: abs(x[1]), reverse=True)
    top_diffs = diffs[:TOP_N]

    # Map angle → joint index for highlighting
    joint_map = {
        "left_elbow_angle": mp_pose.PoseLandmark.LEFT_ELBOW.value,
        "right_elbow_angle": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        "left_shoulder_angle": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        "right_shoulder_angle": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        "left_knee_angle": mp_pose.PoseLandmark.LEFT_KNEE.value,
        "right_knee_angle": mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "left_hip_angle": mp_pose.PoseLandmark.LEFT_HIP.value,
        "right_hip_angle": mp_pose.PoseLandmark.RIGHT_HIP.value,
        "neck_angle_uk": mp_pose.PoseLandmark.NOSE.value,
        "left_wrist_angle_bk": mp_pose.PoseLandmark.LEFT_WRIST.value,
        "right_wrist_angle_bk": mp_pose.PoseLandmark.RIGHT_WRIST.value,
        "hand_angle": mp_pose.PoseLandmark.RIGHT_WRIST.value,
        "angle_for_ardhaChandrasana1": mp_pose.PoseLandmark.RIGHT_HIP.value,
        "angle_for_ardhaChandrasana2": mp_pose.PoseLandmark.LEFT_HIP.value,
    }

    print("\n❌ Top corrections (>|{:.1f}°|, max {}):".format(MIN_DIFF, TOP_N))
    corrections_for_drawing = []
    text_lines = []

    for col, diff in top_diffs:
        direction = "increase" if diff > 0 else "decrease"
        joint_index = joint_map[col]
        corrections_for_drawing.append((col, diff, joint_index))
        line = f"{col}: {direction} by {abs(diff):.1f}°"
        text_lines.append(line)
        print("•", line)

    # -------------------------
    # Draw skeleton and highlights
    # -------------------------
    skel = draw_skeleton(img, landmarks, (255, 255, 255), 2)
    skel = highlight_joints(skel, landmarks, corrections_for_drawing)

    # Put text in top-left
    y0 = 30
    for i, line in enumerate(text_lines):
        y = y0 + i * 25
        cv2.putText(
            skel, line, (30, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 255), 2
        )

    # Save and show
    os.makedirs("Results", exist_ok=True)
    out_path = "Results/pose_feedback.png"
    cv2.imwrite(out_path, skel)
    print(f"\n🖼 Visual feedback saved to: {out_path}")

    cv2.imshow("Pose Feedback", skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================
# 7. RUN
# ============================
if __name__ == "__main__":
    # Change 'test.png' to your uploaded photo filename
    analyze_pose("test.png")
    analyze_pose("testAle.jpeg")