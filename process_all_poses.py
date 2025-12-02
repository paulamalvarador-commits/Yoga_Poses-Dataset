import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import pandas as pd
import os

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils


def detectPose(image, pose, display=False):
    '''
    This function performs pose detection on an image.
    '''
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    '''
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle


def angles_finder(landmarks):
    '''
    This function calculates all the required angles from landmarks.
    '''
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    angle_for_ardhaChandrasana1 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    angle_for_ardhaChandrasana2 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    hand_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])

    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

    neck_angle_uk = calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

    left_wrist_angle_bk = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_wrist_angle_bk = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    return [left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, 
            left_knee_angle, right_knee_angle, angle_for_ardhaChandrasana1, angle_for_ardhaChandrasana2, 
            hand_angle, left_hip_angle, right_hip_angle, neck_angle_uk, left_wrist_angle_bk, right_wrist_angle_bk]


# ========================= MAIN LOOP - PROCESS ALL POSES =========================

# Create dataframe with all columns
df = pd.DataFrame(columns=['Pose', 'Image_Name', 'left_elbow_angle', 'right_elbow_angle', 
                           'left_shoulder_angle', 'right_shoulder_angle', 'left_knee_angle', 
                           'right_knee_angle', 'angle_for_ardhaChandrasana1', 'angle_for_ardhaChandrasana2', 
                           'hand_angle', 'left_hip_angle', 'right_hip_angle', 'neck_angle_uk', 
                           'left_wrist_angle_bk', 'right_wrist_angle_bk'])

# Get all pose folders from TRAIN directory
train_path = 'TRAIN'
pose_folders = [folder for folder in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, folder))]

print(f"Found {len(pose_folders)} pose folders: {pose_folders}")
print("=" * 80)

# Loop through each pose folder
for pose_name in pose_folders:
    pose_path = os.path.join(train_path, pose_name, 'Images')
    
    if not os.path.exists(pose_path):
        print(f"Skipping {pose_name} - Images folder not found at {pose_path}")
        continue
    
    images = [img for img in os.listdir(pose_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nProcessing {pose_name}: {len(images)} images")
    
    for idx, filename in enumerate(images):
        try:
            image_path = os.path.join(pose_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to read {filename}")
                continue
            
            output_image, landmarks = detectPose(image, pose, display=False)
            
            if landmarks:
                angles = angles_finder(landmarks)
                new_row = pd.DataFrame.from_records([{
                    'Pose': pose_name,
                    'Image_Name': filename,
                    'left_elbow_angle': angles[0],
                    'right_elbow_angle': angles[1],
                    'left_shoulder_angle': angles[2],
                    'right_shoulder_angle': angles[3],
                    'left_knee_angle': angles[4],
                    'right_knee_angle': angles[5],
                    'angle_for_ardhaChandrasana1': angles[6],
                    'angle_for_ardhaChandrasana2': angles[7],
                    'hand_angle': angles[8],
                    'left_hip_angle': angles[9],
                    'right_hip_angle': angles[10],
                    'neck_angle_uk': angles[11],
                    'left_wrist_angle_bk': angles[12],
                    'right_wrist_angle_bk': angles[13]
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                print(f"  ✓ {filename}")
            else:
                print(f" No landmarks detected in {filename}")
                
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")

# Save the complete dataset
output_file = 'Results/All_Poses_Angles.csv'
os.makedirs('Results', exist_ok=True)
df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print(f"Processing complete! Saved {len(df)} records to {output_file}")
print(f"Poses processed: {df['Pose'].unique()}")
print(f"\nFirst few rows:")
print(df.head())

pose.close()