# pose_classifier_multiclass.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================
# 1. Load dataset
# ============================================
df = pd.read_csv("Results/All_Poses_Angles.csv")

# Ensure Pose is string
df["Pose"] = df["Pose"].astype(str)

# Angle features (all angle columns)
angle_cols = [c for c in df.columns if "angle" in c.lower()]

X = df[angle_cols]
y = df["Pose"]

print("Angle columns:", angle_cols)
print("Pose classes:", y.unique())

# ============================================
# 2. Train / Test split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 3. Scale features
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 4. Train Random Forest Classifier
# ============================================
clf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train_scaled, y_train)

# ============================================
# 5. Evaluate
# ============================================
y_pred = clf.predict(X_test_scaled)

print("\n================ MULTICLASS POSE CLASSIFIER =================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================
# 6. Save classifier + scaler
# ============================================
os.makedirs("Results/Classifier", exist_ok=True)

joblib.dump(clf, "Results/Classifier/pose_name_classifier.pkl")
joblib.dump(scaler, "Results/Classifier/pose_scaler.pkl")

print("\nSaved:")
print(" - Results/Classifier/pose_name_classifier.pkl")
print(" - Results/Classifier/pose_scaler.pkl")