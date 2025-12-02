import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("Results/Angles_With_BadPoses.csv")

# Features = all angle columns
X = df.drop(columns=["Pose", "Image_Name", "Label"])
y = df["Label"]

# ============================================
# TRAIN/TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ============================================
# SCALE FEATURES (Logistic Regression needs it)
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 1) LOGISTIC REGRESSION
# ============================================
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)

print("\n================ LOGISTIC REGRESSION ================")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ============================================
# 2) RANDOM FOREST (better model)
# ============================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n================ RANDOM FOREST ================")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ============================================
# SAVE MODELS
# ============================================
joblib.dump(log_reg, "logistic_pose_classifier.pkl")
joblib.dump(rf, "random_forest_pose_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModels saved:")
print(" - logistic_pose_classifier.pkl")
print(" - random_forest_pose_classifier.pkl")
print(" - scaler.pkl")
