import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("Results/Angles_With_BadPoses_Controlled.csv")

# Features = all angle columns
X = df.drop(columns=["Pose", "Image_Name", "Label","Error_Type"]) 
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
    n_estimators=400,
    max_depth=None,   # deja que los árboles profundicen más
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1         # usa todos los cores, acelera mucho
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n================ RANDOM FOREST ================")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# SVM

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced"))
])

param_grid = {
    "svm__C": [0.1, 1, 10, 50, 100],
    "svm__gamma": ["scale", 0.01, 0.001, 0.0001],
    "svm__kernel": ["rbf", "poly"]
}

grid = GridSearchCV(
    svm_clf,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Obtener el mejor modelo
best_svm = grid.best_estimator_

# Predecir con el mejor SVM
y_pred_svm = best_svm.predict(X_test)

print("\n================ SVM ================")
print("Mejores hiperparámetros:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, zero_division=0))


# ============================================
# SAVE MODELS
# ============================================
joblib.dump(log_reg, "logistic_pose_classifier_controlled.pkl")
joblib.dump(rf, "random_forest_pose_classifier_controlled.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModels saved:")
print(" - logistic_pose_classifier_controlled.pkl")
print(" - random_forest_pose_classifier_controlled.pkl")
print(" - scaler.pkl")