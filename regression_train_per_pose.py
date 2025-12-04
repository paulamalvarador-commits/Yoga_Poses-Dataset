import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("Results/Angles_With_BadPoses_Controlled.csv")

# Identify angle columns
angle_cols = [c for c in df.columns if "angle" in c.lower()]

# Ensure output folder exists
os.makedirs("Results/Regressors", exist_ok=True)
os.makedirs("Results/Regressor_Evaluations", exist_ok=True)

# -----------------------------
# 2. Train a model PER POSE
# -----------------------------
poses = df["Pose"].unique()

print("\nTraining regressors for poses:")
print(poses)

for pose_name in poses:

    print(f"\n=======================================")
    print(f"   Training regressor for: {pose_name}")
    print(f"=======================================")

    # Filter this pose AND only good examples
    pose_df = df[(df["Pose"] == pose_name) & (df["Label"] == 1)]

    if len(pose_df) < 10:
        print(f"⚠️ Skipping {pose_name}: Not enough GOOD samples.")
        continue

    X = pose_df[angle_cols]
    y = pose_df[angle_cols]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    preds_df = pd.DataFrame(preds, columns=angle_cols)

    # ------------------------------------
    # Save MODEL
    # ------------------------------------
    model_path = f"Results/Regressors/Regressor_{pose_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved: {model_path}")

    # ------------------------------------
    # Save evaluation metrics
    # ------------------------------------
    results = []

    for angle in angle_cols:
        mae = mean_absolute_error(y_test[angle], preds_df[angle])
        rmse = np.sqrt(mean_squared_error(y_test[angle], preds_df[angle]))
        r2 = r2_score(y_test[angle], preds_df[angle])
        results.append([angle, mae, rmse, r2])

    eval_df = pd.DataFrame(results, columns=["Angle", "MAE", "RMSE", "R2"])

    eval_path = f"Results/Regressor_Evaluations/Eval_{pose_name}.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f" Saved evaluation: {eval_path}")

print("\nAll pose regressors trained.")
