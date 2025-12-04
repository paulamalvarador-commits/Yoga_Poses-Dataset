import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("Results/Angles_With_BadPoses_Controlled.csv")

# Identify angle columns (all numeric targets)
angle_cols = [c for c in df.columns if "angle" in c.lower()]

# Use only GOOD poses
good_df = df[df["Label"] == 1]

# -----------------------------
# 2. Build ONE multitarget model
# -----------------------------

# Use all angles as features
X = good_df[angle_cols]
y = good_df[angle_cols]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
)
model.fit(X_train, y_train)

# Predict all angles at once
preds = pd.DataFrame(model.predict(X_test), columns=angle_cols)

# -----------------------------
# 3. Evaluate per-angle
# -----------------------------
results = []

for angle in angle_cols:
    mae = mean_absolute_error(y_test[angle], preds[angle])
    rmse = np.sqrt(mean_squared_error(y_test[angle], preds[angle]))
    r2 = r2_score(y_test[angle], preds[angle])
    results.append([angle, mae, rmse, r2])

# -----------------------------
# 4. Save evaluation matrix
# -----------------------------
eval_df = pd.DataFrame(results, columns=["Angle", "MAE", "RMSE", "R2_Score"])
eval_df.to_csv("Results/Multioutput_Evaluation_Controlled.csv", index=False)

print("\nSaved evaluation to: Results/Multioutput_Evaluation_Controlled.csv")

# -----------------------------
# 5. Save MODEL for real-time use
# -----------------------------
joblib.dump(model, "Results/Multioutput_Regressor.pkl")
print("Saved model to: Results/Multioutput_Regressor.pkl")
