import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("Results/Angles_With_BadPoses_Controlled.csv")

angle_cols = [c for c in df.columns if "angle" in c.lower() and c != "Label"]

# Use only GOOD poses for ground truth
good_df = df[df["Label"] == 1]

results = []

# -----------------------------
# 2. Evaluate each angle model
# -----------------------------
for angle in angle_cols:
    print(f"Evaluating model for: {angle}")

    feature_cols = [c for c in angle_cols if c != angle]

    X = good_df[feature_cols]
    y = good_df[angle]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([angle, mae, rmse, r2])

# -----------------------------
# 3. Save evaluation matrix
# -----------------------------
eval_df = pd.DataFrame(results, columns=["Angle", "MAE", "RMSE", "R2_Score"])
eval_df.to_csv("Results/Regression_Evaluation_controlled.csv", index=False)

print("\nSaved to: Results/Regression_Evaluation_controlled.csv")
