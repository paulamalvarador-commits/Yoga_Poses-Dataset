import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
df = pd.read_csv("Results/Angles_With_BadPoses.csv")

# -------------------------------------------------------------
# 2. IDENTIFY ANGLE COLUMNS (all numeric, but skip Label)
# -------------------------------------------------------------
angle_cols = [col for col in df.columns
              if ("angle" in col.lower()) and (col != "Label")]

print("Using angle columns:", angle_cols)

# -------------------------------------------------------------
# 3. SPLIT GOOD AND BAD POSES
# -------------------------------------------------------------
good_df = df[df["Label"] == 1]    # correct poses
bad_df  = df[df["Label"] == 0]    # incorrect poses

# -------------------------------------------------------------
# 4. TRAIN A REGRESSION MODEL FOR EACH ANGLE
# -------------------------------------------------------------
models = {}
predicted_df = bad_df.copy()

for angle in angle_cols:
    print(f"Training regression for: {angle}")

    # Features: all OTHER angles
    feature_cols = [c for c in angle_cols if c != angle]

    X_train = good_df[feature_cols]
    y_train = good_df[angle]

    # Train Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
    )
    model.fit(X_train, y_train)

    models[angle] = model

    # Predict for BAD poses
    X_bad = bad_df[feature_cols]
    predicted_df[f"{angle}_corrected"] = model.predict(X_bad)

# -------------------------------------------------------------
# 5. SAVE RESULTS
# -------------------------------------------------------------
predicted_df.to_csv("Results/Corrected_Angles.csv", index=False)

print("\nDONE! File saved as Results/Corrected_Angles.csv")
