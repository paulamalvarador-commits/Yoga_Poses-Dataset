import pandas as pd
import numpy as np

# ============================================
# CONFIG: HOW MANY BAD POSES PER GOOD POSE?
# ============================================
N_BAD = 3   # <-- CAMBIA ESTE VALOR

# ============================================
# LOAD ORIGINAL DATA
# ============================================
df = pd.read_csv("Results/All_Poses_Angles.csv")

# Mark all as GOOD
df["Label"] = 1

# Angle columns (all angle fields)
angle_cols = df.columns[2:-1]

# ============================================
# BIOMECHANICAL LIMITS FOR EACH JOINT
# ============================================
def clip_angle(joint, value):
    limits = {
        "elbow": (10, 170),
        "shoulder": (20, 160),
        "knee": (20, 170),
        "hip": (30, 160),
        "neck": (70, 300),
        "wrist": (0, 330)
    }
    for key, (low, high) in limits.items():
        if key in joint:
            return np.clip(value, low, high)
    return value


# ============================================
# BAD POSE GENERATOR
# ============================================
def make_bad_pose(row):
    """
    Generates ONE synthetic bad pose from a good pose.
    Noise_strength chosen randomly.
    """
    noise_strength = np.random.choice([1, 2, 3])
    noisy = row.copy()

    for col in angle_cols:
        angle = row[col]

        # apply different noise amounts
        if noise_strength == 1:
            angle += np.random.uniform(-15, 15)
        elif noise_strength == 2:
            angle += np.random.uniform(-35, 35)
        else:
            angle += np.random.uniform(-70, 70)

        # clip to anatomical limits
        noisy[col] = clip_angle(col.lower(), angle)

    noisy["Label"] = 0  # 0 = bad pose
    return noisy


# ============================================
# GENERATE N_BAD POSES FOR EACH GOOD POSE
# ============================================
bad_samples = []

for _, row in df.iterrows():
    for _ in range(N_BAD):       # <-- AÑADIDO
        bad_samples.append(make_bad_pose(row))

df_bad = pd.DataFrame(bad_samples)

# ============================================
# COMBINE GOOD + BAD
# ============================================
df_final = pd.concat([df, df_bad], ignore_index=True)

print("Final dataset size:", df_final.shape)
print(df_final["Label"].value_counts())

# ============================================
# SAVE
# ============================================
df_final.to_csv("Results/Angles_With_BadPoses.csv", index=False)

print("\nDataset saved as Angles_With_BadPoses.csv")