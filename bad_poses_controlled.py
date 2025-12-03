import pandas as pd
import numpy as np
import os

# ============================================
# CONFIGURACIÓN: Define las ratios de generación
# ============================================
GENERATION_RATIOS = {
    # {nombre de etiqueta : (cantidad a generar, min_error, max_error)}
    'bad_subtle': (0.33, 5, 15),     # 3 poses malas sutiles (±5° a ±15°)
    'bad_medium': (0.33, 15, 30),    # 3 poses malas medias (±15° a ±30°)
    'bad_large': (0.33, 30, 70)      # 2 poses malas grandes (±30° a ±70°)
}

# Read the good poses data
good_poses_df = pd.read_csv("Results/All_Poses_Angles.csv")
# La etiqueta final 'Label' (1=good) se añade más abajo
good_poses_df["Label"] = 1

# List of angle columns (all angle fields)
angle_cols = good_poses_df.columns[2:-1]
NUM_GOOD_POSES = len(good_poses_df)


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
# BAD POSE GENERATOR (MODIFICADO)
# ============================================
def make_bad_pose(row, error_label, min_err, max_err):
    """
    Genera 1 pose mala sintética aplicando un error dentro de un rango definido.
    """
    noisy = row.copy()
    
    # Selecciona un número aleatorio de ángulos a modificar (entre 2 y 4)
    num_angles_to_modify = np.random.randint(2, 5)
    angles_to_modify = np.random.choice(angle_cols, num_angles_to_modify, replace=False)

    for col in angles_to_modify:
        angle = row[col]

        # Calcula el error dentro del rango (min_err, max_err)
        error_amount = np.random.uniform(min_err, max_err) * np.random.choice([-1, 1])
        angle += error_amount

        # clip to anatomical limits
        noisy[col] = clip_angle(col.lower(), angle)

    # Añade la etiqueta de clase de error (string)
    noisy["Error_Type"] = error_label
    # La etiqueta numérica 0/1 sigue siendo 0 para todas las malas
    noisy["Label"] = 0 
    
    return noisy


# ============================================
# GENERATE BAD POSES (BUCLE ANIDADO)
# ============================================
bad_samples = []

for error_type, (ratio, min_err, max_err) in GENERATION_RATIOS.items():
    
    # 1. Calcular el número exacto de poses a generar para esta clase
    num_to_generate = int(NUM_GOOD_POSES * ratio)
    
    # 2. Seleccionar aleatoriamente las poses buenas para usar como base
    # Usamos replace=True para que podamos muestrear la misma pose buena múltiples veces
    if num_to_generate > 0:
        base_poses_for_error = good_poses_df.sample(
            n=num_to_generate, 
            replace=True, 
            random_state=42
        ).iterrows()
    else:
        continue

    print(f"   Generando {num_to_generate} poses de tipo '{error_type}' ({ratio * 100:.1f}%)")
    
    # 3. Generar las poses malas
    for _, row in base_poses_for_error:
        bad_pose_data = make_bad_pose(row, error_type, min_err, max_err)
        bad_samples.append(bad_pose_data)

df_bad = pd.DataFrame(bad_samples)

print(f"\n✅ Generación completada. Total de poses malas: {len(df_bad)}")
# ============================================
# COMBINE GOOD + BAD
# ============================================

# Añade una etiqueta de error de texto a las poses buenas antes de concatenar
good_poses_df["Error_Type"] = "good" 

# Combina good y bad
df_final = pd.concat([good_poses_df, df_bad], ignore_index=True)

# Mezcla (shuffle) el dataset final
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- Distribución Final ---")
print(f"Tamaño final del dataset: {df_final.shape}")
print(df_final["Error_Type"].value_counts())


# ============================================
# SAVE
# ============================================
df_final.to_csv("Results/Angles_With_BadPoses_Controlled.csv", index=False)

print("\nDataset guardado como: Results/Angles_With_BadPoses_Controlled.csv")