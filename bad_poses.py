import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CONFIGURACIÓN INICIAL
# ============================================
N_BAD = 1 
OUTPUT_FILE = "Results/Angles_With_BadPoses.csv"
PCA_PLOT_FILE = 'Dispersion_PCA_Clusters_Binario_Nuevo.png'

# ============================================
# 1. LOAD ORIGINAL DATA
# ============================================
try:
    df = pd.read_csv("Results/All_Poses_Angles.csv")
except FileNotFoundError:
    print("Error: No se encontró 'Results/All_Poses_Angles.csv'. No se puede generar el dataset final.")
    exit()

# Mark all as GOOD
df["Label"] = 1

# Angle columns (all angle fields)
# Ajustamos el slice para asegurar que solo contenga los ángulos
angle_cols = df.columns[2:-1] 

# ============================================
# BIOMECHANICAL LIMITS FOR EACH JOINT
# ============================================
def clip_angle(joint, value):
    limits = {
        "elbow": (10, 170), "shoulder": (20, 160), "knee": (20, 170),
        "hip": (30, 160), "neck": (70, 300), "wrist": (0, 330)
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
    Generates 1 synthetic bad pose from a good pose, with random noise strength.
    """
    noise_strength = np.random.choice([1, 2, 3])
    noisy = row.copy()

    # Selecciona una cantidad aleatoria de ángulos a modificar (entre 2 y 4)
    num_angles_to_modify = np.random.randint(2, 5) 
    angles_to_modify = np.random.choice(angle_cols, num_angles_to_modify, replace=False)

    for col in angles_to_modify:
        angle = row[col]

        # apply different noise amounts based on noise_strength
        if noise_strength == 1:
            angle += np.random.uniform(-15, 15)  # slightly incorrect
        elif noise_strength == 2:
            angle += np.random.uniform(-35, 35)  # medium incorrect
        else:
            angle += np.random.uniform(-70, 70)  # clearly wrong

        # clip to anatomical limits
        noisy[col] = clip_angle(col.lower(), angle)

    noisy["Label"] = 0  # 0 = bad pose
    return noisy


# ============================================
# 2. GENERATE AND COMBINE DATA
# ============================================
bad_samples = []

# Genera N_BAD poses por cada pose buena
for _, row in df.iterrows():
    for _ in range(N_BAD):
        bad_samples.append(make_bad_pose(row))

df_bad = pd.DataFrame(bad_samples)

# COMBINE GOOD + BAD
df_final = pd.concat([df, df_bad], ignore_index=True)

print("Final dataset size:", df_final.shape)
print(df_final["Label"].value_counts())

# ============================================
# 3. VISUALIZACIÓN DE DISPERSIÓN CON PCA
# ============================================

# 3.1. Definir Features (X_pca) y Target (y_pca)
X_pca = df_final[angle_cols]
y_pca = df_final["Label"]

# 3.2. Estandarizar los datos (CRÍTICO para PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# 3.3. Aplicar PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# 3.4. Crear un DataFrame con los resultados de PCA y el target
pca_df = pd.DataFrame(
    data=principal_components, 
    columns=['PC1', 'PC2']
)
pca_df['Target'] = y_pca.map({1: 'Good (1)', 0: 'Bad (0)'}) 

# 3.5. Generar el Scatter Plot de la dispersión (Clusters)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue='Target', 
    data=pca_df, 
    
    # <--- ESTA ES LA LÍNEA CORREGIDA --->
    palette={'Good (1)': '#4daf4a', 'Bad (0)': '#e41a1c'}, 
    
    s=50, 
    alpha=0.7
)

# 3.6. Etiquetas y Título
plt.title('Dispersión de Poses Good vs. Bad (PCA - Nueva Generación)', fontsize=16)
plt.xlabel(f'Primer Componente Principal (PC1 - {pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'Segundo Componente Principal (PC2 - {pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.legend(title='Clase', loc='upper right')
plt.grid(True)

# Guardar el gráfico
plt.savefig(PCA_PLOT_FILE, bbox_inches='tight')

# ============================================
# 4. SAVE FINAL DATASET
# ============================================
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Gráfico de dispersión PCA guardado como {PCA_PLOT_FILE}")
print(f"✅ Dataset final guardado como {OUTPUT_FILE}")