import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Cargar el nuevo archivo CSV
df_new = pd.read_csv("Results/Angles_With_BadPoses_Controlled.csv")

# Inspección inicial
print("Primeras filas del nuevo DataFrame:")
print(df_new.head())
print("\nInformación del nuevo DataFrame:")
print(df_new.info())

# Columnas de características (ángulos) y columna objetivo
feature_cols = [
    'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle',
    'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle',
    'angle_for_ardhaChandrasana1', 'angle_for_ardhaChandrasana2',
    'hand_angle', 'left_hip_angle', 'right_hip_angle',
    'neck_angle_uk', 'left_wrist_angle_bk', 'right_wrist_angle_bk'
]
target_col = 'Label'

# Separar características y la etiqueta
X_new = df_new[feature_cols]
y_new = df_new[target_col]

# Estandarizar los datos
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Aplicar t-SNE
# Usamos los mismos parámetros para consistencia
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne_new = tsne.fit_transform(X_new_scaled)

# Crear un DataFrame para la visualización
tsne_df_new = pd.DataFrame(data = X_tsne_new, columns = ['tSNE Component 1', 'tSNE Component 2'])
tsne_df_new['Label'] = y_new

# Visualización
plt.figure(figsize=(10, 8))
# Separar los datos por Label para la gráfica
group_0 = tsne_df_new[tsne_df_new['Label'] == 0]
group_1 = tsne_df_new[tsne_df_new['Label'] == 1]

plt.scatter(group_0['tSNE Component 1'], group_0['tSNE Component 2'], label='Pose Malas (0)', alpha=0.7, color='red')
plt.scatter(group_1['tSNE Component 1'], group_1['tSNE Component 2'], label='Pose Buenas (1)', alpha=0.7, color='green')

plt.title('Visualización t-SNE de Poses Buenas y Malas (Datos Controlados)')
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.legend()
plt.grid(True)
plt.savefig('tsne_visualization_controlled.png')