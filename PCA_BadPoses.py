import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv("Results/Angles_With_BadPoses.csv")

# Inspección inicial
print("Primeras filas del DataFrame:")
print(df.head())
print("\nInformación del DataFrame:")
print(df.info())

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
X = df[feature_cols]
y = df[target_col]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar t-SNE
# Ajustar el número de iteraciones y el parámetro perplexity si es necesario.
# Con un número pequeño de filas (como 251), los valores por defecto suelen funcionar bien.
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# Crear un DataFrame para la visualización
tsne_df = pd.DataFrame(data = X_tsne, columns = ['tSNE Component 1', 'tSNE Component 2'])
tsne_df['Label'] = y

# Visualización
plt.figure(figsize=(10, 8))
# Separar los datos por Label para la gráfica
group_0 = tsne_df[tsne_df['Label'] == 0]
group_1 = tsne_df[tsne_df['Label'] == 1]

plt.scatter(group_0['tSNE Component 1'], group_0['tSNE Component 2'], label='Pose Malas (0)', alpha=0.7, color='red')
plt.scatter(group_1['tSNE Component 1'], group_1['tSNE Component 2'], label='Pose Buenas (1)', alpha=0.7, color='green')

plt.title('Visualización t-SNE de Poses Buenas y Malas')
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.legend()
plt.grid(True)
plt.savefig('tsne_visualization.png')