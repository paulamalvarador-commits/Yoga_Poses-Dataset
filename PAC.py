import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================
df = pd.read_csv("Results/Angles_With_BadPoses_Controlled.csv")

# 1.1 Definir Features (X)
angle_columns = [
    'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle',
    'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle',
    'angle_for_ardhaChandrasana1', 'angle_for_ardhaChandrasana2',
    'hand_angle', 'left_hip_angle', 'right_hip_angle', 'neck_angle_uk',
    'left_wrist_angle_bk', 'right_wrist_angle_bk'
]
X = df[angle_columns]

# 1.2 Definir Target BINARIO (0=Good, 1=Bad)
# Agrupamos 'subtle' y 'obvious' en una sola clase 'Bad' para la visualización de clusters.
df['Is_Bad'] = df['Error_Type'].apply(lambda x: 0 if x == 'good' else 1)
y_binary = df['Is_Bad']

# ============================================
# 2. ESCALADO (CRÍTICO para PCA)
# ============================================
# PCA es sensible a la escala, por lo que estandarizamos los 14 ángulos.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 3. APLICACIÓN DE PCA
# ============================================
# Reducir los 14 ángulos a 2 Componentes Principales (PC1 y PC2)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Crear un DataFrame para el gráfico
pca_df = pd.DataFrame(
    data=principal_components, 
    columns=['PC1', 'PC2']
)
# Añadir la etiqueta binaria
pca_df['Target'] = y_binary.map({0: 'Good', 1: 'Bad'}) 

# ============================================
# 4. GRÁFICO DE DISPERSIÓN (SCATTER PLOT)
# ============================================
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue='Target', 
    data=pca_df, 
    palette=['#4daf4a', '#e41a1c'], # Colores: Verde para Good, Rojo para Bad
    s=50, 
    alpha=0.7
)

# Etiquetas y Título
plt.title('Visualización de Clusters (PCA) - Poses Good vs. Bad', fontsize=16)
plt.xlabel(f'Primer Componente Principal (PC1 - {pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'Segundo Componente Principal (PC2 - {pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.legend(title='Clase', loc='upper right')
plt.grid(True)

# Guardar y mostrar (asumiendo entorno de notebook o script)
plt.savefig('Dispersion_PCA_Clusters.png', bbox_inches='tight')
plt.show() # Usar si estás en un script local