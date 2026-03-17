import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Configuración estética para gráficos de nivel publicación científica
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
palette = sns.color_palette("husl", 5) # 5 colores para las 5 familias instrumentales

# =====================================================================
# 0. CARGA DE DATOS
# =====================================================================
# Asegúrate de que el nombre del archivo coincida con tu CSV real
df = pd.read_csv('features_huella_teorica_consolidada_irmas.csv')

# =====================================================================
# 1. Visualización Univariante: Histograma de Frecuencia Fundamental (F0)
# =====================================================================
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='F0_mean', hue='Class', element="step", 
             stat="density", common_norm=False, palette=palette, alpha=0.6)
plt.title('Distribución de la Frecuencia Fundamental (F0) por Familia Instrumental', fontsize=14, fontweight='bold')
plt.xlabel('Frecuencia Fundamental Media (Hz)')
plt.ylabel('Densidad')
plt.tight_layout()
plt.savefig('1_histograma_F0.png', dpi=300)
plt.show()

# =====================================================================
# 2. Visualización Multivariante: Boxplot del Centroide Espectral
# =====================================================================
plt.figure(figsize=(10, 6))
# Se añade hue='Class' y legend=False para cumplir con la nueva versión de seaborn
sns.boxplot(data=df, x='Class', y='SpecCentroid', hue='Class', palette=palette, 
            showfliers=False, width=0.6, legend=False)
plt.title('Dispersión del Centroide Espectral (Brillo Acústico)', fontsize=14, fontweight='bold')
plt.xlabel('Familia Instrumental')
plt.ylabel('Centroide Espectral (Hz)')
plt.tight_layout()
plt.savefig('2_boxplot_centroide.png', dpi=300)
plt.show()

# =====================================================================
# 3. Matriz de Correlación: Descriptores Acústicos Básicos
# =====================================================================
plt.figure(figsize=(10, 8))
# Utilizamos exactamente la lista de variables base identificadas en la Fase de Análisis
columnas_basicas = ['F0_mean', 'SpecCentroid', 'RMS', 'SpecRolloff85', 
                    'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'Onset_max']

matriz_corr = df[columnas_basicas].corr()

# Máscara para ocultar la mitad superior (redundante) del mapa de calor
mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))
sns.heatmap(matriz_corr, mask=mascara, annot=True, fmt=".2f", cmap='coolwarm', 
            vmax=1, vmin=-1, square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación de Características Acústicas Básicas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('3_matriz_correlacion.png', dpi=300)
plt.show()

# =====================================================================
# 4. Agrupamiento (Clustering): Proyección t-SNE (Separabilidad de Clases)
# =====================================================================
# Separar etiquetas (y)
y = df['Class']

# Separar características (X) filtrando SOLO las columnas numéricas. 
# Esto evita el error con los nombres de archivo .wav
X = df.drop(columns=['Class']).select_dtypes(include=[np.number])

# Estandarizar los datos (paso obligatorio para que t-SNE funcione bien)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ejecutar t-SNE para reducir la dimensionalidad a 2 ejes espaciales
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)

# Crear DataFrame temporal para la visualización
df_tsne = pd.DataFrame({'Dimensión 1': X_tsne[:, 0], 'Dimensión 2': X_tsne[:, 1], 'Class': y})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_tsne, x='Dimensión 1', y='Dimensión 2', hue='Class', 
                palette=palette, alpha=0.7, s=60, edgecolor=None)
plt.title('Proyección t-SNE del Espacio Acústico (Exploración de Separabilidad)', fontsize=14, fontweight='bold')
plt.legend(title='Familia Instrumental', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('4_tsne_clustering.png', dpi=300)
plt.show()