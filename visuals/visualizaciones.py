import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd

# ==========================================
# CONFIGURACIÓN EXTREMA DE VISIBILIDAD
# (Lienzo pequeño + Fuentes Gigantes)
# ==========================================
plt.rcParams.update({
    'font.size': 14,           # Fuente base gigante
    'axes.titlesize': 18,      # Títulos enormes y destacables
    'axes.labelsize': 16,      # Etiquetas de los ejes muy visibles
    'xtick.labelsize': 14,     # Números del eje X grandes
    'ytick.labelsize': 14,     # Números del eje Y grandes
    'legend.fontsize': 14,     # Textos de la leyenda
    'legend.title_fontsize': 16, # Título de la leyenda
    'axes.linewidth': 1.5      # Bordes del gráfico más gruesos
})

def plot_histograma_f0(df, palette, filename='1_histograma_F0.png'):
    plt.figure(figsize=(6, 4)) 
    
    # bw_adjust=0.5 para definir aún más los picos
    sns.kdeplot(data=df, x='F0_mean', hue='Class', common_norm=False, 
                palette=palette, fill=True, alpha=0.35, linewidth=3.0,
                bw_adjust=0.5, legend=True) 
    
    plt.title('Distribución F0', fontweight='bold', pad=15)
    plt.xlabel('Frecuencia Fundamental (Hz)')
    plt.ylabel('Densidad')
    
    # Zoom extremo: Cortamos en 1000 Hz para estirar las curvas bajas
    plt.xlim(50, 650)
    
    # Corrección de la leyenda: Usamos la función nativa de seaborn
    sns.move_legend(plt.gca(), "upper right", title='Clase', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_boxplot_centroide(df, palette, filename='2_boxplot_centroide.png'):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Class', y='SpecCentroid', hue='Class', palette=palette, 
                showfliers=False, width=0.7, linewidth=2.5, legend=False)
    
    plt.title('Centroide Espectral', fontweight='bold', pad=15)
    plt.xlabel('Instrumento')
    plt.ylabel('Centroide (Hz)')
    
    # Ampliamos el "techo" a 3600 y el piso a 200 para no cortar los bigotes
    plt.ylim(200, 4500)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_matriz_correlacion(df, filename='3_matriz_correlacion.png'):
    plt.figure(figsize=(7, 6)) # Un poco más ancha para que quepan los textos gigantes
    columnas = ['F0_mean', 'SpecCentroid', 'RMS', 'SpecRolloff85', 
                'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'Onset_max']
    matriz_corr = df[columnas].corr()
    mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))
    
    # annot_kws={"size": 14} hace que los números dentro de los cuadros sean enormes
    sns.heatmap(matriz_corr, mask=mascara, annot=True, fmt=".2f", cmap='coolwarm', 
                vmax=1, vmin=-1, square=True, linewidths=1.0, linecolor='white',
                annot_kws={"size": 14, "weight": "bold"}, cbar_kws={"shrink": .8})
    
    plt.title('Correlación de Características', fontweight='bold', pad=15)
    
    # Rotar las etiquetas en 45 grados y alinearlas a la derecha para evitar choques
    plt.xticks(rotation=45, ha='right', fontweight='bold') 
    plt.yticks(rotation=0, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_tsne_clustering(df, palette, filename='4_tsne_clustering.png'):
    y = df['Class']
    X = df.drop(columns=['Class']).select_dtypes(include=[np.number])
    
    X_scaled = StandardScaler().fit_transform(X)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca').fit_transform(X_scaled)
    
    df_tsne = pd.DataFrame({'Dimensión 1': X_tsne[:, 0], 'Dimensión 2': X_tsne[:, 1], 'Class': y})
    
    # Lienzo un poco más alto para que la leyenda gigante quepa abajo sin estorbar
    plt.figure(figsize=(6, 5.5)) 
    
    # Puntos masivos (s=120) para que se distingan los colores incluso de lejos
    sns.scatterplot(data=df_tsne, x='Dimensión 1', y='Dimensión 2', hue='Class', 
                    palette=palette, alpha=0.8, s=120, edgecolor='black', linewidth=0.8) 
    
    plt.title('t-SNE: Espacio Acústico', fontweight='bold', pad=15)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    # Leyenda abajo, en una sola fila (ncol=5)
    plt.legend(title='Clase', bbox_to_anchor=(0.5, -0.2), loc='upper center', 
               ncol=5, frameon=True, borderpad=0.8) 
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()