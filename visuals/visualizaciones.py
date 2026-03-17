import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd

def plot_histograma_f0(df, palette):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='F0_mean', hue='Class', element="step", 
                 stat="density", common_norm=False, palette=palette, alpha=0.6)
    plt.title('Distribución de la Frecuencia Fundamental (F0)', fontsize=14, fontweight='bold')
    plt.xlabel('Frecuencia Fundamental Media (Hz)')
    plt.ylabel('Densidad')
    plt.tight_layout()
    plt.savefig('1_histograma_F0.png', dpi=300)
    plt.show()

def plot_boxplot_centroide(df, palette):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Class', y='SpecCentroid', hue='Class', palette=palette, 
                showfliers=False, width=0.6, legend=False)
    # plt.yscale('log') # Activar si se desea menos 'aplastamiento'
    plt.title('Dispersión del Centroide Espectral (Brillo Acústico)', fontsize=14, fontweight='bold')
    plt.xlabel('Familia Instrumental')
    plt.ylabel('Centroide Espectral (Hz)')
    plt.tight_layout()
    plt.savefig('2_boxplot_centroide.png', dpi=300)
    plt.show()

def plot_matriz_correlacion(df):
    plt.figure(figsize=(10, 8))
    columnas = ['F0_mean', 'SpecCentroid', 'RMS', 'SpecRolloff85', 
                'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'Onset_max']
    matriz_corr = df[columnas].corr()
    mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))
    sns.heatmap(matriz_corr, mask=mascara, annot=True, fmt=".2f", cmap='coolwarm', 
                vmax=1, vmin=-1, square=True, linewidths=.5)
    plt.title('Matriz de Correlación de Características', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('3_matriz_correlacion.png', dpi=300)
    plt.show()

def plot_tsne_clustering(df, palette):
    y = df['Class']
    X = df.drop(columns=['Class']).select_dtypes(include=[np.number])
    
    X_scaled = StandardScaler().fit_transform(X)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca').fit_transform(X_scaled)
    
    df_tsne = pd.DataFrame({'Dimensión 1': X_tsne[:, 0], 'Dimensión 2': X_tsne[:, 1], 'Class': y})
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='Dimensión 1', y='Dimensión 2', hue='Class', 
                    palette=palette, alpha=0.7, s=60)
    plt.title('Proyección t-SNE: Separabilidad del Espacio Acústico', fontsize=14, fontweight='bold')
    plt.legend(title='Familia Instrumental', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('4_tsne_clustering.png', dpi=300)
    plt.show()