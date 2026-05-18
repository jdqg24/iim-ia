import pandas as pd
import numpy as np

# 1. Cargar las matrices de características de cada dataset
# Sustituye con las rutas reales de tus archivos CSV
df_mono = pd.read_csv('data_v1/features_dataset.csv')
df_poly = pd.read_csv('IRMAS_Data/features_dataset.csv')

# 2. Definir los nombres exactos de tus columnas
COLUMNA_CLASE = 'Class' # Ej: 'piano', 'flauta', 'violin', etc.
COLUMNA_F0 = 'F0_mean'  # Cambia por el nombre real de tu columna de F0
COLUMNA_CENTROIDE = 'SpecCentroid' # Cambia por el nombre de tu columna

# 3. Función para calcular las métricas exigidas
def calcular_estadisticos(df, nombre_caracteristica):
    # Agrupar por clase y calcular métricas
    stats = df.groupby(COLUMNA_CLASE).agg(
        Media=(nombre_caracteristica, 'mean'),
        Desv_Est=(nombre_caracteristica, 'std'),
        Q1=(nombre_caracteristica, lambda x: x.quantile(0.25)),
        Q3=(nombre_caracteristica, lambda x: x.quantile(0.75))
    )
    
    # Calcular el IQR (Rango Intercuartílico = Q3 - Q1)
    stats['IQR'] = stats['Q3'] - stats['Q1']
    
    # Redondear a 2 decimales para que se vea bien en formato APA
    stats = stats.round(2)
    
    # Retornar solo las columnas que van en la tesis
    return stats[['Media', 'Desv_Est', 'IQR']]

# 4. Calcular los valores para la Tabla de Frecuencia Fundamental (F0)
print("=== TABLA X: ESTADÍSTICOS DE FRECUENCIA FUNDAMENTAL (F0) ===")
stats_f0_mono = calcular_estadisticos(df_mono, COLUMNA_F0)
stats_f0_poly = calcular_estadisticos(df_poly, COLUMNA_F0)

# Unir ambos resultados para facilitar copiar y pegar
tabla_f0_final = pd.merge(
    stats_f0_mono, stats_f0_poly, 
    on=COLUMNA_CLASE, 
    suffixes=(' (Mono)', ' (Poli)')
)
print(tabla_f0_final)
print("\n")

# 5. Calcular los valores para la Tabla del Centroide Espectral
print("=== TABLA Y: ESTADÍSTICOS DE CENTROIDE ESPECTRAL ===")
stats_centroide_mono = calcular_estadisticos(df_mono, COLUMNA_CENTROIDE)
stats_centroide_poly = calcular_estadisticos(df_poly, COLUMNA_CENTROIDE)

tabla_centroide_final = pd.merge(
    stats_centroide_mono, stats_centroide_poly, 
    on=COLUMNA_CLASE, 
    suffixes=(' (Mono)', ' (Poli)')
)
print(tabla_centroide_final)