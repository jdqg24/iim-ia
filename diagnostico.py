import pandas as pd

# Cargar el dataset
df = pd.read_csv("data/features_dataset.csv")

# Seleccionar un par de columnas clave para ver si hay diferencias reales
columnas_prueba = ['RMS', 'Tempo_BPM', 'SpecCentroid', 'ZCR']

print("Promedios por clase en el CSV:")
print(df.groupby('Class')[columnas_prueba].mean())

print("\nValores nulos por clase (antes de imputar):")
print(df.groupby('Class')[columnas_prueba].apply(lambda x: x.isna().sum())) 