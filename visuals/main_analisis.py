import pandas as pd
import config_visual as cfg
import visualizaciones as vis

# 0. Carga y Configuración
df = pd.read_csv('features_huella_teorica_consolidada_irmas.csv')
palette = cfg.set_scientific_style()

# 1. Ejecutar Análisis Visual
vis.plot_histograma_f0(df, palette)
vis.plot_boxplot_centroide(df, palette)
vis.plot_matriz_correlacion(df)
vis.plot_tsne_clustering(df, palette)

print("¡Análisis Visual de la Fase de Análisis completado exitosamente!")