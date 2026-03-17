import seaborn as sns

def set_scientific_style():
    """Configuración estética global para nivel publicación científica."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    # Retornamos la paleta para usarla en los módulos
    return sns.color_palette("husl", 5)