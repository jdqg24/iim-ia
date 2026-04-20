import os
import random

# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
# Lista de las carpetas que quieres balancear
carpetas_objetivo = ['../IRMAS_Data/pia', '../IRMAS_Data/vio', '../IRMAS_Data/sax', '../IRMAS_Data/flu', '../IRMAS_Data/gac']  # Ajusta según tus nombres de carpeta

# Número exacto de audios que quieres conservar por carpeta
objetivo_muestras = 400

# =====================================================================
# 2. PROCESO DE SUBMUESTREO ALEATORIO
# =====================================================================
print("Iniciando balanceo de clases (Random Subsampling)...\n")

for carpeta in carpetas_objetivo:
    if not os.path.exists(carpeta):
        print(f"⚠️ La carpeta '{carpeta}' no existe. Saltando...")
        continue
        
    # Obtener solo los archivos de audio (ignoramos subcarpetas u otros archivos)
    archivos = [f for f in os.listdir(carpeta) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    total_archivos = len(archivos)
    
    print(f"📁 Analizando '{carpeta}': Se encontraron {total_archivos} audios.")
    
    if total_archivos > objetivo_muestras:
        # Calcular cuántos hay que eliminar
        excedente = total_archivos - objetivo_muestras
        
        # Seleccionar ALEATORIAMENTE los archivos que serán eliminados
        archivos_a_eliminar = random.sample(archivos, excedente)
        
        # Eliminar los archivos seleccionados
        eliminados_count = 0
        for archivo in archivos_a_eliminar:
            ruta_completa = os.path.join(carpeta, archivo)
            try:
                os.remove(ruta_completa)
                eliminados_count += 1
            except Exception as e:
                print(f"❌ Error al eliminar {archivo}: {e}")
                
        print(f"✅ Se eliminaron {eliminados_count} audios al azar.")
        print(f"🎯 La carpeta '{carpeta}' ahora tiene exactamente {objetivo_muestras} audios.\n")
        
    elif total_archivos == objetivo_muestras:
        print(f"✨ La carpeta '{carpeta}' ya tiene exactamente {objetivo_muestras} audios. No se requiere acción.\n")
        
    else:
        print(f"⚠️ Atención: La carpeta '{carpeta}' tiene SOLO {total_archivos} audios (menos del objetivo de {objetivo_muestras}). No se eliminó nada.\n")

print("="*50)
print("🎉 BALANCEO DE CLASES COMPLETADO 🎉")
print("="*50)