import os
import pandas as pd
import numpy as np
import librosa
import random

# =====================================================================
# 1. CONFIGURACIÓN MULTI-CLASE
# =====================================================================
# Diccionario con las rutas de las carpetas y su respectiva etiqueta.
# Ajusta las rutas según tu estructura local.
carpetas_clases = {
    'IRMAS_Data/pia/': 'pia',  # Piano
    'IRMAS_Data/vio/': 'vio',  # Violín
    'IRMAS_Data/flu/': 'flu',  # Flauta
    'IRMAS_Data/gac/': 'gac',  # Guitarra Acústica
    'IRMAS_Data/sax/': 'sax'   # Saxofón
}

output_csv = 'features_huella_teorica_consolidada_irmas.csv'
limite_muestras = 300  # Límite por clase (Total esperado: 1500 muestras)

datos_extraidos = []

# =====================================================================
# 2. EXTRACCIÓN Y SUBMUESTREO ALEATORIO ITERATIVO
# =====================================================================
print(f"Iniciando extracción masiva para {len(carpetas_clases)} clases...")

for input_folder, etiqueta_clase in carpetas_clases.items():
    print("\n" + "-"*50)
    print(f"Procesando clase: [{etiqueta_clase.upper()}] en {input_folder}")
    print("-"*50)

    # Verificar si la carpeta existe para evitar caídas del script
    if not os.path.exists(input_folder):
        print(f"⚠️ La carpeta {input_folder} no existe. Saltando clase...")
        continue

    # Obtener todos los archivos .wav
    todos_los_archivos = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    # Validar si hay suficientes archivos
    if len(todos_los_archivos) < limite_muestras:
        print(f"⚠️ Advertencia: Solo hay {len(todos_los_archivos)} audios (se esperaban {limite_muestras}).")
    
    # Submuestreo
    archivos_seleccionados = random.sample(todos_los_archivos, min(limite_muestras, len(todos_los_archivos)))
    archivos_procesados = 0

    for filename in archivos_seleccionados:
        filepath = os.path.join(input_folder, filename)
        
        try:
            # Carga del audio
            y, sr = librosa.load(filepath, sr=22050)
            
            # Extracción F0 (pYIN)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0
            
            # Descriptores Espectrales y de Energía
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y))
            rolloff85 = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
            
            # Envolvente de Ataque (Onset)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_max = np.max(onset_env)
            
            # Construcción de la fila de datos
            fila = {
                'filename': filename, 'Class': etiqueta_clase, 'F0_mean': f0_mean,
                'SpecCentroid': spec_centroid, 'RMS': rms, 'SpecRolloff85': rolloff85,
                'MFCC1_mean': np.mean(mfccs[0]), 'MFCC2_mean': np.mean(mfccs[1]), 
                'MFCC3_mean': np.mean(mfccs[2]), 'Onset_max': onset_max
            }
            
            datos_extraidos.append(fila)
            archivos_procesados += 1
            
            # Feedback en consola cada 50 audios para no saturar la terminal
            if archivos_procesados % 50 == 0:
                print(f"  ⏳ [{etiqueta_clase.upper()}] Procesados {archivos_procesados}/{len(archivos_seleccionados)} audios...")
                
        except Exception as e:
            print(f"  ❌ Error en {filename}: {e}")
            
    print(f"✅ Clase {etiqueta_clase.upper()} completada con {archivos_procesados} muestras viables.")

# =====================================================================
# 3. EXPORTACIÓN CONSOLIDADA
# =====================================================================
df_features = pd.DataFrame(datos_extraidos)
df_features.to_csv(output_csv, index=False)

print("\n" + "="*50)
print(f"🎉 EXTRACCIÓN MASIVA COMPLETADA 🎉")
print(f"📊 Total de muestras en el dataset maestro: {len(df_features)}")
print(f"💾 Guardado exitosamente en: {output_csv}")
print("="*50)