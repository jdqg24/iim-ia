import os
import librosa
import numpy as np
import soundfile as sf

# =====================================================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =====================================================================
# Carpeta de entrada: Debe contener subcarpetas por instrumento (ej. audios_crudos/pia/)
input_base_folder = 'audios_crudos/' 

# Carpeta principal de salida
output_base_folder = 'gac/' 

window_length_sec = 3.0  
target_sr = 22050        

# Umbral de decibelios para considerar que hay "silencio"
# (Si el volumen baja más de 30dB respecto al pico máximo, se corta)
umbral_db = 30 

# =====================================================================
# 2. PROCESAMIENTO AUTOMATIZADO POR ENERGÍA
# =====================================================================
print("Iniciando segmentación basada en energía acústica...")
estadisticas = {}

# Recorrer cada subcarpeta (cada instrumento)
for instrumento in os.listdir(input_folder := input_base_folder):
    ruta_instrumento = os.path.join(input_folder, instrumento)
    
    if not os.path.isdir(ruta_instrumento): continue
    
    # Crear carpeta de salida para este instrumento
    out_instrument_folder = os.path.join(output_base_folder, instrumento)
    os.makedirs(out_instrument_folder, exist_ok=True)
    
    estadisticas[instrumento] = 0
    
    # Procesar cada audio largo dentro de la carpeta del instrumento
    for filename in os.listdir(ruta_instrumento):
        if not filename.endswith(('.wav', '.mp3', '.flac', '.ogg')): continue
            
        file_path = os.path.join(ruta_instrumento, filename)
        
        try:
            # 1. Cargar el audio completo
            y, sr = librosa.load(file_path, sr=target_sr)
            
            # 2. MAGIA ACÚSTICA: Detectar intervalos donde el instrumento SÍ suena
            # Devuelve una lista de [inicio, fin] de las partes ruidosas (útiles)
            intervalos_activos = librosa.effects.split(y, top_db=umbral_db)
            
            # 3. Unir solo las partes donde hay música (elimina los silencios del medio)
            y_denso = np.concatenate([y[start:end] for start, end in intervalos_activos])
            
            # 4. Rebanar en ventanas exactas de 3 segundos
            samples_per_window = int(window_length_sec * sr)
            num_windows = len(y_denso) // samples_per_window 
            
            for i in range(num_windows):
                chunk = y_denso[i * samples_per_window : (i + 1) * samples_per_window]
                
                base_name = os.path.splitext(filename)[0]
                out_filename = f"{base_name}_chunk_{i+1:03d}.wav"
                out_filepath = os.path.join(out_instrument_folder, out_filename)
                
                sf.write(out_filepath, chunk, sr)
                estadisticas[instrumento] += 1
                
            print(f"✅ {instrumento.upper()} | {filename} -> {num_windows} audios generados.")
            
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")

# =====================================================================
# 3. REPORTE FINAL DE BALANCEO
# =====================================================================
print("\n" + "="*50)
print("🎉 SEGMENTACIÓN INTELIGENTE COMPLETADA 🎉")
print("="*50)
print("Resumen de fragmentos puros de 3s generados:")
total = 0
for inst, count in estadisticas.items():
    print(f"- {inst.upper()}: {count} audios")
    total += count
print("-" * 20)
print(f"TOTAL: {total} audios listos para extracción.")
print("="*50)