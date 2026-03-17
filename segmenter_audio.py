import os
import librosa
import soundfile as sf
import numpy as np

def segmentar_solo(input_path, output_folder, segment_duration=3.0, top_db=30):
    """
    Divide un audio largo en fragmentos de duracion fija,
    eliminando silencios profundos.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Carpeta creada: {output_folder}")

    print(f"⏳ Cargando y normalizando audio: {os.path.basename(input_path)}")
    # Cargamos a la misma tasa de tu proyecto
    y, sr = librosa.load(input_path, sr=44100)

    # 1. Eliminar silencios largos al inicio y final
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

    # 2. Calcular cuántas muestras tiene cada fragmento (3 segundos)
    samples_per_segment = int(segment_duration * sr)
    
    # 3. Dividir y guardar
    total_segments = len(y_trimmed) // samples_per_segment
    print(f"✂️ Segmentando en {total_segments} fragmentos de {segment_duration}s...")

    count = 0
    for i in range(total_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = y_trimmed[start:end]

        # Solo guardar si el segmento tiene energía suficiente (no es puro silencio)
        if np.max(np.abs(segment)) > 0.05:
            output_filename = f"vio_solo_segment_{count:03d}.wav"
            output_path = os.path.join(output_folder, output_filename)
            
            sf.write(output_path, segment, sr)
            count += 1

    print(f"✅ ¡Proceso terminado! {count} archivos guardados en {output_folder}")

if __name__ == "__main__":
    # === CONFIGURA TUS RUTAS AQUÍ ===
    ARCHIVO_LARGO = "data/raw/vio/solo_completo.mp3" 
    CARPETA_DESTINO = "data/raw/vio/nuevos_fragmentos"
    
    segmentar_solo(ARCHIVO_LARGO, CARPETA_DESTINO)