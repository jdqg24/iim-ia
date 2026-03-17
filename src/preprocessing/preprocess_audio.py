# src/preprocessing/preprocess_audio.py
import librosa
import numpy as np

SAMPLE_RATE = 44100
MIN_SAMPLES = 2048  # Tamaño mínimo para evitar warnings con n_fft=2048

def load_and_normalize(file_path, sr=SAMPLE_RATE):
    """
    Carga un archivo de audio, lo convierte a mono y normaliza la amplitud.
    Rellena con ceros (zero-padding) si el audio es demasiado corto para las ventanas de FFT.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    
    # Normalización de amplitud
    if len(y) > 0 and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
        
    # ===== SOLUCIÓN AL WARNING DE n_fft =====
    # Si el audio es más corto que nuestra ventana de análisis, 
    # rellenamos el final con silencios (ceros)
    if len(y) < MIN_SAMPLES:
        padding_needed = MIN_SAMPLES - len(y)
        y = np.pad(y, (0, padding_needed), mode='constant')
        
    return y