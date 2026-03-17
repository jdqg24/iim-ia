# src/utils/audio_utils.py
import numpy as np
import librosa

# ===== Temporal Features =====
def compute_temporal_features(y, sr):
    """
    Calcula características temporales básicas de la señal:
    RMS, Zero-Crossing Rate, Temporal Centroid, Attack Time y Crest Factor
    """
    rms = np.sqrt(np.mean(y**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.sum(np.arange(len(y)) * y**2) / (np.sum(y**2) + 1e-8)
    peak = np.max(np.abs(y))
    attack_time = np.argmax(np.abs(y)) / sr
    crest_factor = peak / (rms + 1e-8)
    return np.array([rms, zcr, centroid, attack_time, crest_factor])


# ===== Spectral Features =====
def compute_spectral_features(y, sr):
    """
    Calcula características espectrales generales (promediadas):
    Spectral Centroid, Bandwidth, Rolloff, Flatness, Flux, Skew y Kurtosis
    """
    spec = np.abs(librosa.stft(y, n_fft=2048))
    centroid = np.mean(librosa.feature.spectral_centroid(S=spec, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=spec, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=0.85))
    flatness = np.mean(librosa.feature.spectral_flatness(S=spec))
    flux = np.mean(np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0)))
    skew = np.mean(spec.mean(axis=1) - np.mean(spec))
    kurt = np.mean((spec.mean(axis=1) - np.mean(spec))**4)
    return np.array([centroid, bandwidth, rolloff, flatness, flux, skew, kurt])


# ===== Advanced Spectral Features =====
def compute_spectral_contrast(y, sr):
    """
    Calcula el contraste espectral (diferencia de energía entre picos y valles).
    Devuelve una matriz 2D (n_bands x tiempo).
    """
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    return contrast


# ===== Tonal & Harmonic Features =====
def compute_chroma_features(y, sr):
    """
    Calcula características que representan el contenido armónico/tonal.
    Devuelve matrices 2D a lo largo del tiempo.
    """
    # Chroma STFT (Energía de los 12 tonos)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Chroma CENS (Variante suavizada y normalizada)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    
    # Tonnetz (Centroides tonales). Requiere la componente armónica para no fallar con ruido.
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    
    return chroma_stft, chroma_cens, tonnetz


# ===== Rhythm & Pitch Features =====
def compute_rhythm_and_pitch(y, sr):
    """
    Calcula la frecuencia fundamental (Pitch), la fuerza de inicio (Onset) 
    y estima el Tempo global (BPM).
    """
    # Onset strength (Fuerza de los "golpes" o inicios de nota)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Tempo (BPM) - Devuelve un escalar, así que lo dejamos como un valor único
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    # Pitch (F0) usando el algoritmo YIN. 
    # Sustituimos los NaNs (momentos de silencio/ruido sin tono) por 0.
    f0 = librosa.yin(y, fmin=50, fmax=2000)
    f0 = np.nan_to_num(f0)
    
    return onset_env, f0, tempo


# ===== MFCC Features =====
def compute_mfcc_features(y, sr, n_mfcc=13):
    """
    Calcula MFCCs y sus derivadas (delta y delta-delta)
    """
    mfccs = np.array(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc))
    delta = np.array(librosa.feature.delta(mfccs))
    delta2 = np.array(librosa.feature.delta(mfccs, order=2))
    return mfccs, delta, delta2

def compute_envelope_stats(y):
    """
    Calcula la tasa de decaimiento. Un piano decae mucho más rápido que un saxofón.
    """
    # Calculamos la envolvente simple
    envelope = np.abs(y)
    # Buscamos dónde está el pico máximo
    max_idx = np.argmax(envelope)
    # Medimos cuánto tiempo tarda en caer a la mitad de su energía después del pico
    decay_segment = envelope[max_idx:]
    if len(decay_segment) > 1:
        half_power = decay_segment[0] / 2
        # Buscamos el primer punto por debajo de la mitad
        decay_time = np.argmax(decay_segment < half_power)
    else:
        decay_time = 0
    return np.array([decay_time])


# ===== Estadísticos =====
def compute_stats(array):
    """
    Calcula estadísticas de un array 1D o 2D:
    mean, std, max, min, range, skew, kurtosis
    """
    array = np.array(array)
    if array.ndim == 1:
        rows = [array]
    else:
        rows = array  # cada fila es un coeficiente MFCC u otra feature 2D

    stats_list = []
    for row in rows:
        mean = np.mean(row)
        std = np.std(row)
        max_ = np.max(row)
        min_ = np.min(row)
        range_ = max_ - min_
        skew = np.mean(((row - mean) / (std + 1e-8)) ** 3)
        kurt = np.mean(((row - mean) / (std + 1e-8)) ** 4)
        stats_list.extend([mean, std, max_, min_, range_, skew, kurt])

    return stats_list

def compute_energy_ratios(y, sr):
    # Calculamos el espectro de potencia
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Energía en bajos (0-200 Hz) - Típico de Piano y Guitarra
    low_mask = freqs < 200
    low_energy = np.sum(S[low_mask, :])
    
    # Energía total
    total_energy = np.sum(S) + 1e-8
    
    # Ratio de energía baja
    low_ratio = low_energy / total_energy
    
    return np.array([low_ratio])

def compute_harmonic_purity(y, sr):
    # Separamos la señal en componente armónica y percusiva
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # 1. Ratio Armónico: ¿Qué porcentaje del sonido es música pura?
    harmonic_energy = np.sum(y_harmonic**2)
    total_energy = np.sum(y**2) + 1e-8
    harmonic_ratio = harmonic_energy / total_energy
    
    # 2. HNR (Harmonic-to-Noise Ratio) simplificado
    # Comparamos la energía armónica contra la percusiva/ruido
    noise_energy = np.sum(y_percussive**2) + 1e-8
    hnr = 10 * np.log10(harmonic_energy / noise_energy + 1e-8)
    
    return np.array([harmonic_ratio, hnr])

def compute_vibrato_features(y, sr):
    # Extraemos F0
    f0 = librosa.yin(y, fmin=65, fmax=2000) # Rango estándar de instrumentos
    f0 = np.nan_to_num(f0)
    
    active_f0 = f0[f0 > 0]
    if len(active_f0) < 2:
        return np.array([0.0, 0.0, 0.0])

    # 1. Extensión (Cents): Cuánto oscila la nota
    f0_cents = 1200 * np.log2(active_f0 / 10.0)
    vibrato_extent = np.std(f0_cents)
    
    # 2. Velocidad: Rapidez de los cambios
    vibrato_speed = np.mean(np.abs(np.diff(active_f0)))
    
    # 3. Estabilidad armónica
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    return np.array([vibrato_extent, vibrato_speed, spec_flatness])

def compute_bow_vs_reed_features(y, sr):
    """
    Características diseñadas para separar Cuerdas Frotadas (Violín) 
    de Vientos de Caña (Saxo/Clarinete).
    """
    # 1. Fricción del Arco (Bow Scrape) vs Ruido de Aire
    # Calculamos la energía en frecuencias muy altas (>8000 Hz)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    high_freq_mask = freqs > 8000
    # Ratio de energía aguda respecto a la energía total
    high_freq_energy = np.sum(S[high_freq_mask, :]) / (np.sum(S) + 1e-6)
    
    # 2. Ratio de Armónicos Impares vs Pares (OER)
    # Extraemos F0 para buscar los armónicos
    f0_series = librosa.yin(y, fmin=130, fmax=2000)
    f0_valid = f0_series[f0_series > 0]
    
    if len(f0_valid) == 0:
        return np.array([high_freq_energy, 0.0])
        
    f0 = np.median(f0_valid)
    
    odd_energy = 0.0
    even_energy = 0.0
    
    # Analizamos hasta el 6to armónico
    for i in range(2, 7): # Empezamos en 2 para saltar la fundamental
        h_freq = f0 * i
        if h_freq > sr / 2:
            break
            
        # Buscamos el bin de frecuencia más cercano al armónico
        idx = np.argmin(np.abs(freqs - h_freq))
        # Calculamos la energía local de ese armónico
        energy = np.mean(S[max(0, idx-1):min(len(freqs), idx+2), :])
        
        if i % 2 == 0:
            even_energy += energy # Armónicos 2, 4, 6
        else:
            odd_energy += energy  # Armónicos 3, 5
            
    oer = odd_energy / (even_energy + 1e-6)
    
    return np.array([high_freq_energy, oer])

def compute_anti_confusion_features(y, sr):
    """
    Añade características para diferenciar el ataque percusivo (Piano)
    del soplido/vibrato (Saxo/Flauta/Violín).
    """
    # 1. Crest Factor: Mide la "picudez" de la onda. Un piano tiene picos más altos.
    rms = np.sqrt(np.mean(y**2))
    crest_factor = np.max(np.abs(y)) / (rms + 1e-8)
    
    # 2. Log Attack Time: El tiempo que tarda en alcanzar el pico desde que empieza a sonar.
    # El piano es casi instantáneo. El saxo o violín pueden ser más lentos.
    attack_time = np.argmax(np.abs(y)) / sr
    
    # 3. Spectral Rolloff (90%): Mide la energía en las frecuencias más agudas.
    # Los instrumentos de percusión (piano/guitarra) suelen tener más rolloff.
    rolloff_90 = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.90))
    
    return np.array([crest_factor, attack_time, rolloff_90])