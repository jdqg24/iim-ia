# src/features/extract_features_parallel.py
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import scipy.signal

from src.preprocessing.preprocess_audio import load_and_normalize
from src.utils.audio_utils import (
    compute_temporal_features,
    compute_spectral_features,
    compute_spectral_contrast,
    compute_chroma_features,
    compute_rhythm_and_pitch,
    compute_mfcc_features,
    compute_envelope_stats,
    compute_anti_confusion_features,
    compute_energy_ratios,
    compute_harmonic_purity,
    compute_vibrato_features,
    compute_bow_vs_reed_features,  # <--- Comma añadida aquí
    compute_stats
)

def process_audio(file_path, class_name):
    """
    Procesa un solo archivo de audio y devuelve su vector de características
    junto con su nombre y clase.
    """
    file_name = os.path.basename(file_path)
    y = load_and_normalize(file_path)
    sr = 44100 

    # 🚨 FILTRO PASA-ALTOS (Corta todo por debajo de 100 Hz)
    # Limpia ruidos de viento, golpes de micro y frecuencias basura
    b, a = scipy.signal.butter(N=4, Wn=100 / (sr / 2), btype='high')
    y = scipy.signal.filtfilt(b, a, y)

    # ===== FORZAR PADDING =====
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')

    # 1. Vectores Escalares Temporales y Espectrales (5 + 7 = 12)
    temp_feat = compute_temporal_features(y, sr).tolist()
    spec_feat = compute_spectral_features(y, sr).tolist()
    
    # 2. Pack Anti-Confusión (Ataque y Brillo) (3)
    anti_conf_feat = compute_anti_confusion_features(y, sr).tolist()
    
    # 3. Análisis de Energía y Pureza (1 + 2 = 3)
    energy_ratio_feat = compute_energy_ratios(y, sr).tolist()
    harmonic_purity_feat = compute_harmonic_purity(y, sr).tolist() 
    
    # 4. Características de Envolvente y Vibrato (1 + 3 = 4)
    env_feat = compute_envelope_stats(y).tolist()
    vibrato_feat = compute_vibrato_features(y, sr).tolist() 

    # 5. Características de Cuerda vs Viento (2)
    bow_reed_feat = compute_bow_vs_reed_features(y, sr).tolist()

    # 6. Matrices 2D y Series Temporales
    contrast = compute_spectral_contrast(y, sr)
    chroma_stft, chroma_cens, tonnetz = compute_chroma_features(y, sr)
    onset_env, f0, tempo = compute_rhythm_and_pitch(y, sr)
    mfcc, delta, delta2 = compute_mfcc_features(y, sr)

    # 7. Calcular Estadísticos para aplanar matrices
    contrast_stats = compute_stats(contrast)
    chroma_stft_stats = compute_stats(chroma_stft)
    chroma_cens_stats = compute_stats(chroma_cens)
    tonnetz_stats = compute_stats(tonnetz)
    onset_stats = compute_stats(onset_env)
    f0_stats = compute_stats(f0)
    mfcc_stats = compute_stats(mfcc)
    delta_stats = compute_stats(delta)
    delta2_stats = compute_stats(delta2)

    # 8. Concatenación Final (573 columnas en total)
    final_vector = (
        temp_feat +           # 5
        spec_feat +           # 7
        anti_conf_feat +      # 3
        energy_ratio_feat +   # 1
        harmonic_purity_feat + # 2
        env_feat +            # 1
        vibrato_feat +        # 3 
        bow_reed_feat +       # 2 <--- Cuerdas vs Vientos integrados aquí
        contrast_stats +      # 49
        chroma_stft_stats +   # 84
        chroma_cens_stats +   # 84
        tonnetz_stats +       # 42
        onset_stats +         # 7
        f0_stats +            # 7
        [tempo] +             # 1
        mfcc_stats +          # 91
        delta_stats +         # 91
        delta2_stats          # 91
    )
    
    return [file_name, class_name] + final_vector


def extract_all_features(dataset_dir, output_csv, n_jobs=-1):
    """
    Escanea un directorio estructurado por carpetas de clases y extrae
    las características de todos los audios.
    """
    audio_files = []
    valid_extensions = {'.wav', '.mp3', '.flac', '.ogg'}

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"📁 Carpeta de clase detectada: {class_name}")
            for file in os.listdir(class_dir):
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    file_path = os.path.join(class_dir, file)
                    audio_files.append((file_path, class_name))
                    
    total_audios = len(audio_files)
    clases_encontradas = sorted(list(set([c for _, c in audio_files])))
    
    print(f"🔍 Buscando en: {dataset_dir}")
    print(f"📦 Se encontraron {total_audios} audios. Clases: {clases_encontradas}")

    if total_audios == 0:
        print("❌ No se encontraron audios.")
        return

    # 2. Paralelización
    features_list = Parallel(n_jobs=n_jobs)(
        delayed(process_audio)(file_path, class_name) for file_path, class_name in tqdm(
            audio_files, total=total_audios, desc="Extrayendo features"
        )
    )

    # 3. Generar Nombres de Columnas
    feature_names = []
    stats = ["mean", "std", "max", "min", "range", "skew", "kurt"]

    # Temporales y Espectrales base
    feature_names.extend(["RMS", "ZCR", "TemporalCentroid", "AttackTime_v1", "CrestFactor_v1"])
    feature_names.extend(["SpecCentroid", "SpecBandwidth", "SpecRolloff85", "SpecFlatness", "SpecFlux", "SpecSkew", "SpecKurt"])
    
    # Pack de Viento y Anti-Confusión
    feature_names.extend(["CrestFactor_v2", "AttackTime_v2", "SpecRolloff90"])
    feature_names.append("LowEnergyRatio")
    feature_names.extend(["HarmonicRatio", "HNR"]) 
    feature_names.append("DecayTime")
    feature_names.extend(["Vibrato_Extent", "Vibrato_Speed", "Harmonic_Flatness"]) 
    feature_names.extend(["HighFreq_Energy", "OddEvenRatio"]) # <--- Nombres de las columnas de Cuerda vs Viento

    # Matrices aplanadas
    for i in range(1, 8):
        for s in stats: feature_names.append(f"Contrast_band{i}_{s}")
    for i in range(1, 13):
        for s in stats: feature_names.append(f"ChromaSTFT_{i}_{s}")
    for i in range(1, 13):
        for s in stats: feature_names.append(f"ChromaCENS_{i}_{s}")
    for i in range(1, 7):
        for s in stats: feature_names.append(f"Tonnetz_{i}_{s}")

    # Ritmo y Pitch
    for s in stats: feature_names.append(f"Onset_{s}")
    for s in stats: feature_names.append(f"F0_{s}")
    feature_names.append("Tempo_BPM")

    # MFCCs (Coeficientes, Deltas y Delta-Deltas)
    for prefix in ["MFCC", "MFCC_delta", "MFCC_delta2"]:
        for i in range(1, 14):
            for s in stats:
                feature_names.append(f"{prefix}{i}_{s}")

    cols = ["FileName", "Class"] + feature_names

    # 4. Validación final y Guardado
    if len(features_list[0]) != len(cols):
        print(f"⚠️ Desajuste detectado: Vector={len(features_list[0])}, Columnas={len(cols)}")
        min_len = min(len(features_list[0]), len(cols))
        features_list = [vec[:min_len] for vec in features_list]
        cols = cols[:min_len]

    df_features = pd.DataFrame(features_list, columns=cols)
    df_features.to_csv(output_csv, index=False)
    print(f"\n✅ Extracción exitosa. {len(cols)-2} características guardadas en: {output_csv}")