# predict_audio.py
import os
import sys
import joblib
import numpy as np
from src.preprocessing.preprocess_audio import load_and_normalize
from src.utils.audio_utils import compute_temporal_features, compute_spectral_features, compute_mfcc_features

# ===== Cargar modelo =====
MODEL_PATH = "RandomForest_audio_model.pkl"  # o SVM
saved = joblib.load(MODEL_PATH)
model = saved["model"]
scaler = saved["scaler"]
label_encoder = saved["label_encoder"]

def compute_stats(array):
    """Calcula stats de un array 1D o 2D aplanando correctamente"""
    array = np.array(array)
    if array.ndim == 1:
        rows = [array]
    elif array.ndim == 2:
        rows = array
    stats_list = []
    for row in rows:
        row_mean = np.mean(row)
        row_std = np.std(row)
        row_max = np.max(row)
        row_min = np.min(row)
        row_range = row_max - row_min
        row_skew = np.mean(((row - row_mean) / (row_std+1e-8))**3)
        row_kurt = np.mean(((row - row_mean) / (row_std+1e-8))**4)
        stats_list.extend([row_mean, row_std, row_max, row_min, row_range, row_skew, row_kurt])
    return stats_list

def extract_features_for_prediction(file_path):
    y = load_and_normalize(file_path)
    
    # Temporal, Spectral, MFCC
    temp_feat = compute_temporal_features(y, sr=44100)
    spec_feat = compute_spectral_features(y, sr=44100)
    mfcc, delta, delta2 = compute_mfcc_features(y, sr=44100)
    
    # Stats
    temp_stats = compute_stats(temp_feat)
    spec_stats = compute_stats(spec_feat)
    mfcc_stats = compute_stats(mfcc)
    delta_stats = compute_stats(delta)
    delta2_stats = compute_stats(delta2)
    
    final_vector = temp_stats + spec_stats + mfcc_stats + delta_stats + delta2_stats
    return np.array(final_vector).reshape(1, -1)

def predict_audio(file_path):
    features = extract_features_for_prediction(file_path)
    features_scaled = scaler.transform(features)
    pred_encoded = model.predict(features_scaled)
    pred_label = label_encoder.inverse_transform(pred_encoded)
    return pred_label[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict_audio.py ruta_al_audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.isfile(audio_file):
        print(f"Archivo no encontrado: {audio_file}")
        sys.exit(1)
    
    prediction = predict_audio(audio_file)
    print(f"Instrumento predicho: {prediction}")