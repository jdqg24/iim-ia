# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal

from src.preprocessing.preprocess_audio import load_and_normalize
from src.utils.audio_utils import (
    compute_temporal_features, compute_spectral_features,
    compute_spectral_contrast, compute_chroma_features,
    compute_rhythm_and_pitch, compute_mfcc_features, 
    compute_envelope_stats, compute_anti_confusion_features,
    compute_energy_ratios, compute_harmonic_purity,
    compute_vibrato_features, compute_bow_vs_reed_features, # <--- Nuevas armas
    compute_stats
)

# === 1. Configuración de la página ===
st.set_page_config(page_title="Instrument AI", page_icon="🎵", layout="wide")

st.title("Clasificador de Instrumentos Musicales")
st.markdown("""
Esta aplicación utiliza un modelo **XGBoost** de alto rendimiento optimizado con ingeniería de características físicas.
Identifica: **Piano, Guitarra Acústica, Violín, Flauta y Saxofón**.
""")

# === 2. Cargar el Modelo ===
@st.cache_resource
def load_model():
    return joblib.load("src/models/Audio_XGBoost_Model.pkl")

try:
    model_data = load_model()
    pipeline = model_data['pipeline']
    le = model_data['label_encoder']
except Exception as e:
    st.error("No se encontró el modelo. Verifica las rutas en 'src/models/'.")
    st.stop()

# === 3. Alineación de Características (Espejo del Entrenamiento) ===
def get_feature_names():
    """Recrea exactamente los nombres de las 573 columnas originales antes del filtrado."""
    feature_names = []
    stats = ["mean", "std", "max", "min", "range", "skew", "kurt"]

    # Bloque 1: Básicos y Anti-Confusión
    feature_names.extend(["RMS", "ZCR", "TemporalCentroid", "AttackTime_v1", "CrestFactor_v1"])
    feature_names.extend(["SpecCentroid", "SpecBandwidth", "SpecRolloff85", "SpecFlatness", "SpecFlux", "SpecSkew", "SpecKurt"])
    feature_names.extend(["CrestFactor_v2", "AttackTime_v2", "SpecRolloff90"])
    feature_names.append("LowEnergyRatio")
    feature_names.extend(["HarmonicRatio", "HNR"]) 
    feature_names.append("DecayTime")
    
    # Bloque 2: Nuevas Características Físicas
    feature_names.extend(["Vibrato_Extent", "Vibrato_Speed", "Harmonic_Flatness"])
    feature_names.extend(["HighFreq_Energy", "OddEvenRatio"])

    # Bloque 3: Matrices y Series Temporales
    for i in range(1, 8):
        for s in stats: feature_names.append(f"Contrast_band{i}_{s}")
    for i in range(1, 13):
        for s in stats: feature_names.append(f"ChromaSTFT_{i}_{s}")
    for i in range(1, 13):
        for s in stats: feature_names.append(f"ChromaCENS_{i}_{s}")
    for i in range(1, 7):
        for s in stats: feature_names.append(f"Tonnetz_{i}_{s}")

    for s in stats: feature_names.append(f"Onset_{s}")
    for s in stats: feature_names.append(f"F0_{s}")
    feature_names.append("Tempo_BPM")

    for prefix in ["MFCC", "MFCC_delta", "MFCC_delta2"]:
        for i in range(1, 14):
            for s in stats:
                feature_names.append(f"{prefix}{i}_{s}")
    return feature_names

def extract_single_feature_vector(file_path):
    y = load_and_normalize(file_path)
    sr = 44100 
    
    # Filtro Pasa-Altos (Igual que en el entrenamiento)
    b, a = scipy.signal.butter(N=4, Wn=100 / (sr / 2), btype='high')
    y = scipy.signal.filtfilt(b, a, y)

    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')

    # Extracción de todos los bloques
    temp_feat = compute_temporal_features(y, sr).tolist()
    spec_feat = compute_spectral_features(y, sr).tolist()
    anti_conf_feat = compute_anti_confusion_features(y, sr).tolist()
    energy_ratio_feat = compute_energy_ratios(y, sr).tolist()
    harmonic_purity_feat = compute_harmonic_purity(y, sr).tolist() 
    env_feat = compute_envelope_stats(y).tolist()
    vibrato_feat = compute_vibrato_features(y, sr).tolist()
    bow_reed_feat = compute_bow_vs_reed_features(y, sr).tolist()

    contrast = compute_spectral_contrast(y, sr)
    chroma_stft, chroma_cens, tonnetz = compute_chroma_features(y, sr)
    onset_env, f0, tempo = compute_rhythm_and_pitch(y, sr)
    mfcc, delta, delta2 = compute_mfcc_features(y, sr)

    # Concatenación Final (573 features)
    final_vector = (
        temp_feat + spec_feat + anti_conf_feat + energy_ratio_feat + 
        harmonic_purity_feat + env_feat + vibrato_feat + bow_reed_feat +
        compute_stats(contrast) + compute_stats(chroma_stft) + 
        compute_stats(chroma_cens) + compute_stats(tonnetz) + 
        compute_stats(onset_env) + compute_stats(f0) + [tempo] + 
        compute_stats(mfcc) + compute_stats(delta) + compute_stats(delta2)
    )
    
    # 🚨 FILTRO DE COLUMNAS DELTA2 (Alineación con el modelo final) 🚨
    df_temp = pd.DataFrame([final_vector], columns=get_feature_names())
    columnas_a_borrar = [col for col in df_temp.columns if "delta2" in col]
    df_final = df_temp.drop(columns=columnas_a_borrar)
    
    return df_final.values, y, sr

# === 4. Interfaz de Usuario ===
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Entrada de Audio")
    uploaded_file = st.file_uploader("Carga tu archivo de audio (WAV, MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        umbral = st.slider("Sensibilidad (Umbral de Confianza)", 0.0, 1.0, 0.60)
        
        if st.button("🚀 Clasificar Instrumento", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_filepath = tmp_file.name
            
            try:
                with st.spinner('Analizando huella digital del audio...'):
                    X_pred, y_audio, sr_audio = extract_single_feature_vector(tmp_filepath)
                    X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    probabilidades = pipeline.predict_proba(X_pred)[0]
                    idx_ganador = np.argmax(probabilidades)
                    instrumento_raw = le.inverse_transform([idx_ganador])[0]
                    certeza = probabilidades[idx_ganador]

                    instrument_map = {
                        "gac": "Guitarra Acústica", "pia": "Piano", 
                        "vio": "Violín", "flu": "Flauta", "sax": "Saxofón"
                    }

                    if certeza < umbral:
                        st.warning(f"### ⚠️ Confianza Insuficiente ({certeza*100:.1f}%)")
                        st.info("El modelo detecta características mixtas o ruido dominante. Intenta con un audio más aislado.")
                    else:
                        nombre = instrument_map.get(instrumento_raw.lower(), instrumento_raw)
                        st.balloons()
                        st.success(f"### Predicción: **{nombre}**")
                        st.metric("Confianza del Modelo", f"{certeza*100:.2f}%")

                    # Gráfico de barras
                    df_probs = pd.DataFrame({
                        'Instrumento': [instrument_map.get(c.lower(), c) for c in le.classes_],
                        'Confianza (%)': probabilidades * 100
                    }).sort_values(by='Confianza (%)', ascending=True)
                    st.bar_chart(df_probs.set_index('Instrumento'))

                    st.session_state['y_vis'] = y_audio
                    st.session_state['sr_vis'] = sr_audio

            except Exception as e:
                st.error(f"Error técnico: {e}")
            finally:
                if os.path.exists(tmp_filepath): os.remove(tmp_filepath)

with col2:
    st.subheader("Laboratorio de Análisis Visual")
    if 'y_vis' in st.session_state:
        y_vis = st.session_state['y_vis']
        sr_vis = st.session_state['sr_vis']

        st.write("**Espectrograma Mel (Energía vs Tiempo)**")
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y_vis, sr=sr_vis, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_vis, ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        st.pyplot(fig)

        st.write("**Envolvente de Amplitud (Waveform)**")
        fig2, ax2 = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y_vis, sr=sr_vis, ax=ax2, color='skyblue')
        st.pyplot(fig2)
    else:
        st.info("Sube un audio para visualizar su firma acústica.")