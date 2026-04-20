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
import seaborn as sns

from src.preprocessing.preprocess_audio import load_and_normalize
from src.utils.audio_utils import (
    compute_temporal_features, compute_spectral_features,
    compute_spectral_contrast, compute_chroma_features,
    compute_rhythm_and_pitch, compute_mfcc_features, 
    compute_envelope_stats, compute_anti_confusion_features,
    compute_energy_ratios, compute_harmonic_purity,
    compute_vibrato_features, compute_bow_vs_reed_features,
    compute_stats
)

# === 1. Configuración de la página (Estilo Corporativo) ===
st.set_page_config(page_title="Sistema de Inferencia Acústica", layout="wide")

# Estilo global para gráficos Matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.linewidth': 1.0,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#ced4da',
    'grid.color': '#e9ecef'
})

st.title("Sistema de Identificación Acústica de Instrumentos")
st.markdown("""
Plataforma analítica fundamentada en **Extreme Gradient Boosting (XGBoost)** para la clasificación multiclase de señales acústicas. 
El motor de inferencia opera sobre un vector de características multidominio para aislar la huella acústica en entornos aislados y polifónicos.
""")
st.divider()

# === 2. Cargar el Modelo ===
@st.cache_resource
def load_model():
    return joblib.load("src/models/Audio_XGBoost_Model.pkl")

try:
    model_data = load_model()
    pipeline = model_data['pipeline']
    le = model_data['label_encoder']
except Exception as e:
    st.error("Excepción de I/O: No se localizó el archivo del modelo predictivo en el directorio 'src/models/'.")
    st.stop()

# === 3. Alineación de Características ===
def get_feature_names():
    feature_names = []
    stats = ["mean", "std", "max", "min", "range", "skew", "kurt"]

    feature_names.extend(["RMS", "ZCR", "TemporalCentroid", "AttackTime_v1", "CrestFactor_v1"])
    feature_names.extend(["SpecCentroid", "SpecBandwidth", "SpecRolloff85", "SpecFlatness", "SpecFlux", "SpecSkew", "SpecKurt"])
    feature_names.extend(["CrestFactor_v2", "AttackTime_v2", "SpecRolloff90"])
    feature_names.append("LowEnergyRatio")
    feature_names.extend(["HarmonicRatio", "HNR"]) 
    feature_names.append("DecayTime")
    
    feature_names.extend(["Vibrato_Extent", "Vibrato_Speed", "Harmonic_Flatness"])
    feature_names.extend(["HighFreq_Energy", "OddEvenRatio"])

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
    
    b, a = scipy.signal.butter(N=4, Wn=100 / (sr / 2), btype='high')
    y = scipy.signal.filtfilt(b, a, y)

    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')

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

    final_vector = (
        temp_feat + spec_feat + anti_conf_feat + energy_ratio_feat + 
        harmonic_purity_feat + env_feat + vibrato_feat + bow_reed_feat +
        compute_stats(contrast) + compute_stats(chroma_stft) + 
        compute_stats(chroma_cens) + compute_stats(tonnetz) + 
        compute_stats(onset_env) + compute_stats(f0) + [tempo] + 
        compute_stats(mfcc) + compute_stats(delta) + compute_stats(delta2)
    )
    
    df_temp = pd.DataFrame([final_vector], columns=get_feature_names())
    columnas_a_borrar = [col for col in df_temp.columns if "delta2" in col]
    df_final = df_temp.drop(columns=columnas_a_borrar)
    
    return df_final.values, y, sr

# === 4. Interfaz de Usuario (Dashboard Ejecutivo) ===

# Diccionario de mapeo
instrument_map = {
    "gac": "Guitarra Acústica", "pia": "Piano", 
    "vio": "Violín", "flu": "Flauta", "sax": "Saxofón"
}

# Barra lateral para controles
with st.sidebar:
    st.subheader("Parámetros de Entrada")
    uploaded_file = st.file_uploader("Archivo fuente (WAV, MP3)", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
    st.markdown("---")
    st.subheader("Configuración Algorítmica")
    umbral = st.slider("Tolerancia de Confianza Mínima", 0.0, 1.0, 0.60, 0.05, 
                       help="Define el porcentaje mínimo de probabilidad requerido para emitir un veredicto válido.")
    
    ejecutar = st.button("Ejecutar Análisis Predictivo", use_container_width=True, type="primary")

# Lienzo principal
if uploaded_file is not None and ejecutar:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name
    
    try:
        with st.spinner('Extrayendo descriptores acústicos y evaluando modelo...'):
            X_pred, y_audio, sr_audio = extract_single_feature_vector(tmp_filepath)
            X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
            
            probabilidades = pipeline.predict_proba(X_pred)[0]
            idx_ganador = np.argmax(probabilidades)
            instrumento_raw = le.inverse_transform([idx_ganador])[0]
            certeza = probabilidades[idx_ganador]
            
            nombre_instrumento = instrument_map.get(instrumento_raw.lower(), instrumento_raw)

        # --- SECCIÓN 1: REPORTE DE INFERENCIA ---
        st.subheader("Reporte de Inferencia")
        
        if certeza < umbral:
            st.warning(f"Diagnóstico Inconcluso: La certeza algorítmica ({certeza*100:.2f}%) no supera el umbral establecido del {umbral*100:.0f}%. Es probable que exista un alto nivel de enmascaramiento armónico.")
        else:
            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric(label="Clase Identificada", value=nombre_instrumento)
            col_met2.metric(label="Nivel de Certeza", value=f"{certeza*100:.2f} %")
            col_met3.metric(label="Tasa de Muestreo", value=f"{sr_audio} Hz")
            
            st.success("Clasificación completada satisfactoriamente según los parámetros de la matriz analítica.")

        # --- SECCIÓN 2: DISTRIBUCIÓN DE PROBABILIDAD MULTICLASE ---
        st.markdown("#### Función de Probabilidad Multiclase")
        
        df_probs = pd.DataFrame({
            'Instrumento': [instrument_map.get(c.lower(), c) for c in le.classes_],
            'Probabilidad (%)': probabilidades * 100
        }).sort_values(by='Probabilidad (%)', ascending=True)

        fig_bar, ax_bar = plt.subplots(figsize=(10, 3))
        ax_bar.barh(df_probs['Instrumento'], df_probs['Probabilidad (%)'], color='#2c7da0')
        ax_bar.set_xlabel('Nivel de Confianza (%)')
        ax_bar.set_xlim(0, 100)
        ax_bar.grid(axis='x', linestyle='--', alpha=0.7)
        for spine in ['top', 'right']:
            ax_bar.spines[spine].set_visible(False)
            
        st.pyplot(fig_bar)

        # --- SECCIÓN 3: LABORATORIO DE ANÁLISIS VISUAL (Estilo EDA) ---
        st.markdown("---")
        st.subheader("Laboratorio de Análisis Paramétrico")
        
        tab1, tab2, tab3 = st.tabs(["Espectrograma Mel", "Evolución del Centroide", "Morfología de la Onda"])
        
        with tab1:
            st.markdown("**Distribución de Energía (Dominio Tiempo-Frecuencia)**")
            fig_mel, ax_mel = plt.subplots(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y_audio, sr=sr_audio, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            # Uso de colormap corporativo (viridis o crest) en lugar de magma
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_audio, ax=ax_mel, cmap='viridis')
            fig_mel.colorbar(img, ax=ax_mel, format='%+2.0f dB')
            st.pyplot(fig_mel)

        with tab2:
            st.markdown("**Estabilidad Tímbrica: Centroide Espectral a lo largo del tiempo**")
            fig_cent, ax_cent = plt.subplots(figsize=(10, 3))
            cent = librosa.feature.spectral_centroid(y=y_audio, sr=sr_audio)[0]
            times = librosa.times_like(cent, sr=sr_audio)
            
            ax_cent.plot(times, cent, color='#d00000', linewidth=1.5)
            ax_cent.set_ylabel('Frecuencia (Hz)')
            ax_cent.set_xlabel('Tiempo (s)')
            ax_cent.fill_between(times, cent, color='#d00000', alpha=0.2)
            
            for spine in ['top', 'right']:
                ax_cent.spines[spine].set_visible(False)
            st.pyplot(fig_cent)

        with tab3:
            st.markdown("**Envolvente de Amplitud (Dominio Temporal)**")
            fig_wave, ax_wave = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y_audio, sr=sr_audio, ax=ax_wave, color='#343a40', alpha=0.8)
            ax_wave.set_ylabel('Amplitud')
            ax_wave.set_xlabel('Tiempo (s)')
            
            for spine in ['top', 'right']:
                ax_wave.spines[spine].set_visible(False)
            st.pyplot(fig_wave)

    except Exception as e:
        st.error(f"Fallo en la ejecución del proceso analítico: {e}")
    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)

elif uploaded_file is None:
    st.info("A la espera de datos. Por favor, cargue un archivo de audio en el panel lateral para iniciar la evaluación.")