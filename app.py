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
import time

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

# === 1. Configuración de la página ===
st.set_page_config(page_title="Inferencia Acústica", layout="wide")

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

st.title("Identificación Acústica de Instrumentos Musicales")
st.markdown("""
Plataforma analítica fundamentada en **Extreme Gradient Boosting (XGBoost)** para la clasificación multiclase de señales acústicas. 
El motor de inferencia opera sobre ventanas temporales secuenciales para mitigar la latencia y alinear la topología de la señal con el entorno de entrenamiento.
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

# === 3. Extracción de Características ===
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

def extract_single_feature_vector(y, sr):
    b, a = scipy.signal.butter(N=4, Wn=100 / (sr / 2), btype='high')
    y_filt = scipy.signal.filtfilt(b, a, y)

    if len(y_filt) < 2048:
        y_filt = np.pad(y_filt, (0, 2048 - len(y_filt)), mode='constant')

    temp_feat = compute_temporal_features(y_filt, sr).tolist()
    spec_feat = compute_spectral_features(y_filt, sr).tolist()
    anti_conf_feat = compute_anti_confusion_features(y_filt, sr).tolist()
    energy_ratio_feat = compute_energy_ratios(y_filt, sr).tolist()
    harmonic_purity_feat = compute_harmonic_purity(y_filt, sr).tolist() 
    env_feat = compute_envelope_stats(y_filt).tolist()
    vibrato_feat = compute_vibrato_features(y_filt, sr).tolist()
    bow_reed_feat = compute_bow_vs_reed_features(y_filt, sr).tolist()

    contrast = compute_spectral_contrast(y_filt, sr)
    chroma_stft, chroma_cens, tonnetz = compute_chroma_features(y_filt, sr)
    onset_env, f0, tempo = compute_rhythm_and_pitch(y_filt, sr)
    mfcc, delta, delta2 = compute_mfcc_features(y_filt, sr)

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
    
    return df_final.values

# === 4. Panel de Control y Dashboard Ejecutivo ===

instrument_map = {
    "gac": "Guitarra Acústica", "pia": "Piano", 
    "vio": "Violín", "flu": "Flauta", "sax": "Saxofón"
}

with st.sidebar:
    st.subheader("Parámetros de Entrada")
    uploaded_file = st.file_uploader("Archivo fuente (WAV, MP3)", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
    st.markdown("---")
    st.subheader("Configuración Algorítmica")
    umbral = st.slider("Confianza Mínima Global", 0.0, 1.0, 0.60, 0.05, 
                       help="Umbral mínimo de certeza promedio para considerar la inferencia como válida.")
    
    ejecutar = st.button("Ejecutar Análisis Predictivo", use_container_width=True, type="primary")

if uploaded_file is not None and ejecutar:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name
    
    try:
        y_full = load_and_normalize(tmp_filepath)
        sr_full = 44100
        
        window_sec = 3.0
        window_samples = int(window_sec * sr_full)
        total_chunks = max(1, int(np.ceil(len(y_full) / window_samples)))
        
        # --- UI DE PROCESAMIENTO EN TIEMPO REAL ---
        st.subheader("Estado de Procesamiento Analítico")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown("#### Función de Probabilidad (Actualización en Vivo)")
        chart_placeholder = st.empty() # Lienzo dinámico para el gráfico
        
        probabilidades_acumuladas = []

        # Bucle iterativo de inferencia
        for i in range(total_chunks):
            start = i * window_samples
            end = start + window_samples
            y_chunk = y_full[start:end]
            
            status_text.text(f"Extrayendo descriptores - Ventana {i+1} de {total_chunks} ({(i*window_sec):.1f}s - {((i+1)*window_sec):.1f}s)...")
            
            X_pred = extract_single_feature_vector(y_chunk, sr_full)
            X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
            
            probs_chunk = pipeline.predict_proba(X_pred)[0]
            probabilidades_acumuladas.append(probs_chunk)
            
            # Generar gráfico temporal para la ventana actual
            df_probs_live = pd.DataFrame({
                'Instrumento': [instrument_map.get(c.lower(), c) for c in le.classes_],
                'Probabilidad (%)': probs_chunk * 100
            }).sort_values(by='Probabilidad (%)', ascending=True)

            fig_live, ax_live = plt.subplots(figsize=(10, 3))
            ax_live.barh(df_probs_live['Instrumento'], df_probs_live['Probabilidad (%)'], color='#adb5bd') # Color neutro mientras procesa
            ax_live.set_xlabel(f'Nivel de Confianza - Segmento Actual (%)')
            ax_live.set_xlim(0, 100)
            ax_live.grid(axis='x', linestyle='--', alpha=0.7)
            for spine in ['top', 'right']:
                ax_live.spines[spine].set_visible(False)
            
            # Actualizar lienzo y liberar memoria
            chart_placeholder.pyplot(fig_live)
            plt.close(fig_live) 
            
            progress_bar.progress((i + 1) / total_chunks)

        # --- CONSOLIDACIÓN DE RESULTADOS ---
        status_text.text("Consolidando resultados y promediando probabilidades...")
        probabilidades_globales = np.mean(probabilidades_acumuladas, axis=0)
        idx_ganador = np.argmax(probabilidades_globales)
        instrumento_raw = le.inverse_transform([idx_ganador])[0]
        certeza_global = probabilidades_globales[idx_ganador]
        
        nombre_instrumento = instrument_map.get(instrumento_raw.lower(), instrumento_raw)
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # Actualizar el lienzo dinámico con el gráfico FINAL (Ponderado)
        df_probs_final = pd.DataFrame({
            'Instrumento': [instrument_map.get(c.lower(), c) for c in le.classes_],
            'Probabilidad (%)': probabilidades_globales * 100
        }).sort_values(by='Probabilidad (%)', ascending=True)

        fig_final, ax_final = plt.subplots(figsize=(10, 3))
        ax_final.barh(df_probs_final['Instrumento'], df_probs_final['Probabilidad (%)'], color='#2c7da0') # Color corporativo final
        ax_final.set_xlabel('Nivel de Confianza Global Promediado (%)')
        ax_final.set_xlim(0, 100)
        ax_final.grid(axis='x', linestyle='--', alpha=0.7)
        for spine in ['top', 'right']:
            ax_final.spines[spine].set_visible(False)
            
        chart_placeholder.pyplot(fig_final)
        plt.close(fig_final)

        # --- SECCIÓN 1: REPORTE DE INFERENCIA ---
        st.markdown("---")
        st.subheader("Reporte de Inferencia Global")
        
        if certeza_global < umbral:
            st.warning(f"Diagnóstico Inconcluso: La certeza promedio ({certeza_global*100:.2f}%) no supera el umbral establecido del {umbral*100:.0f}%. El audio presenta características desconocidas o altamente mixtas.")
        else:
            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric(label="Clase Predominante Identificada", value=nombre_instrumento)
            col_met2.metric(label="Nivel de Certeza Ponderado", value=f"{certeza_global*100:.2f} %")
            col_met3.metric(label="Segmentos Evaluados", value=f"{total_chunks} ventanas (3s)")
            
            st.success("Análisis secuencial completado satisfactoriamente.")

        # --- SECCIÓN 2: LABORATORIO DE ANÁLISIS VISUAL ---
        st.markdown("---")
        st.subheader("Análisis Paramétrico")
        
        tab1, tab2, tab3 = st.tabs(["Espectrograma Mel", "Evolución del Centroide", "Morfología de la Onda"])
        
        with tab1:
            st.markdown("**Distribución de Energía Global (Dominio Tiempo-Frecuencia)**")
            fig_mel, ax_mel = plt.subplots(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y_full, sr=sr_full, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_full, ax=ax_mel, cmap='viridis')
            fig_mel.colorbar(img, ax=ax_mel, format='%+2.0f dB')
            st.pyplot(fig_mel)

        with tab2:
            st.markdown("**Estabilidad Tímbrica: Centroide Espectral a lo largo del tiempo**")
            fig_cent, ax_cent = plt.subplots(figsize=(10, 3))
            cent = librosa.feature.spectral_centroid(y=y_full, sr=sr_full)[0]
            times = librosa.times_like(cent, sr=sr_full)
            
            ax_cent.plot(times, cent, color='#d00000', linewidth=1.5)
            ax_cent.set_ylabel('Frecuencia (Hz)')
            ax_cent.set_xlabel('Tiempo (s)')
            ax_cent.fill_between(times, cent, color='#d00000', alpha=0.2)
            
            for spine in ['top', 'right']:
                ax_cent.spines[spine].set_visible(False)
            st.pyplot(fig_cent)

        with tab3:
            st.markdown("**Envolvente de Amplitud (Dominio Temporal Submuestreado)**")
            fig_wave, ax_wave = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y_full, sr=sr_full, ax=ax_wave, color='#343a40', alpha=0.8, max_points=10000)
            ax_wave.set_ylabel('Amplitud')
            ax_wave.set_xlabel('Tiempo (s)')
            
            for spine in ['top', 'right']:
                ax_wave.spines[spine].set_visible(False)
            st.pyplot(fig_wave)

    except Exception as e:
        st.error(f"Fallo crítico en la ejecución del proceso analítico: {e}")
    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)

elif uploaded_file is None:
    st.info("A la espera de datos. Por favor, cargue un archivo de audio en el panel lateral para iniciar la evaluación.")