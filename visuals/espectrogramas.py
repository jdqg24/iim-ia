import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def set_plot_style():
    """Configuración estética básica para los gráficos de espectrogramas."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14})

def generar_espectrogramas_individuales(diccionario_audios):
    """
    Toma un audio de ejemplo de cada clase, genera su espectrograma,
    lo guarda individualmente y lo muestra en una ventana interactiva.
    """
    print("Iniciando generación de espectrogramas individuales...")
    set_plot_style()
    
    for input_path, etiqueta_clase in diccionario_audios.items():
        print("\n" + "-"*50)
        print(f"Analizando audio para clase: [{etiqueta_clase.upper()}] - {input_path}")
        
        if not os.path.exists(input_path):
            print(f"⚠️ El archivo {input_path} no existe. Saltando instrumento...")
            continue

        try:
            # 1. Crear una nueva figura individual
            # Un tamaño de 10x4 es ideal para insertarlo a lo ancho en una página Word
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # 2. Cargar el audio (3 segundos)
            y, sr = librosa.load(input_path, sr=22050, duration=3.0) 
            
            # 3. Calcular el Espectrograma de Tiempo Corto (STFT)
            D = librosa.stft(y, n_fft=2048, hop_length=512)
            
            # 4. Convertir la amplitud a decibelios
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # 5. Graficar el espectrograma
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr, ax=ax, cmap='magma')
            
            # 6. Configuraciones estéticas del gráfico
            ax.set_title(f'Espectrograma de Análisis: {etiqueta_clase.upper()} (polifónico)', fontweight='bold')
            ax.set_ylabel('Frecuencia (Hz - Log)')
            ax.set_xlabel('Tiempo (s)')
            
            # 7. Añadir barra de color individual (ahora sin problemas de espacio)
            fig.colorbar(img, ax=ax, format="%+2.0f dB", label='Potencia (dB)')
            
            plt.tight_layout()
            
            # 8. Guardar la imagen en disco
            output_filename = f'espectrograma_{etiqueta_clase}.png'
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"✅ Imagen guardada exitosamente: {output_filename}")
            
            # 9. Mostrar la ventana y pausar
            print("👁️  Visualizando ventana... Ciérrala para continuar con el siguiente.")
            plt.show()
            
            # 10. Limpiar la figura de la memoria para que no se superpongan
            plt.close(fig)

        except Exception as e:
            print(f"❌ Error procesando {input_path}: {e}")
            
    print("\n🎉 Todos los espectrogramas han sido procesados y guardados.")

# ==========================================
# CONFIGURACIÓN Y EJECUCIÓN 
# ==========================================
audios_ejemplo = {
    '../IRMAS_Data/pia/[pia][jaz_blu]1531__1.wav': 'pia',     
    '../IRMAS_Data/vio/[vio][jaz_blu]2109__2.wav': 'vio',    
    '../IRMAS_Data/flu/[flu][jaz_blu]0458__2.wav': 'flu',    
    '../IRMAS_Data/gac/[gac][pop_roc]0706__2.wav': 'gac',  
    '../IRMAS_Data/sax/[sax][jaz_blu]1758__2.wav': 'sax'    
}

# Ejecutar el script
generar_espectrogramas_individuales(audios_ejemplo)