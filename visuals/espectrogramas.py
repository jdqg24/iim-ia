import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14})

def generar_espectrograma(input_path, etiqueta_clase):
    print(f"Analizando audio: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: El archivo {input_path} no existe.")
        return

    try:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 4))
        
        y, sr = librosa.load(input_path, sr=22050, duration=3.0)
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr, ax=ax, cmap='magma')
        
        ax.set_title(f'Espectrograma de Análisis: {etiqueta_clase.upper()}', fontweight='bold')
        ax.set_ylabel('Frecuencia (Hz - Log)')
        ax.set_xlabel('Tiempo (s)')
        
        fig.colorbar(img, ax=ax, format="%+2.0f dB", label='Potencia (dB)')
        
        plt.tight_layout()
        
        output_filename = f'espectrograma_{etiqueta_clase}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Imagen guardada: {output_filename}")
        
        plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"Error procesando {input_path}: {e}")


if __name__ == '__main__':
    audio_path = '../a.mp3'
    etiqueta = 'sax_error_violin'
    
    generar_espectrograma(audio_path, etiqueta)