# Manual de usuario

## 1. Introducción

Este proyecto permite analizar archivos de audio y clasificar instrumentos musicales mediante características acústicas y un modelo de clasificación XGBoost. También incluye herramientas para extraer características, entrenar modelos, explorar datos y generar visualizaciones.

El flujo recomendado para un usuario es:

1. Instalar las dependencias.
2. Preparar un conjunto de audio identificado por clase.
3. Extraer características del conjunto.
4. Entrar el modelo.
5. Ejecutar la aplicación Streamlit y subir un archivo de audio.

## 2. Requisitos

- Python 3.10 o superior.
- Windows PowerShell.
- Acceso al directorio del proyecto.
- Los archivos de audio debe estar en formato WAV, MP3, FLAC u OGG.
- El entorno virtual del proyecto debe estar activado antes de ejecutar los scripts.

El proyecto usa las siguientes dependencias:

- Streamlit para la interfaz web.
- NumPy, Pandas y SciPy para procesamiento numérico.
- Librosa y SoundFile para audio.
- scikit-learn para preprocessing, pipelines y métricas.
- XGBoost para clasificación.
- Joblib para serialización y paralelismo.
- Matplotlib y Seaborn para gráficos.
- tqdm para barras de progreso.

Las versiones exactas se encuentran en [`requirements.txt`](requirements.txt).

## 3. Instalación

### 3.1. Abrir PowerShell en la raíz del proyecto

Ejecute:

```powershell
Set-Location "C:\Users\juanq\OneDrive\Desktop\tg\iim-ia"
```

### 3.2. Activar el entorno virtual

Si el entorno virtual ya existe en la carpeta `venv`:

```powershell
.\venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución de scripts, habilítalo temporalmente:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3.3. Instalar las dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3.4. Verificar la instalación

```powershell
python -c "import streamlit, numpy, pandas, scipy, librosa, sklearn, xgboost, joblib; print('Dependencias importadas correctamente')"
```

## 4. Preparación de datos

### 4.1. Estructura esperada del dataset

La extractor busca carpetas de clases dentro del directorio de datos. Por ejemplo:

```text
data_v3/
├── gac/
│   ├── guitarra_1.wav
│   └── guitarra_2.wav
├── pia/
│   ├── piano_1.wav
│   └── piano_2.wav
├── vio/
│   ├── violin_1.wav
│   └── violin_2.wav
└── sax/
    ├── sax_1.wav
    └── sax_2.wav
```

El nombre de cada carpeta se utiliza como etiqueta de clase. Los archivos de audio se pueden guardar en formatos compatibles con Librosa.

### 4.2. Ejecutar la extracción

Desde la raíz del proyecto:

```powershell
python src/main.py
```

El script usa:

- Origen: `data_v3/`.
- Resultado: `data_v3/features_dataset.csv`.
- Procesamiento: paralelo mediante Joblib.

La extracción puede tardar mucho tiempo dependiendo del número de archivos y de la potencia del equipo.

> La salida debe contener una columna `FileName`, una columna `Class` y las columnas de características generadas.

## 5. Entrenamiento de modelos

### 5.1. Modelo base

Ejecuta:

```powershell
python training/train_model_1.py
```

Este script:

1. Carga el dataset de características.
2. Codifica las etiquetas.
3. Aplica imputación y escalado.
4. Ejecuta validación cruzada con XGBoost.
5. Genera métricas, una matriz de confusión y un modelo base.

Guarda el modelo en el directorio actual del proceso, normalmente la raíz del proyecto, con el nombre:

```text
Audio_XGBoost_Base_Model.pkl
```

Para que la aplicación lo utilice, copía el archivo al directorio `src/models` con el mismo nombre.

### 5.2. Modelo optimizado

Ejecuta:

```powershell
python training/train_model_2.py
```

Este script elimina las características `delta2` y entrena un modelo optimizado. Guarda el resultado en el directorio actual del proceso con el nombre:

```text
Audio_XGBoost_Pruned_Model.pkl
```

Para que la aplicación lo utilice, copía el archivo al directorio `src/models` con el mismo nombre.

El archivo incluye:

- `pipeline`.
- `label_encoder`.
- `expected_features`.

La función de la lista `expected_features` es mantener el orden exacto de las columnas que debe recibir el modelo durante la inferencia.

## 6. Ejecución de la aplicación

### 6.1. Iniciar la aplicación

Desde la raíz del proyecto:

```powershell
python app.py
```

Streamlit abrirá automáticamente una dirección local similar a:

```text
http://localhost:8501
```

### 6.2. Uso de la interfaz

1. Abre la dirección mostrada por Streamlit.
2. Selecciona un archivo WAV o MP3 mediante el cargador de archivos.
3. Ajusta el umbral de confianza si se desea.
4. Pulsa **Ejecutar Análisis Predictivo**.
5. Revisa:
   - Clase predominante.
   - Porcentaje de confianza.
   - Probabilidades por instrumento.
   - Comparativa de energía y espectrograma.

La aplicación procesa el audio en segmentos de cinco segundos y promedia las probabilidades de los segmentos.

### 6.3. Modelo requerido por la aplicación

La aplicación carga:

```text
src/models/Audio_XGBoost_Pruned_Model.pkl
```

Este archivo ya está incluido en `src/models`. Si se reentrena un modelo nuevo, reemplázalo en esa ubicación y mantenga la estructura esperada: `pipeline`, `label_encoder` y `expected_features`.

Si el archivo no existe o no tiene esa estructura, la aplicación mostrará un error y detendrá la ejecución.

## 7. Predicción mediante línea de comandos

El script [`predict_audio.py`](predict_audio.py) acepta un archivo de audio como argumento:

```powershell
python predict_audio.py "ruta/al/audio.wav"
```

Ejemplo:

```powershell
python predict_audio.py "C:\Audios\piano.wav"
```

La salida será similar a:

```text
Instrumento predicho: Piano
```

### Importante

Este script carga:

```text
RandomForest_audio_model.pkl
```

Ese archivo no está incluido en la estructura visible del proyecto. Por tanto, la CLI solo funciona cuando ese modelo existe en la raíz del proyecto y tiene el formato esperado:

- `model`.
- `scaler`.
- `label_encoder`.

Para usar el modelo XGBoost incluido, es preferible ejecutar la aplicación Streamlit en lugar de la CLI.

## 8. Visualización de espectrogramas

Para generar un espectrograma individual:

```powershell
python visuals/espectrogramas.py
```

El scriptactual usa una ruta fija y una etiqueta de clase. Para adaptar el comportamiento, modifica los valores de `audio_path` y `etiqueta` en [`visuals/espectrogramas.py`](visuals/espectrogramas.py).

## 9. Estrutura recomendada de trabajo

```text
C:\Users\juanq\OneDrive\Desktop\tg\iim-ia
├── venv\
├── data_v3\
├── src\models\
├── training\
├── visuals\
└── requirements.txt
```

## 10. Comandos habituales

### Activar el entorno

```powershell
.\venv\Scripts\Activate.ps1
```

### Verificar versiones de Python

```powershell
python --version
```

### Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

### Ejecutar la app

```powershell
python app.py
```

### Ejecutar la extracción

```powershell
python src/main.py
```

### Ejecutar el entrenamiento

```powershell
python training/train_model_2.py
```

## 11. Solución de problemas

### El entorno virtual no se activa

Ejecute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### No se encuentra un modelo

Comprueba que el archivo exista en la carpeta correcta. La aplicación requiere:

```text
src/models/Audio_XGBoost_Pruned_Model.pkl
```

La CLI requiere un archivo con el nombre y formato indicados anteriormente.

### Error de importación

Verifica que el entorno activo sea el correcto:

```powershell
where.exe python
python -c "import sys; print(sys.executable)"
```

Después reinstala las dependencias:

```powershell
python -m pip install -r requirements.txt
```

### El archivo de audio no se encuentra

Confirma que la ruta sea correcta y que el archivo exista:

```powershell
Test-Path "C:\Audios\piano.wav"
```

### El archivo de datos no contiene caracteres

Comprueba las columnas del CSV:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data_v3/features_dataset.csv'); print(df.columns.tolist()); print(df.shape)"
```

## 12. Advertencias

- El modelo debe corresponder al mismo esquema de características que genera la extractor.
- Cambios en la estructura de las etiquetas o en el orden de las columnas pueden provocar errores en la inferencia.
- Los archivos de audio, datos y modelos pueden ser grandes y no deben subirse al repositorio si no se desea versionarlos.
- Los resultados de clasificación dependen de la calidad del audio, del tipo de instrumento y del conjunto de entrenamiento.
- La aplicación Streamlit es la opción más completa para el flujo actual porque carga el modelo XGBoost optimizado incluido en el proyecto.
