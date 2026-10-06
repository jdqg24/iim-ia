# Estructura del proyecto

## 1. Descripción general

Este proyecto implementa una aplicación de clasificación de instrumentos musicales a partir de audio. El flujo principal es:

1. Cargar y normalizar el audio.
2. Extraer características temporales, espectrales, cromáticas, rítmicas y MFCC.
3. Preparar un conjunto de datos con las características y etiquetas.
4. Entrar y evaluar un modelo XGBoost.
5. Exponer la predicción mediante una aplicación Streamlit.
6. Generar visualizaciones de espectrogramas, energía y análisis acústico.

## 2. Estructura del repositorio

```text
.
├── app.py                         # Aplicación Streamlit de inferencia
├── app (og).py                   # Versión original de la aplicación
├── analisis_cuantitativo.py      # Análisis cuantitativo de los datos
├── EDA.py                         # Exploración de datos y estadísticas
├── optimized_train.py             # Entrenamiento optimizado o alternativo
├── predict_audio.py               # Predicción de un archivo de audio
├── requirements.txt               # Dependencias fijadas con versiones
├── STRUCTURE.md                   # Documentación de la estructura
├── src/
│   ├── main.py                    # Punto de entrada de extracción de características
│   ├── features/
│   │   └── extract_features.py    # Extracción paralela de características del dataset
│   ├── models/                    # Modelos serializados de producción
│   ├── preprocessing/
│   │   └── preprocess_audio.py    # Carga, mono, normalización y relleno de audio
│   └── utils/
│       ├── audio_segmentator.py   # Segmentación del audio
│       ├── audio_utils.py          # Funciones de características acústicas
│       ├── audio_utils_v2.py       # Versión ampliada de características acústicas
│       └── class_balancer.py       # Balanceo de clases
├── training/
│   ├── train_model_1.py           # Modelo base con XGBoost
│   └── train_model_2.py           # Modelo optimizado con poda de características
└── visuals/
    ├── config_visual.py           # Configuración común de visualizaciones
    ├── espectrogramas.py           # Generación de espectrogramas
    ├── main_analisis.py            # Análisis principal de visualización
    └── visualizaciones.py          # Visualizaciones variadas
```

## 3. Módulos principales

### Aplicación

- `app.py`: interfaz Streamlit que carga un archivo de audio, divide el sonido en segmentos de cinco segundos, extrae características y ejecuta inferencia con el modelo XGBoost. Muestra probabilidades, confianza y visualizaciones.
- `app (og).py`: versión original de la aplicación, mantenida para referencia.
- `predict_audio.py`: ejecutable de CLI que predice el instrumento de un archivo concreto. Actualmente espera un modelo almacenado en una ruta compatible con el formato esperado.

### Procesamiento de audio

- `src/preprocessing/preprocess_audio.py`: carga WAV/MP3 mediante Librosa, convierte a mono, normaliza la amplitud y rellena archivos cortos con ceros.
- `src/features/extract_features.py`: escanea carpetas de clases, procesa audios en paralelo y genera un CSV con características y etiquetas.
- `src/utils/audio_utils.py`: implementación de características temporales, espectrales, MFCC, contraste, cromática, ritmo, energía y otros descriptors.
- `src/utils/audio_utils_v2.py`: implementación ampliada utilizada por la aplicación para construir el vector de características de inference.
- `src/utils/audio_segmentator.py`: utilidades para dividir o segmentar señales de audio.
- `src/utils/class_balancer.py`: utilidades para equilibrar clases en el conjunto de entrenamiento.

### Entrenamiento

- `training/train_model_1.py`: entrenamiento base con XGBoost, imputación, escalado, validación cruzada y exportación del modelo base.
- `training/train_model_2.py`: entrenamiento optimizado, eliminación de características delta2 y exportación del pipeline con la lista exacta de características esperadas.
- `optimized_train.py`: script de entrenamiento alternativo u optimizado en el flujo principal.

### Visualización

- `visuals/espectrogramas.py`: genera un espectrograma de un archivo audio.
- `visuals/main_analisis.py`: análisis gráfico principal y exploración de los datos.
- `visuals/config_visual.py`: configuraciones compartidas de estilos y gráficos.
- `visuals/visualizaciones.py`: visualizaciones de datos, métricas y resultados.

### Análisis de datos

- `EDA.py`: exploración de datos de audio y características.
- `analisis_cuantitativo.py`: cálculo de métricas cuantitativas y estadísticas.

## 4. Ruta de datos y modelos

### Datos de entrenamiento

La extracción espera un directorio con una carpeta por clase, donde cada archivo de audio puede ser WAV, MP3, FLAC u OGG. El resultado se espera en un archivo CSV con una columna `FileName`, una `Class` y las características restantes.

La ruta de entrada del extractor se define mediante los parámetros `dataset_dir` y `output_csv` en `src/features/extract_features.py` y `src/main.py`.

### Modelos serializados

La carpeta `src/models` contiene artefactos de modelo ya entrenados:

- `Audio_XGBoost_Model.pkl`: modelo base.
- `Audio_XGBoost_Pruned_Model.pkl`: modelo optimizado con características podadas.
- `monofonico + poda.pkl`: modelo monofónico con poda.
- `polifonico (irmas).pkl`: modelo polifónico con características de clases o instrumentos.

La aplicación principal carga `src/models/Audio_XGBoost_Pruned_Model.pkl` y espera que el archivo incluya `pipeline`, `label_encoder` y `expected_features`.

## 5. Flujo de ejecución recomendado

### Extracción de características

```powershell
python src/main.py
```

El script usa `data_v3/` como origen y `data_v3/features_dataset.csv` como salida.

### Entrenamiento

```powershell
python training/train_model_1.py
python training/train_model_2.py
```

Los scripts deben ejecutarse desde la raíz del proyecto o con las rutas relativas correctas.

### Inferencia

```powershell
python app.py
```

Después de iniciar Streamlit, se puede abrir la aplicación desde el enlace local mostrado por el servidor.

### Predicción de un archivo

```powershell
python predict_audio.py ruta/al/audio.wav
```

## 6. Dependencias

El archivo `requirements.txt` contiene las versiones exactas instaladas en el entorno virtual:

- Streamlit para la interfaz web.
- NumPy, Pandas y SciPy para procesamiento numérico.
- Librosa y SoundFile para audio.
- scikit-learn para preprocessing, evaluación y pipelines.
- XGBoost para clasificación.
- Joblib para serialización y paralelismo.
- Matplotlib y Seaborn para gráficos.
- tqdm para barras de progreso.

## 7. Notas de mantenimiento

- Los scripts utilizan rutas relativas, por lo que se recomienda ejecutar desde la raíz del proyecto.
- Los archivos de datos, CSV y modelos pueden ser grandes; están excluidos del control de versiones en `.gitignore`.
- `src/features/extract_features.py` y `src/main.py` deben mantenerse alineados con el nombre de las características que se generan y con el esquema esperado por el modelo.
- Antes de cambiar la extracción de características, se debe revisar la cantidad y nombres de columnas, porque el modelo depende del orden y la lista de características esperadas.
