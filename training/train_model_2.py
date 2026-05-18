# train_model_v2.py (Iteración 2: Modelo Optimizado con Poda de Características - Sin Gráficos)
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === 1. Cargar y Podar Características (Feature Pruning) ===
print("=== FASE 2: ENTRENAMIENTO DE MODELO OPTIMIZADO (PRUNED) ===")
print("Cargando dataset...")
df = pd.read_csv("../musical_instruments/features_dataset.csv")

# LA PODA: Las derivadas espectrales de segundo orden (delta2) introducen ruido matemático 
# en entornos polifónicos. Las eliminamos para optimizar la dimensionalidad.
cols_to_drop = [col for col in df.columns if "delta2" in col]
df_pruned = df.drop(columns=cols_to_drop)

# Separar variables predictoras y objetivo
X = df_pruned.drop(columns=["FileName", "Class"]).values
y_text = df_pruned["Class"].values
feature_names = df_pruned.drop(columns=["FileName", "Class"]).columns

print("\n--- Reducción de Dimensionalidad ---")
print(f"Características originales (Base): {len(df.columns) - 2}")
print(f"Características podadas (delta2): {len(cols_to_drop)}")
print(f"Total de características activas para el entrenamiento: {len(feature_names)}\n")

# Limpiar valores infinitos generados por divisiones por cero en el ETL
X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

# Codificar etiquetas de texto a numéricas
le = LabelEncoder()
y = le.fit_transform(y_text)

# === 2. Crear el Pipeline con XGBoost Optimizado ===
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        n_estimators=300,          # Reducido para evitar overfitting en el espacio podado
        max_depth=6,               # Aumentado para capturar relaciones más complejas
        learning_rate=0.1,         # Tasa de aprendizaje acelerada
        subsample=0.8,             
        colsample_bytree=0.8,      # Incrementado para forzar la exploración de la física restante
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1
    ))
])

# === 3. Validación K-Fold y Recolección de Predicciones ===
print("Iniciando validación cruzada K-Fold (K=5) en el espacio reducido...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list = []

y_true_all = []
y_pred_all = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
    print(f"Fold {fold} - Accuracy: {acc:.4f}")

print(f"\n=== Promedio Accuracy K-Fold (Optimizado): {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f} ===")

# === 4. Reporte ===
print("\nReporte Global (Modelo Podado):")
print(classification_report(y_true_all, y_pred_all, target_names=le.classes_, digits=3))

# === 5. Entrenar Modelo Final y Extraer Importancia de Características ===
print("\nEntrenando modelo definitivo con el 100% del dataset reducido...")
pipeline.fit(X, y)

classifier = pipeline.named_steps['classifier']
importances = classifier.feature_importances_

indices_sorted = np.argsort(importances)[::-1]
print("\n=== TOP 20 CARACTERÍSTICAS MÁS IMPORTANTES (MODELO OPTIMIZADO) ===")
for i in range(20):
    idx = indices_sorted[i]
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# === 6. Exportación para Producción ===
# NUEVO: Ahora también guardamos la lista exacta de nombres de columnas (feature_names)
joblib.dump({
    'pipeline': pipeline, 
    'label_encoder': le,
    'expected_features': list(feature_names)  # <- Esta es la clave del enrutador dinámico
}, "Audio_XGBoost_Pruned_Model.pkl")

print("\nPipeline optimizado y codificador guardados en 'Audio_XGBoost_Pruned_Model.pkl'.")
print("PROCESO FINALIZADO CON ÉXITO.")