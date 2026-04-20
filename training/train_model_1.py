# train_model_v1.py (Iteración 1: Modelo Base sin Poda)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# === 1. Cargar features ===
print("=== FASE 1: ENTRENAMIENTO DE MODELO BASE ===")
print("Cargando dataset...")
df = pd.read_csv("IRMAS_Data/features_dataset.csv")

# Descartamos metadatos para obtener el conteo exacto de características acústicas
X = df.drop(columns=["FileName", "Class"]).values
y_text = df["Class"].values
feature_names = df.drop(columns=["FileName", "Class"]).columns

print(f"Dataset cargado intacto. No se eliminaron columnas.")
print(f"Total de características activas para el entrenamiento: {len(feature_names)}")

# Limpiar valores infinitos generados por divisiones por cero en el ETL
X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

# Codificar etiquetas de texto a numéricas
le = LabelEncoder()
y = le.fit_transform(y_text)

# === 2. Crear el Pipeline con XGBoost ===
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        n_estimators=500,          
        max_depth=5,               
        learning_rate=0.05,        
        subsample=0.8,             
        colsample_bytree=0.4,      
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1
    ))
])

# === 3. Validación K-Fold y Recolección de Predicciones ===
print("\nIniciando validación cruzada K-Fold (K=5) con XGBoost...")
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

print(f"\n=== Promedio Accuracy K-Fold: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f} ===")

# === 4. Reporte y Matriz de Confusión ===
print("\nReporte Global:")
print(classification_report(y_true_all, y_pred_all, target_names=le.classes_, digits=3))

print("\nGenerando matriz de confusión...")
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matriz de Confusión Base (Full Features)")
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
plt.savefig("matriz_confusion_base.png", dpi=300)
print("Matriz guardada como 'matriz_confusion_base.png'.")

# === 5. Entrenar Modelo Final y Extraer Importancia de Características ===
print("\nEntrenando modelo final con el 100% del dataset...")
pipeline.fit(X, y)

classifier = pipeline.named_steps['classifier']
importances = classifier.feature_importances_

indices_sorted = np.argsort(importances)[::-1]
print("\n=== TOP 20 CARACTERÍSTICAS MÁS IMPORTANTES (MODELO BASE) ===")
for i in range(20):
    idx = indices_sorted[i]
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# Guardar este modelo inicial con un nombre diferente para no sobrescribir el optimizado
joblib.dump({'pipeline': pipeline, 'label_encoder': le}, "Audio_XGBoost_Base_Model.pkl")
print("\nPipeline base y codificador guardados en 'Audio_XGBoost_Base_Model.pkl'.")