# train_model.py
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
print("Cargando dataset...")
df = pd.read_csv("data_v3/features_dataset.csv")

# 🚨 EL FILTRO DE GRASA (Feature Pruning) 🚨
# Eliminamos todas las derivadas de 2do orden (delta2) 
# para forzar al modelo a usar las variables físicas e ignorar el ruido matemático.
columnas_a_borrar = [col for col in df.columns if "delta2" in col]
df = df.drop(columns=columnas_a_borrar)

print(f"Dataset reducido: Se eliminaron {len(columnas_a_borrar)} columnas de ruido.")
print(f"Total de columnas activas para el entrenamiento: {len(df.columns)}")

X = df.drop(columns=["FileName", "Class"]).values
y_text = df["Class"].values

# Limpiar valores infinitos
X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

le = LabelEncoder()
y = le.fit_transform(y_text)
feature_names = df.drop(columns=["FileName", "Class"]).columns

# === 2. Crear el Pipeline con XGBoost Ajustado ===
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        n_estimators=500,          # Subimos ligeramente la cantidad para compensar árboles más simples
        max_depth=5,               # Reducido de 7 a 4: Evita que se vicie con características dominantes
        learning_rate=0.05,        
        subsample=0.8,             
        colsample_bytree=0.4,      # Reducido a 40%: Fuerza al modelo a mirar las columnas nuevas
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1
    ))
])

# === 3. Validación K-Fold y Recolección de Predicciones ===
print("Iniciando validación cruzada K-Fold con XGBoost...")
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
plt.title("Matriz de Confusión Global (XGBoost)")
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=300)
print("Matriz guardada como 'matriz_confusion.png'.")

# === 5. Entrenar Modelo Final ===
print("\nEntrenando modelo final con el 100% del dataset...")
pipeline.fit(X, y)

classifier = pipeline.named_steps['classifier']
importances = classifier.feature_importances_

indices_sorted = np.argsort(importances)[::-1]
print("\n=== TOP 20 CARACTERÍSTICAS MÁS IMPORTANTES ===")
for i in range(20):
    idx = indices_sorted[i]
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

joblib.dump({'pipeline': pipeline, 'label_encoder': le}, "src/models/Audio_XGBoost_Model.pkl")
print("\nPipeline y codificador guardados en 'src/models/Audio_XGBoost_Model.pkl'.")