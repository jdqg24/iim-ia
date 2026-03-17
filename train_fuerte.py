# train_fuerte.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# === 1. Cargar features ===
print("Cargando dataset...")
df = pd.read_csv("data/features_dataset.csv")

X = df.drop(columns=["FileName", "Class"]).values
y_text = df["Class"].values

X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

le = LabelEncoder()
y = le.fit_transform(y_text)
feature_names = df.drop(columns=["FileName", "Class"]).columns

# === 2. Crear el Pipeline Base ===
# Dejamos XGBoost con n_jobs=1 aquí para que RandomizedSearchCV maneje el paralelismo
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        n_jobs=1 
    ))
])

# === 3. Definir la Cuadrícula de Búsqueda (Grid) ===
param_distributions = {
    'classifier__n_estimators': [300, 500, 800, 1000],
    'classifier__max_depth': [5, 7, 9, 11],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'classifier__gamma': [0, 0.1, 0.5, 1, 2]
}

# Configurar la búsqueda aleatoria
print("Iniciando búsqueda de hiperparámetros (RandomizedSearchCV)...")
print("⚠️ Esto puede tardar un rato. ¡Ve por un café! ☕\n")

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,          # Probará 30 combinaciones aleatorias
    cv=3,               # 3 Folds para evaluar cada combinación rápidamente
    scoring='accuracy',
    n_jobs=-1,          # Usar todos los núcleos del procesador
    verbose=2,          # Mostrar el progreso en consola
    random_state=42
)

# === 4. Ejecutar la Búsqueda y Obtener el Mejor Modelo ===
random_search.fit(X, y)

print("\n" + "="*50)
print("🏆 ¡Búsqueda completada!")
print(f"Mejor Accuracy en CV: {random_search.best_score_:.4f}")
print("Mejores hiperparámetros encontrados:")
for param, value in random_search.best_params_.items():
    print(f" - {param.replace('classifier__', '')}: {value}")
print("="*50 + "\n")

# Extraer el mejor pipeline entrenado
best_pipeline = random_search.best_estimator_

# === 5. Evaluación Final Robusta con el Mejor Modelo ===
print("Generando reporte global con validación cruzada (5-Fold)...")
kf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Obtener predicciones limpias para todo el dataset usando el mejor modelo
y_pred_all = cross_val_predict(best_pipeline, X, y, cv=kf_final, n_jobs=-1)

acc_final = accuracy_score(y, y_pred_all)
print(f"\n=== Accuracy Global Final: {acc_final:.4f} ===")

print("\nReporte Global:")
print(classification_report(y, y_pred_all, target_names=le.classes_, digits=3))

print("\nGenerando matriz de confusión...")
cm = confusion_matrix(y, y_pred_all)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matriz de Confusión Final (XGBoost Optimizado)")
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=300)
print("Matriz guardada como 'matriz_confusion.png'.")

# === 6. Guardar el Modelo Campeón ===
# Entrenar una última vez con absolutamente todos los datos
best_pipeline.fit(X, y)

joblib.dump({'pipeline': best_pipeline, 'label_encoder': le}, "src/models/Audio_XGBoost_Optimized.pkl")
print("\nModelo optimizado y codificador guardados en 'src/models/Audio_XGBoost_Optimized.pkl'.")