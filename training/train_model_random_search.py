import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
import scipy.stats as stats
import joblib

print("=== FASE DE OPTIMIZACIÓN: RANDOM SEARCH CV ===")
print("Cargando dataset híbrido...")
df = pd.read_csv("./features_dataset.csv")

X = df.drop(columns=["FileName", "Class"]).values
y_text = df["Class"].values
feature_names = df.drop(columns=["FileName", "Class"]).columns

print(f"Dataset cargado. Total de características: {len(feature_names)}")

X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

le = LabelEncoder()
y = le.fit_transform(y_text)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1
    ))
])

param_dist = {
    'classifier__n_estimators': stats.randint(100, 800),
    'classifier__max_depth': stats.randint(3, 11),
    'classifier__learning_rate': stats.uniform(0.01, 0.29),
    'classifier__min_child_weight': stats.randint(1, 11),
    'classifier__gamma': stats.uniform(0, 1.0),
    'classifier__reg_lambda': stats.uniform(0, 5.0),
    'classifier__reg_alpha': stats.uniform(0, 1.0),
    'classifier__colsample_bytree': stats.uniform(0.3, 0.5), 
    'classifier__subsample': stats.uniform(0.6, 0.4) 
}

print("\nIniciando Random Search (60 iteraciones) con Stratified K-Fold (K=5)...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average='macro')

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=60,
    scoring=scorer,
    cv=kf,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)

print("\n=== OPTIMIZACIÓN FINALIZADA ===")
print(f"Mejor F1-Score (Macro) alcanzado: {random_search.best_score_:.4f}")
print("Hiperparámetros de la mejor configuración:")
for param, value in random_search.best_params_.items():
    print(f" - {param.replace('classifier__', '')}: {value:.4f}" if isinstance(value, float) else f" - {param.replace('classifier__', '')}: {value}")

best_model = random_search.best_estimator_
y_pred = best_model.predict(X)

print("\nReporte Global del Modelo Optimizado:")
print(classification_report(y, y_pred, target_names=le.classes_, digits=3))

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matriz de Confusión (Modelo Optimizado)")
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
plt.savefig("matriz_confusion_optimizado.png", dpi=300)

joblib.dump({
    'pipeline': best_model, 
    'label_encoder': le,
    'expected_features': list(feature_names)
}, "Audio_XGBoost_Optimized_Model.pkl")
print("\nModelo optimizado exportado exitosamente.")
