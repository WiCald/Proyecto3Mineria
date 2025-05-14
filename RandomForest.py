import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Carga y preparación de datos usando la hoja 'Divorcios Mes y Año'
def load_and_prepare_data():
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / "resumen_estadistico_divorcios_2013_2023.xlsx"
    df = pd.read_excel(file_path, sheet_name="Divorcios Mes y Año")

    # Convertir columnas útiles a numérico
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Divorcios'] = pd.to_numeric(df['Divorcios'], errors='coerce')

    # Mapear nombres de mes en español a valor numérico
    month_map = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4,
        'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8,
        'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    df['Mes'] = df['Mes'].map(month_map)

    # Eliminar filas incompletas
    df = df.dropna(subset=['Year', 'Mes', 'Divorcios'])
    df['Mes'] = df['Mes'].astype(int)

    # Definir variables predictoras y objetivo
    X = df[['Year', 'Mes']]
    y = df['Divorcios']

    # Split: 70% entrenamiento / 30% prueba
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento y evaluación de modelos Random Forest
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    # Modelo base
    model_base = RandomForestRegressor(random_state=42)
    model_base.fit(X_train, y_train)
    pred_base = model_base.predict(X_test)
    results['Base'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, pred_base)),
        'r2': r2_score(y_test, pred_base)
    }

    # Modelo tuneado manualmente
    model_tuned = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    model_tuned.fit(X_train, y_train)
    pred_tuned = model_tuned.predict(X_test)
    results['Tuned'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, pred_tuned)),
        'r2': r2_score(y_test, pred_tuned)
    }

    # Búsqueda de hiperparámetros con GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    pred_best = best_model.predict(X_test)
    results['GridSearch'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, pred_best)),
        'r2': r2_score(y_test, pred_best)
    }

    return results, grid_search.best_params_, best_model

# Visualización comparativa de resultados
def visualize_results(results):
    labels = list(results.keys())
    rmses = [results[name]['rmse'] for name in labels]
    r2s = [results[name]['r2'] for name in labels]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(labels, rmses)
    ax1.set_ylabel('RMSE')

    ax2 = ax1.twinx()
    ax2.plot(labels, r2s, marker='o')
    ax2.set_ylabel('R²')

    for i, v in enumerate(rmses):
        ax1.text(i, v, f"{v:.2f}", ha='center', va='bottom')
    for i, v in enumerate(r2s):
        ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')

    plt.title('Comparación de Modelos Random Forest')
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'comparacion_rf_mes_anio.png', dpi=300)
    plt.show()

# Función principal
def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    results, best_params, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("=== RESULTADOS ===")
    for name, metrics in results.items():
        print(f"Modelo {name}: RMSE = {metrics['rmse']:.2f}, R² = {metrics['r2']:.2f}")
    print("Mejores parámetros:", best_params)

    # Guardar el modelo final
    joblib.dump(best_model, Path(__file__).resolve().parent / 'modelo_rf_mes_anio.pkl')
    print("Modelo final guardado: modelo_rf_mes_anio.pkl")

    # Mostrar gráfica comparativa
    visualize_results(results)

if __name__ == '__main__':
    main()
