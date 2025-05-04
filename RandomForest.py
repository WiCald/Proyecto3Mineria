import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

def load_and_prepare_data():
    df = pd.read_excel('resumen_estadistico_divorcios_2013_2023.xlsx', 
                      sheet_name='Datos por Departamento')    
    X = df.drop(columns=['Total', 'Departamento'])
    y = df['Total']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    
    #Modelo 1 base
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    
    #Modelo 2 tuneo
    rf_tuned = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_tuned.fit(X_train, y_train)
    
    #Modelo 3 hiperparametros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        error_score='raise'
    )
    rf_grid.fit(X_train, y_train)
    
    #modelo final
    best_params = rf_grid.best_params_
    rf_final = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    rf_final.fit(X_train, y_train)
    
    #evaluacion
    def evaluate_model(model, X, y_true):
        y_pred = model.predict(X)
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
    
    return {
        'models': {
            'Base': rf_base,
            'Tuneado': rf_tuned,
            'Final': rf_final
        },
        'metrics': {
            'Base': evaluate_model(rf_base, X_test, y_test),
            'Tuneado': evaluate_model(rf_tuned, X_test, y_test),
            'Final': evaluate_model(rf_final, X_test, y_test)
        },
        'best_params': best_params
    }

def visualize_results(results, df):
    
    #ComparaciOn de modelos
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    models = list(results['metrics'].keys())
    rmse_values = [results['metrics'][m]['RMSE'] for m in models]
    r2_values = [results['metrics'][m]['R2'] for m in models]
    
    bars = ax1.bar(models, rmse_values, color='lightcoral', alpha=0.7)
    ax1.set_ylabel('RMSE', color='lightcoral')
    ax1.tick_params(axis='y', labelcolor='lightcoral')
    
    ax2 = ax1.twinx()
    ax2.plot(models, r2_values, 'o-', color='teal', linewidth=2, markersize=8)
    ax2.set_ylabel('R² Score', color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')
    
    # Añadir valores
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}', ha='center', va='bottom')
    
    for i, txt in enumerate(r2_values):
        ax2.text(i, txt, f'{txt:.3f}', ha='center', va='bottom')
    
    plt.title('Comparación de Modelos')
    plt.tight_layout()
    plt.savefig('comparacion_modelos.png', dpi=300)
    plt.show()
    

def main():
    """Función principal"""
    # 1. Cargar datos
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    df = pd.read_excel('resumen_estadistico_divorcios_2013_2023.xlsx', 
                      sheet_name='Datos por Departamento')
    
    # 2. Modelado
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 3. Resultados
    print("\n=== RESULTADOS ===")
    for name, metrics in results['metrics'].items():
        print(f"\nModelo {name}:")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"R²: {metrics['R2']:.4f}")
    
    print("\nMejores parámetros encontrados:")
    print(results['best_params'])
    
    # 4. Guardar modelo final
    joblib.dump(results['models']['Final'], 'modelo_final_divorcios.pkl')
    
    # 5. Visualización
    visualize_results(results, df)

if __name__ == "__main__":
    main()