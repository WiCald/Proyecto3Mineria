import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import matplotlib.pyplot as plt

def normalize(text):
    """Quita tildes y convierte el texto a Title Case."""
    nfkd = unicodedata.normalize('NFKD', str(text))
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).title()

def compute_stats(series):
    s = pd.to_numeric(series, errors='coerce')
    return pd.Series({
        'Mean': s.mean(), 
        'Median': s.median(),
        'Mode': s.mode().iloc[0] if not s.mode().empty else np.nan,
        'Min': s.min(), 
        'Q1 (25%)': s.quantile(0.25),
        'Q3 (75%)': s.quantile(0.75), 
        'Max': s.max(),
        'Range': s.max() - s.min(), 
        'IQR': s.quantile(0.75) - s.quantile(0.25),
        'Variance': s.var(), 
        'Std Dev': s.std(),
        'Coef Variation (%)': (s.std() / s.mean() * 100) if s.mean() else np.nan,
        'Skewness': s.skew(), 
        'Kurtosis': s.kurtosis(),
        '10th Percentile': s.quantile(0.10), 
        '90th Percentile': s.quantile(0.90)
    })

def get_column(df, desired):
    """Busca en df una columna cuyo nombre, en mayúsculas, sea igual a desired."""
    for col in df.columns:
        if col.upper() == desired.upper():
            return col
    return None

# Directorio donde se encuentran este script (.py) y los archivos .sav
script_dir = Path(__file__).resolve().parent
sav_files = sorted(script_dir.glob("divorcios*.sav"))

all_rows, stats, month_summary, raw_frames = [], {}, [], []

for file in sav_files:
    year = int(file.stem.replace("divorcios", ""))
    df = pd.read_spss(file)
    df['Year'] = year

    # Detectar columnas clave de forma insensible a mayúsculas
    dept_col = next((c for c in df.columns if c.upper() == 'DEPOCU'), None)
    month_col = next((c for c in df.columns if c.upper() == 'MESOCU'), None)
    
    # Normalizar todas las columnas categóricas y unificar variantes
    for col in df.select_dtypes(include=['object', 'category']).columns:
        # Si la columna es categórica, se agrega la categoría '' para evitar errores
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.add_categories('')
        
        df[col] = df[col].fillna('').astype(str).apply(normalize).str.strip()
        
        # Unificar variantes (ejemplo para "Mestizo/Ladino")
        df[col] = df[col].replace({
            "Mestizo / Ladino": "Mestizo/Ladino",
            "Mestizo - Ladino": "Mestizo/Ladino",
            "Mestizo, Ladino": "Mestizo/Ladino",
            # Agrega aquí otros reemplazos según sea necesario
        })
        
        # Si la columna es categórica, actualizar las categorías para quitar las no usadas
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.remove_unused_categories()
    
    # Crear columnas estándar a partir de las detectadas
    if dept_col is not None:
        df['Departamento'] = df[dept_col]
    else:
        df['Departamento'] = 'No especificado'
    
    if month_col is not None:
        df['Mes'] = df[month_col]
    else:
        df['Mes'] = 'No especificado'
    
    raw_frames.append(df)

    # Conteo por Departamento
    counts = df['Departamento'].value_counts().reset_index()
    counts.columns = ['Departamento', 'Divorcios']
    counts['Year'] = year
    all_rows.append(counts)
    stats[year] = compute_stats(counts['Divorcios'])

    # Mes con más divorcios
    mc = df['Mes'].value_counts()
    month_summary.append({
        'Year': year,
        'Mes con más divorcios': mc.idxmax() if not mc.empty else 'No especificado',
        'Total divorcios': mc.max() if not mc.empty else 0
    })

df_all = pd.concat(all_rows, ignore_index=True)
df_stats = pd.DataFrame(stats).T
df_month = pd.DataFrame(month_summary).sort_values('Year')

# Pivot: Tabla de datos por Departamento con años y Total
df_pivot = df_all.pivot_table(
    index='Departamento',
    columns='Year',
    values='Divorcios',
    aggfunc='sum',
    fill_value=0
)
df_pivot['Total'] = df_pivot.sum(axis=1)
df_pivot.reset_index(inplace=True)

# Tablas de frecuencia para variables categóricas (excluyendo NACHOM y NACMUJ)
df_raw = pd.concat(raw_frames, ignore_index=True)
freq_tables = {}
for col in df_raw.select_dtypes(include=['object']).columns:
    if col.upper() in ('NACHOM', 'NACMUJ'):
        continue
    vc = df_raw[col].fillna('No especificado').value_counts(dropna=False)
    freq_tables[col] = pd.DataFrame({
        'Count': vc,
        'Percent (%)': (vc / vc.sum() * 100).round(2)
    })

# Exportar todo a un único Excel
output = script_dir / "resumen_estadistico_divorcios_2013_2023.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
    df_pivot.to_excel(writer, sheet_name='Datos por Departamento', index=False)
    df_stats.to_excel(writer, sheet_name='Resumen Estadístico')
    df_month.to_excel(writer, sheet_name='Mes con más divorcios', index=False)
    for col, table in freq_tables.items():
        sheet_name = f"Freq_{col}"[:31]
        table.to_excel(writer, sheet_name=sheet_name, index=True)

print(f"Archivo Excel generado en: {output}")

# ------------------------------
# Gráficas de Profesión (o Nivel de Educación) y Pueblo/Etnia por Sexo
df_all_raw = pd.concat(raw_frames, ignore_index=True)

# Primero se intenta obtener las columnas de profesión.
col_prof_h = get_column(df_all_raw, "PROFHOM")
col_prof_m = get_column(df_all_raw, "PROFMUJ")

if col_prof_h is None or col_prof_m is None:
    # Si no existen, se recurre a las columnas de nivel de educación.
    col_prof_h = get_column(df_all_raw, "ESCHOM")
    col_prof_m = get_column(df_all_raw, "ESCMUJ")
    prof_title = "Nivel de Educación"
else:
    prof_title = "Profesión"

# Filtrar valores "Ignorado" en ambas columnas
if col_prof_h is None or col_prof_m is None:
    print("No se encontraron columnas para profesión o nivel de educación en hombres y/o mujeres.")
else:
    # Obtener las series de conteos y eliminar la categoría "Ignorado"
    prof_h = df_all_raw[col_prof_h].value_counts().drop("Ignorado", errors='ignore').sort_values(ascending=False)
    prof_m = df_all_raw[col_prof_m].value_counts().drop("Ignorado", errors='ignore').sort_values(ascending=False)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)
    axes[0].bar(prof_h.index, prof_h.values, color='skyblue')
    axes[0].set_title(f'{prof_title} - Hombres')
    axes[0].set_xlabel(prof_title)
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=90)
    
    axes[1].bar(prof_m.index, prof_m.values, color='lightcoral')
    axes[1].set_title(f'{prof_title} - Mujeres')
    axes[1].set_xlabel(prof_title)
    axes[1].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(script_dir / "profesion_o_educacion_por_sexo.png")
    plt.close()
    print(f"Gráfica 'profesion_o_educacion_por_sexo.png' generada mostrando {prof_title.lower()}.")

# Gráfica de Pueblo/Etnia por Sexo (ignorando "Ignorado")
col_pueblo_h = get_column(df_all_raw, "PUEHOM")
col_pueblo_m = get_column(df_all_raw, "PUEMUJ")

if col_pueblo_h is None or col_pueblo_m is None:
    print("No se encontraron las columnas de pueblo/etnia para hombres y/o mujeres.")
else:
    pueblo_h = df_all_raw[col_pueblo_h].value_counts().drop("Ignorado", errors='ignore').sort_values(ascending=False)
    pueblo_m = df_all_raw[col_pueblo_m].value_counts().drop("Ignorado", errors='ignore').sort_values(ascending=False)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)
    axes[0].bar(pueblo_h.index, pueblo_h.values, color='mediumseagreen')
    axes[0].set_title('Pueblo/Etnia - Hombres')
    axes[0].set_xlabel('Pueblo/Etnia')
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=90)
    
    axes[1].bar(pueblo_m.index, pueblo_m.values, color='mediumpurple')
    axes[1].set_title('Pueblo/Etnia - Mujeres')
    axes[1].set_xlabel('Pueblo/Etnia')
    axes[1].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(script_dir / "pueblo_etnia_por_sexo.png")
    plt.close()
    print("Gráfica 'pueblo_etnia_por_sexo.png' generada.")

# ------------------------------
# Comparación de Edades en Intervalos de 5 años
# Se usan las columnas "EDADHOM" y "EDADMUJ" para edades de hombres y mujeres respectivamente.
if "EDADHOM" in df_all_raw.columns and "EDADMUJ" in df_all_raw.columns:
    df_all_raw["EDADHOM"] = pd.to_numeric(df_all_raw["EDADHOM"], errors='coerce')
    df_all_raw["EDADMUJ"] = pd.to_numeric(df_all_raw["EDADMUJ"], errors='coerce')
    
    # Definir intervalos de 5 en 5 años, por ejemplo, de 0 a 100 años.
    bins = range(0, 105, 5)  # Hasta 100 años
    labels = [f"{i}-{i+4}" for i in range(0, 100, 5)]
    
    df_all_raw["Rango_Edad_H"] = pd.cut(df_all_raw["EDADHOM"], bins=bins, labels=labels, right=False)
    df_all_raw["Rango_Edad_M"] = pd.cut(df_all_raw["EDADMUJ"], bins=bins, labels=labels, right=False)
    
    freq_edad_hombres = df_all_raw["Rango_Edad_H"].value_counts().sort_index()
    freq_edad_mujeres = df_all_raw["Rango_Edad_M"].value_counts().sort_index()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)
    
    axes[0].bar(freq_edad_hombres.index, freq_edad_hombres.values, color='steelblue')
    axes[0].set_title("Distribución de Edades - Hombres")
    axes[0].set_xlabel("Edad (intervalos de 5 años)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(freq_edad_mujeres.index, freq_edad_mujeres.values, color='salmon')
    axes[1].set_title("Distribución de Edades - Mujeres")
    axes[1].set_xlabel("Edad (intervalos de 5 años)")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(script_dir / "edad_por_sexo.png")
    plt.close()
    print("Gráfica 'edad_por_sexo.png' generada.")
else:
    print("No se encontraron las columnas de edad para hombres y/o mujeres. Verifica los nombres de las columnas.")

