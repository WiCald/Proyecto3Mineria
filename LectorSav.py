import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

def normalize(text):
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

# Leer todos los .sav desde la carpeta donde esté este script
script_dir = Path(__file__).resolve().parent
sav_files = sorted(script_dir.glob("divorcios*.sav"))

all_rows, stats = [], {}

for file in sav_files:
    year = int(file.stem.replace("divorcios",""))
    df = pd.read_spss(file)
    counts = df['DEPOCU'].value_counts().reset_index()
    counts.columns = ['Departamento','Divorcios']
    counts['Departamento'] = counts['Departamento'].apply(normalize)
    counts['Year'] = year

    all_rows.append(counts)
    stats[year] = compute_stats(counts['Divorcios'])

df_all = pd.concat(all_rows, ignore_index=True)
df_stats = pd.DataFrame(stats).T

# Pivotear datos
df_pivot = df_all.pivot_table(
    index='Departamento', 
    columns='Year', 
    values='Divorcios', 
    aggfunc='sum', 
    fill_value=0
)
df_pivot['Total'] = df_pivot.sum(axis=1)
df_pivot.reset_index(inplace=True)

# Exportar
output = script_dir / "resumen_estadistico_divorcios_2013_2023.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
    df_pivot.to_excel(writer, sheet_name='Datos por Departamento', index=False)
    df_stats.to_excel(writer, sheet_name='Resumen Estadístico')

print(f"Archivo guardado en: {output}")
