import pandas as pd
from pathlib import Path

# Directorio donde están los archivos divorcios*.sav (actual directorio de trabajo)
dir_path = Path.cwd()
sav_files = sorted(dir_path.glob("divorcios*.sav"))

# Cargar y contar divorcios por departamento y año
all_counts = []
for file in sav_files:
    try:
        year = int(file.stem.replace("divorcios", ""))
    except ValueError:
        continue  # Ignorar archivos que no sigan el patrón
    df = pd.read_spss(file)
    # Ajusta el nombre de la columna si tu SPSS usa otro
    dept_col = next((c for c in df.columns if c.upper() == 'DEPOCU'), None)
    df['Departamento'] = df[dept_col] if dept_col else 'No especificado'
    counts = df['Departamento'].value_counts().reset_index()
    counts.columns = ['Departamento', 'Divorcios']
    counts['Year'] = year
    all_counts.append(counts)

if not all_counts:
    print("No se encontraron archivos divorcios*.sav o no hay datos.")
    exit()

# Consolidar en un solo DataFrame
df_all = pd.concat(all_counts, ignore_index=True)

# Pivot para obtener columnas por año
df_pivot = df_all.pivot_table(
    index='Departamento',
    columns='Year',
    values='Divorcios',
    aggfunc='sum',
    fill_value=0
).reset_index()

# Detectar valores atípicos por año usando IQR
years = [c for c in df_pivot.columns if isinstance(c, int)]
outliers = []
for year in years:
    # Serie de divorcios por departamento para el año
    series = df_pivot.set_index('Departamento')[year]
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    # Mascara de outliers
    mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
    for dept, val in series[mask].items():
        outliers.append((year, dept, int(val)))

# Mostrar resultados
if outliers:
    print(f"Total de valores atípicos detectados: {len(outliers)}")
    for year, dept, val in outliers:
        print(f"Año {year}, Departamento: {dept}, Divorcios: {val}")
else:
    print("No se detectaron valores atípicos.")