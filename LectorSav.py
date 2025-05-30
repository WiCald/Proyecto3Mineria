import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import LabelEncoder

# Funciones auxiliares

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


# Directorio y archivos SAV
script_dir = Path(__file__).resolve().parent
sav_files = sorted(script_dir.glob("divorcios*.sav"))

# Estructuras para almacenar
all_rows, stats, month_summary, raw_frames = [], {}, [], []

# Procesar cada año
for file in sav_files:
    year = int(file.stem.replace("divorcios", ""))
    df = pd.read_spss(file)
    df['Year'] = year

    # Columnas clave
    dept_col = next((c for c in df.columns if c.upper() == 'DEPOCU'), None)
    month_col = next((c for c in df.columns if c.upper() == 'MESOCU'), None)

    # Normalizar categóricas y unificar variantes
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.add_categories('')
        df[col] = df[col].fillna('').astype(str).apply(normalize).str.strip()
        df[col] = df[col].replace({
            "Mestizo / Ladino": "Mestizo/Ladino",
            "Mestizo - Ladino": "Mestizo/Ladino",
            "Mestizo, Ladino": "Mestizo/Ladino",
        })
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.remove_unused_categories()

    # Crear columnas estándar
    df['Departamento'] = df[dept_col] if dept_col is not None else 'No especificado'
    df['Mes'] = df[month_col] if month_col is not None else 'No especificado'
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

# Datos concatenados
df_all = pd.concat(all_rows, ignore_index=True)
df_stats = pd.DataFrame(stats).T
df_month = pd.DataFrame(month_summary).sort_values('Year')

# ----------------------
# Añadido: Conteo de divorcios por Mes y Año
df_all_raw = pd.concat(raw_frames, ignore_index=True)
df_mes_year = (
    df_all_raw
    .groupby(['Year', 'Mes'], observed=False)
    .size()
    .reset_index(name='Divorcios')
    .sort_values(['Year', 'Mes'])
)
# ----------------------

# Pivot: Departamentos vs Años
df_pivot = df_all.pivot_table(
    index='Departamento',
    columns='Year',
    values='Divorcios',
    aggfunc='sum',
    fill_value=0
)
df_pivot['Total'] = df_pivot.sum(axis=1)
df_pivot.reset_index(inplace=True)

# Frecuencias categóricas
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

# Exportar Excel
output = script_dir / "resumen_estadistico_divorcios_2013_2023.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
    df_pivot.to_excel(writer, sheet_name='Datos por Departamento', index=False)
    df_stats.to_excel(writer, sheet_name='Resumen Estadístico')
    df_month.to_excel(writer, sheet_name='Mes con más divorcios', index=False)
    df_mes_year.to_excel(writer, sheet_name='Divorcios Mes y Año', index=False)
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


# OCUPACIONES

if 'CIUOHOM' not in df.columns or 'CIUOMUJ' not in df.columns:
    print("Error: No se encontraron las columnas CIUOHOM y/o CIUOMUJ en los datos")
    exit()

# Limpieza de ocupaciones
df['Ocupacion_Hombre'] = df['CIUOHOM'].astype(str).apply(normalize).str.strip()
df['Ocupacion_Mujer'] = df['CIUOMUJ'].astype(str).apply(normalize).str.strip()

# Filtrar valores no válidos
invalid_values = ['', 'No Especificado', 'Ignorado', 'No Aplica', 'Nan', 'None']
df = df[~df['Ocupacion_Hombre'].isin(invalid_values) & ~df['Ocupacion_Mujer'].isin(invalid_values)]

## 1. Frecuencia de ocupaciones en divorcios (Top 10)
print("\n" + "="*80)
print("FRECUENCIA DE OCUPACIONES EN DIVORCIOS (TOP 10)")
print("="*80)

# Para hombres
top_ocup_hombres = df['Ocupacion_Hombre'].value_counts().head(10)
print("\nOcupaciones más comunes en hombres divorciados:")
print(top_ocup_hombres.to_string())

# Para mujeres
top_ocup_mujeres = df['Ocupacion_Mujer'].value_counts().head(10)
print("\nOcupaciones más comunes en mujeres divorciadas:")
print(top_ocup_mujeres.to_string())

# Gráfico comparativo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Ocupaciones más comunes en divorcios')

# Gráfico hombres
ax1.barh(top_ocup_hombres.index, top_ocup_hombres.values, color='skyblue')
ax1.set_title('Hombres')
ax1.set_xlabel('Número de Divorcios')

# Gráfico mujeres
ax2.barh(top_ocup_mujeres.index, top_ocup_mujeres.values, color='lightcoral')
ax2.set_title('Mujeres')
ax2.set_xlabel('Número de Divorcios')

plt.tight_layout()
plt.show()

## 2. Correlación entre ocupaciones de la pareja
print("\n" + "="*80)
print("CORRELACIÓN ENTRE OCUPACIONES DE LA PAREJA")
print("="*80)

# Codificar ocupaciones numéricamente para el análisis
le = LabelEncoder()
ocup_combinadas = pd.concat([df['Ocupacion_Hombre'], df['Ocupacion_Mujer']]).unique()
le.fit(ocup_combinadas)

df['Ocupacion_Hombre_Code'] = le.transform(df['Ocupacion_Hombre'])
df['Ocupacion_Mujer_Code'] = le.transform(df['Ocupacion_Mujer'])

# Calcular correlación
corr, p_value = pointbiserialr(df['Ocupacion_Hombre_Code'], df['Ocupacion_Mujer_Code'])
print(f"\nCorrelación entre ocupaciones de la pareja: {corr:.3f} (p-value: {p_value:.4f})")

# Interpretación
if p_value < 0.05:
    print("Existe una correlación estadísticamente significativa entre las ocupaciones de los cónyuges.")
    print("Las parejas tienden a tener ocupaciones similares." if corr > 0 else "Las parejas tienden a tener ocupaciones diferentes.")
else:
    print("No hay evidencia de correlación significativa entre las ocupaciones de los cónyuges.")

# Gráfico de dispersión con regresión
plt.figure(figsize=(10, 6))
plt.scatter(df['Ocupacion_Hombre_Code'], df['Ocupacion_Mujer_Code'], alpha=0.3)

# Añadir línea de regresión
z = np.polyfit(df['Ocupacion_Hombre_Code'], df['Ocupacion_Mujer_Code'], 1)
p = np.poly1d(z)
plt.plot(df['Ocupacion_Hombre_Code'], p(df['Ocupacion_Hombre_Code']), "r--")

plt.title('Relación entre Ocupaciones de la Pareja')
plt.xlabel('Código Ocupación Hombre')
plt.ylabel('Código Ocupación Mujer')
plt.grid(True)
plt.show()

## 3. Combinaciones más frecuentes
print("\n" + "="*80)
print("COMBINACIONES DE OCUPACIONES MÁS FRECUENTES")
print("="*80)

# Crear variable combinada
df['Combinacion_Ocupaciones'] = df['Ocupacion_Hombre'] + " + " + df['Ocupacion_Mujer']
top_combinaciones = df['Combinacion_Ocupaciones'].value_counts().head(10)

print("\nCombinaciones más frecuentes:")
print(top_combinaciones.to_string())

# Gráfico de combinaciones
plt.figure(figsize=(12, 6))
plt.barh(top_combinaciones.index, top_combinaciones.values, color='lightgreen')
plt.title('Top 10 Combinaciones de Ocupaciones en Divorcios')
plt.xlabel('Número de Divorcios')
plt.gca().invert_yaxis()  # Mostrar el más frecuente arriba
plt.grid(axis='x')
plt.show()

## 4. Análisis por sector ocupacional
print("\n" + "="*80)
print("DISTRIBUCIÓN POR SECTORES OCUPACIONALES")
print("="*80)

# Definición simplificada de sectores
sectores = {
    'Profesional': ['Ingeniero', 'Médico', 'Abogado', 'Arquitecto', 'Doctor', 'Licenciado'],
    'Técnico': ['Técnico', 'Tecnólogo', 'Analista', 'Programador'],
    'Administrativo': ['Administrativo', 'Secretario', 'Asistente', 'Oficinista'],
    'Comercio': ['Comerciante', 'Vendedor', 'Cajero'],
    'Servicios': ['Mesero', 'Cocinero', 'Chofer', 'Seguridad'],
    'Otros': []
}

def asignar_sector(ocupacion):
    ocupacion = ocupacion.lower()
    for sector, palabras_clave in sectores.items():
        if any(palabra.lower() in ocupacion for palabra in palabras_clave):
            return sector
    return 'Otros'

df['Sector_Hombre'] = df['Ocupacion_Hombre'].apply(asignar_sector)
df['Sector_Mujer'] = df['Ocupacion_Mujer'].apply(asignar_sector)

# Tabla de contingencia
contingency = pd.crosstab(df['Sector_Hombre'], df['Sector_Mujer'])
print("\nTabla de contingencia de sectores:")
print(contingency)

# Prueba chi-cuadrado
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nTest Chi-Cuadrado: chi2 = {chi2:.2f}, p-value = {p:.4f}")

# Gráfico de calor manual
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(contingency, cmap='YlOrRd')

# Añadir valores en las celdas
for (i, j), val in np.ndenumerate(contingency):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')

# Configuración del gráfico
ax.set_xticks(np.arange(len(contingency.columns)))
ax.set_yticks(np.arange(len(contingency.index)))
ax.set_xticklabels(contingency.columns, rotation=45)
ax.set_yticklabels(contingency.index)
ax.xaxis.set_ticks_position('bottom')
plt.title('Combinaciones de Sectores Ocupacionales')
plt.xlabel('Sector Mujer')
plt.ylabel('Sector Hombre')
plt.colorbar(cax, label='Número de Divorcios')
plt.show()