import os
import pandas as pd
import numpy as np
import pyreadstat

os.makedirs("datos_procesados", exist_ok=True)

# ==============================================================================
# 1. CARGAR Y EXPLORAR CADA TABLA (TAMAÑO, TIPOS, NULOS, DUPLICADOS)
# ==============================================================================
print("=" * 70)
print("1. EXPLORACIÓN DE TABLAS ORIGINALES")
print("=" * 70)

df_per, meta_per = pyreadstat.read_sav("EH2025_Persona.sav")
df_viv, meta_viv = pyreadstat.read_sav("EH2025_Vivienda_1.sav")
df_seg, meta_seg = pyreadstat.read_sav("EH2025_Seguridad_Alimentaria.sav")
df_eq,  meta_eq  = pyreadstat.read_sav("EH2025_Equipamiento.sav")

tablas = {
    "Persona": df_per,
    "Vivienda": df_viv,
    "Seguridad_Alimentaria": df_seg,
    "Equipamiento": df_eq
}

for nombre, d in tablas.items():
    print(f"\nTabla: {nombre}")
    print(f"  - Tamaño: {d.shape[0]:,} filas x {d.shape[1]} columnas")
    print(f"  - Tipos: {d.dtypes.value_counts().to_dict()}")
    print(f"  - Total celdas nulas: {int(d.isnull().sum().sum()):,}")
    print(f"  - Duplicados exactos: {int(d.duplicated().sum())}")
    if "folio" in d.columns:
        print(f"  - Folios únicos: {d['folio'].nunique():,}")

def obtener_significado_codigos(meta, var_name, extra_note=""):
    label_set = meta.variable_to_label.get(var_name)
    if label_set and label_set in meta.value_labels:
        vals = meta.value_labels[label_set]
        texto = "; ".join([f"{int(k) if float(k).is_integer() else k}={v}" for k, v in vals.items()])
        return f"{texto} {extra_note}".strip()
    return extra_note.strip() if extra_note else "Valor numérico continuo / conteo"

def obtener_desc_original(meta, var_name, fallback=""):
    return meta.column_names_to_labels.get(var_name, fallback)

# ==============================================================================
# 2. LIMPIAR, RECODIFICAR Y AGREGAR PERSONA A NIVEL HOGAR (CON ETIQUETA POBREZA)
# ==============================================================================
print("\n" + "=" * 70)
print("2. LIMPIEZA, RECODIFICACIÓN Y AGREGACIÓN DE PERSONA")
print("=" * 70)

# --- Vivienda (sin s06a_01 ni s06a_02) ---
cols_viv = [
    'folio', 'depto', 'area', 'factor',
    's06a_06', 's06a_03', 's06a_05', 's06a_07', 's06a_09', 's06a_12', 's06a_15', 's06a_19',
    's06a_17'
]
viv_limpia = df_viv[cols_viv].copy()
viv_limpia['s06a_17'] = viv_limpia['s06a_17'].replace(0, 1).fillna(1)

# --- Seguridad alimentaria ---
preguntas_fies = [f's07a_0{i}' for i in range(1, 9)]
seg_limpia = df_seg[['folio'] + preguntas_fies].copy()
for col in preguntas_fies:
    seg_limpia[col] = np.where(seg_limpia[col] == 1, 1, 0)

puntaje_fies = seg_limpia[preguntas_fies].sum(axis=1)
seg_limpia['nivel_inseguridad_alim'] = pd.cut(
    puntaje_fies, bins=[-1, 0, 3, 6, 8], labels=[0, 1, 2, 3]
).astype(int)
seg_limpia = seg_limpia[['folio', 'nivel_inseguridad_alim']]

# --- Equipamiento (excluyendo celular [item 9], radio, minicomponente y bicicleta) ---
etiquetas_items_eq = meta_eq.value_labels.get(meta_eq.variable_to_label.get('item'), {})
df_eq['posee'] = np.where(df_eq['s08c_1'] == 1, 1, 0)
eq_pivot = df_eq.pivot(index='folio', columns='item', values='posee').reset_index()
eq_pivot.columns = ['folio'] + [f'equip_item_{int(c)}' for c in eq_pivot.columns[1:]]
cols_todos_items = [c for c in eq_pivot.columns if c.startswith('equip_item_')]
eq_pivot['total_activos_hogar'] = eq_pivot[cols_todos_items].sum(axis=1)

items_excluir = {'equip_item_9'}
palabras_ruido = ['radio', 'minicomponente', 'bicicleta']
for i in range(1, 19):
    col = f'equip_item_{i}'
    nombre = etiquetas_items_eq.get(float(i), etiquetas_items_eq.get(i, f"Ítem {i}"))
    if any(p in nombre.lower() for p in palabras_ruido):
        items_excluir.add(col)

cols_items_validos = [c for c in cols_todos_items if c not in items_excluir]
eq_limpia = eq_pivot[['folio', 'total_activos_hogar'] + cols_items_validos].copy()

# --- Persona -> Agregación a nivel hogar + etiqueta de pobreza (p0, pext0) + jefe ---
per = df_per.copy()
assert (per.groupby('folio')['p0'].nunique() == 1).all(), "p0 varía dentro de un hogar"
assert (per.groupby('folio')['pext0'].nunique() == 1).all(), "pext0 varía dentro de un hogar"

per['es_menor_14'] = (per['s01a_03'] < 14).astype(int)
per['es_adulto_mayor'] = (per['s01a_03'] >= 60).astype(int)
per['esta_ocupado'] = (per['condact'] == 1).astype(int)

agg_hogar = per.groupby('folio').agg(
    tamano_hogar=('nro', 'count'),
    num_menores_14=('es_menor_14', 'sum'),
    num_adultos_mayores=('es_adulto_mayor', 'sum'),
    num_ocupados=('esta_ocupado', 'sum'),
    p0=('p0', 'first'),
    pext0=('pext0', 'first')
).reset_index()

jefes = per[per['s01a_05'] == 1].drop_duplicates(subset=['folio'], keep='first').copy()
mapa_jefe = {
    's01a_02': 'jefe_sexo',
    's01a_09': 'jefe_indigena',
    's01a_10': 'jefe_estado_civil',
    'aestudio': 'jefe_anios_estudio',
    's04b_12': 'jefe_tipo_ocupacion'
}
jefes_limpio = jefes[['folio'] + list(mapa_jefe.keys())].rename(columns=mapa_jefe)
jefes_limpio['jefe_tipo_ocupacion'] = jefes_limpio['jefe_tipo_ocupacion'].fillna(0)
jefes_limpio['jefe_anios_estudio'] = jefes_limpio['jefe_anios_estudio'].fillna(0)

# ==============================================================================
# 3. UNIR POR FOLIO, VALIDAR HOGARES Y ENTREGAR base_hogares.csv + DICCIONARIO
# ==============================================================================
print("\n" + "=" * 70)
print("3. UNIÓN POR FOLIO, VALIDACIÓN Y EXPORTACIÓN")
print("=" * 70)

m1 = viv_limpia.merge(agg_hogar, on='folio', how='outer', indicator='_m1')
m2 = m1.drop(columns=['_m1']).merge(jefes_limpio, on='folio', how='outer', indicator='_m2')
m3 = m2.drop(columns=['_m2']).merge(seg_limpia, on='folio', how='outer', indicator='_m3')
base_hogares = m3.drop(columns=['_m3']).merge(eq_limpia, on='folio', how='outer', indicator='_m4')

print("Validación cruce Vivienda <-> Persona:", m1['_m1'].value_counts().to_dict())
print("Validación cruce + Jefes             :", m2['_m2'].value_counts().to_dict())
print("Validación cruce + Seg. Alimentaria  :", m3['_m3'].value_counts().to_dict())
print("Validación cruce + Equipamiento      :", base_hogares['_m4'].value_counts().to_dict())

# Quitar indicador y 'folio' (no sirve como variable)
base_hogares = base_hogares.drop(columns=['_m4', 'folio'])

# Variables derivadas (espacio y rangos)
base_hogares['personas_por_dormitorio'] = (base_hogares['tamano_hogar'] / base_hogares['s06a_17']).round(2)
base_hogares['hacinamiento'] = (base_hogares['personas_por_dormitorio'] > 2.5).astype(int)
base_hogares = base_hogares.drop(columns=['s06a_17'])

base_hogares['jefe_estudio_rango'] = pd.cut(
    base_hogares['jefe_anios_estudio'], bins=[-1, 6, 12, 30], labels=[0, 1, 2]
).astype(int)
base_hogares['tamano_hogar_rango'] = pd.cut(
    base_hogares['tamano_hogar'], bins=[0, 2, 4, 30], labels=[0, 1, 2]
).astype(int)

# Recodificar tipos enteros sin decimales
for col in [c for c in base_hogares.columns if c not in ['factor', 'personas_por_dormitorio']]:
    base_hogares[col] = base_hogares[col].round().astype('Int64')

assert len(base_hogares) == 12726, f"Se esperaban 12.726 hogares, hay {len(base_hogares)}"
print(f"[OK] Validación de hogares exitosa: {len(base_hogares):,} hogares x {base_hogares.shape[1]} variables.")

base_hogares.to_csv("datos_procesados/base_hogares.csv", index=False)

# --- Diccionario propio ---
redundancia_map = {
    'personas_por_dormitorio': 'Espacio',
    'hacinamiento': 'Espacio',
    'tamano_hogar': 'Tamaño hogar',
    'tamano_hogar_rango': 'Tamaño hogar',
    'jefe_anios_estudio': 'Educación jefe',
    'jefe_estudio_rango': 'Educación jefe'
}

especificaciones = [
    ('factor', 'Factor de expansión', 'Factor de ponderación muestral INE', 'Vivienda', 'Numérica continua', 'Peso muestral', 'Peso / Descriptor'),
    ('p0', 'Pobreza moderada', 'Condición de pobreza moderada del hogar', 'Persona', 'Binaria', obtener_significado_codigos(meta_per, 'p0'), 'Etiqueta / Descriptor'),
    ('pext0', 'Pobreza extrema', 'Condición de pobreza extrema del hogar', 'Persona', 'Binaria', obtener_significado_codigos(meta_per, 'pext0'), 'Etiqueta / Descriptor'),
    ('depto', 'Departamento', 'Departamento de residencia', 'Vivienda', 'Categórica', obtener_significado_codigos(meta_viv, 'depto'), 'Descriptor'),
    ('area', 'Área geográfica', 'Área urbana o rural', 'Vivienda', 'Binaria', obtener_significado_codigos(meta_viv, 'area'), 'Condiciones de vida'),
    ('s06a_06', 'Material piso', obtener_desc_original(meta_viv, 's06a_06'), 'Vivienda', 'Categórica', obtener_significado_codigos(meta_viv, 's06a_06'), 'Condiciones de vida'),
    ('s06a_03', 'Material paredes', obtener_desc_original(meta_viv, 's06a_03'), 'Vivienda', 'Categórica', obtener_significado_codigos(meta_viv, 's06a_03'), 'Condiciones de vida'),
    ('s06a_05', 'Material techo', obtener_desc_original(meta_viv, 's06a_05'), 'Vivienda', 'Categórica', obtener_significado_codigos(meta_viv, 's06a_05'), 'Condiciones de vida'),
    ('s06a_07', 'Procedencia agua', obtener_desc_original(meta_viv, 's06a_07'), 'Vivienda', 'Ordinal', obtener_significado_codigos(meta_viv, 's06a_07'), 'Condiciones de vida'),
    ('s06a_09', 'Tipo de baño', obtener_desc_original(meta_viv, 's06a_09'), 'Vivienda', 'Ordinal', obtener_significado_codigos(meta_viv, 's06a_09'), 'Condiciones de vida'),
    ('s06a_12', 'Energía eléctrica', obtener_desc_original(meta_viv, 's06a_12'), 'Vivienda', 'Binaria', obtener_significado_codigos(meta_viv, 's06a_12'), 'Condiciones de vida'),
    ('s06a_15', 'Combustible cocina', obtener_desc_original(meta_viv, 's06a_15'), 'Vivienda', 'Categórica', obtener_significado_codigos(meta_viv, 's06a_15'), 'Condiciones de vida'),
    ('s06a_19', 'Acceso a internet', obtener_desc_original(meta_viv, 's06a_19'), 'Vivienda', 'Binaria', obtener_significado_codigos(meta_viv, 's06a_19'), 'Condiciones de vida'),
    ('personas_por_dormitorio', 'Personas por dormitorio', 'Miembros del hogar por dormitorio', 'Ingeniería', 'Numérica continua', 'Ratio personas/dormitorio', 'Condiciones de vida'),
    ('hacinamiento', 'Hacinamiento', 'Más de 2.5 personas por dormitorio', 'Ingeniería', 'Binaria', '0=No; 1=Sí (>2.5)', 'Carencias / Reglas'),
    ('tamano_hogar', 'Tamaño del hogar', 'Número total de miembros del hogar', 'Persona', 'Numérica discreta', 'Conteo de personas', 'Condiciones de vida'),
    ('tamano_hogar_rango', 'Rango tamaño hogar', 'Tamaño del hogar en rangos', 'Ingeniería', 'Ordinal', '0=Bajo (1-2); 1=Medio (3-4); 2=Alto (5+)', 'Rangos / Reglas'),
    ('num_menores_14', 'Menores de 14 años', 'Cantidad de miembros <14 años', 'Persona', 'Numérica discreta', 'Conteo de menores', 'Condiciones de vida'),
    ('num_adultos_mayores', 'Adultos mayores', 'Cantidad de miembros >=60 años', 'Persona', 'Numérica discreta', 'Conteo adultos mayores', 'Condiciones de vida'),
    ('num_ocupados', 'Miembros ocupados', 'Cantidad de miembros ocupados', 'Persona', 'Numérica discreta', 'Conteo ocupados', 'Condiciones de vida'),
    ('jefe_anios_estudio', 'Años estudio jefe', 'Años de estudio aprobados del jefe/a', 'Persona (Jefe)', 'Numérica discreta', '0 a 22 años', 'Condiciones de vida'),
    ('jefe_estudio_rango', 'Rango estudio jefe', 'Escolaridad del jefe/a en rangos', 'Ingeniería', 'Ordinal', '0=Bajo (0-6); 1=Medio (7-12); 2=Alto (>12)', 'Rangos / Reglas'),
    ('jefe_tipo_ocupacion', 'Categoría ocupacional jefe', 'Tipo de ocupación del jefe/a', 'Persona (Jefe)', 'Categórica', obtener_significado_codigos(meta_per, 's04b_12', '| 0=No aplica'), 'Condiciones de vida'),
    ('total_activos_hogar', 'Total activos hogar', 'Suma de bienes durables del hogar', 'Equipamiento', 'Numérica discreta', '0 a 18 bienes', 'Condiciones de vida'),
    ('nivel_inseguridad_alim', 'Nivel inseguridad alim.', 'Nivel de inseguridad alimentaria FIES', 'Seg. Alimentaria', 'Ordinal', '0=Segura; 1=Leve; 2=Moderada; 3=Severa', 'Descriptor'),
    ('jefe_sexo', 'Sexo jefe/a', 'Sexo del jefe/a de hogar', 'Persona (Jefe)', 'Binaria', obtener_significado_codigos(meta_per, 's01a_02'), 'Descriptor'),
    ('jefe_indigena', 'Jefe indígena', 'Autoidentificación indígena del jefe/a', 'Persona (Jefe)', 'Categórica', obtener_significado_codigos(meta_per, 's01a_09'), 'Descriptor'),
    ('jefe_estado_civil', 'Estado civil jefe/a', 'Estado civil del jefe/a de hogar', 'Persona (Jefe)', 'Categórica', obtener_significado_codigos(meta_per, 's01a_10'), 'Descriptor'),
]

for col in cols_items_validos:
    idx_num = int(col.replace('equip_item_', ''))
    nombre_item = etiquetas_items_eq.get(float(idx_num), etiquetas_items_eq.get(idx_num, f"Ítem {idx_num}"))
    especificaciones.append((
        col, f'Posee: {nombre_item}', f'Tenencia de {nombre_item}',
        'Equipamiento', 'Binaria', '0=No; 1=Sí', 'Equipamiento / Reglas'
    ))

df_dic = pd.DataFrame(
    especificaciones,
    columns=['nombre_original', 'nombre_legible', 'descripcion', 'tabla_origen', 'tipo_dato', 'significado_codigos', 'rol']
)
df_dic['grupo_redundancia'] = df_dic['nombre_original'].map(redundancia_map).fillna('Ninguno')

assert set(base_hogares.columns) == set(df_dic['nombre_original'])
df_dic.to_csv("datos_procesados/diccionario_datos_eh2025.csv", index=False, encoding='utf-8-sig')

print("[OK] Entregables generados: 'datos_procesados/base_hogares.csv' y 'datos_procesados/diccionario_datos_eh2025.csv'.")