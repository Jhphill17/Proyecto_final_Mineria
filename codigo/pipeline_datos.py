# -*- coding: utf-8 -*-
"""
Pipeline de datos - Proyecto final Mineria de Datos (pobreza por ingreso, EH 2025 INE)
Genera desde los .sav crudos:
  - datos/procesados/base_hogares_completa.csv   (folio + predictores + factor + p0 + pext0)
  - datos/procesados/X_predictores.csv           (folio + predictores)
  - datos/procesados/metadatos_y_etiquetas.csv   (folio + factor + p0 + pext0)
  - datos/procesados/diccionario_propio.csv
  - docs/lista_negra_ingreso.csv
Uso: python codigo/pipeline_datos.py   (desde la raiz del proyecto)
"""
import os
import re
import numpy as np
import pandas as pd
import pyreadstat

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRUDOS = os.path.join(RAIZ, "datos", "crudos")
# Si no existe datos/crudos, usar los .sav de la raiz (estructura actual del equipo)
if not os.path.isdir(CRUDOS):
    CRUDOS = RAIZ
PROC = os.path.join(RAIZ, "datos", "procesados")
DOCS = os.path.join(RAIZ, "docs")
os.makedirs(PROC, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

# ==============================================================================
# 0. CARGA
# ==============================================================================
print("=" * 70)
print("0. CARGA DE TABLAS CRUDAS (.sav)")
print("=" * 70)

df_per, meta_per = pyreadstat.read_sav(os.path.join(CRUDOS, "EH2025_Persona.sav"))
df_viv, meta_viv = pyreadstat.read_sav(os.path.join(CRUDOS, "EH2025_Vivienda_1.sav"))
df_seg, meta_seg = pyreadstat.read_sav(os.path.join(CRUDOS, "EH2025_Seguridad_Alimentaria.sav"))
df_eq,  meta_eq  = pyreadstat.read_sav(os.path.join(CRUDOS, "EH2025_Equipamiento.sav"))

for d in (df_per, df_viv, df_seg, df_eq):
    d["folio"] = d["folio"].astype(str).str.strip()

# ==============================================================================
# 1. LISTA NEGRA DE INGRESO (anti-fuga) - se genera y se verifica
# ==============================================================================
print("\n1. LISTA NEGRA DE INGRESO")

# Variables de ingreso/gasto del modulo de persona (etiquetas del diccionario INE)
patron_ingreso = re.compile(r"ingreso|salario|sueldo|gasto|bono|remuner|pag[oó]|ganancia", re.I)
cols_ingreso = []
for c in df_per.columns:
    lab = str(meta_per.column_names_to_labels.get(c, ""))
    if patron_ingreso.search(lab):
        cols_ingreso.append(c)
# Agregados de pobreza/ingreso ya calculados por el INE (no son p0/pext0 pero derivan de ingreso)
cols_ingreso += ["p1", "p2", "pext1", "pext2", "z", "zext"]
cols_ingreso = sorted(set(cols_ingreso))

# Tablas crudas que NO se usan por contener gasto/ingreso o no ser necesarias
TABLAS_EXCLUIDAS = [
    ("EH2025_Gastos_Alimentarios.sav", "Gasto alimentario del hogar (proxy de ingreso)", "No se usa: riesgo de fuga de datos"),
    ("EH2025_Gastos_No_Alimentarios.sav", "Gasto no alimentario del hogar (proxy de ingreso)", "No se usa: riesgo de fuga de datos"),
    ("EH2025_Discriminacion.sav", "Modulo de discriminacion", "No se usa: fuera del alcance del proyecto"),
]

lista_negra = pd.DataFrame(
    [("Persona", c, str(meta_per.column_names_to_labels.get(c, ""))[:120], "Variable de ingreso/gasto o derivada de ingreso")
     for c in cols_ingreso],
    columns=["tabla", "variable", "etiqueta_ined", "motivo"])
for arch, desc, motivo in TABLAS_EXCLUIDAS:
    lista_negra.loc[len(lista_negra)] = ["Tabla completa", arch, desc, motivo]
lista_negra.to_csv(os.path.join(DOCS, "lista_negra_ingreso.csv"), index=False, encoding="utf-8-sig")
print(f"  {len(cols_ingreso)} variables de ingreso/gasto en Persona + 3 tablas excluidas -> docs/lista_negra_ingreso.csv")

# ==============================================================================
# 2. VIVIENDA -> variables de la vivienda (una fila por hogar)
# ==============================================================================
print("\n2. VIVIENDA")
cols_viv = [
    "folio", "depto", "area", "factor",
    "s06a_01", "s06a_02", "s06a_03", "s06a_05", "s06a_06",
    "s06a_07", "s06a_09", "s06a_10", "s06a_11", "s06a_12",
    "s06a_13", "s06a_14", "s06a_15", "s06a_16", "s06a_17",
    "s06a_18", "s06a_19", "s06a_20",
]
viv = df_viv[cols_viv].copy()
assert viv["folio"].is_unique, "Vivienda no tiene una fila por hogar"

# s06a_17 = cuartos usados exclusivamente para dormir. 0 = el hogar no reporta
# cuarto exclusivo (duerme en cuartos de uso multiple). Se imputa 1 (minimo
# un espacio para dormir) -> ver decisiones_limpieza.md
viv["num_dormitorios"] = viv["s06a_17"].replace(0, 1)
n_cero = int((df_viv["s06a_17"] == 0).sum())
viv = viv.drop(columns=["s06a_17"])

# Nulos "no aplica": s06a_10 y s06a_11 solo aplican a ciertos tipos de bano;
# s06a_20 solo aplica a hogares con internet (s06a_19==1). 0 = "no aplica".
viv["s06a_10"] = viv["s06a_10"].fillna(0)
viv["s06a_11"] = viv["s06a_11"].fillna(0)
viv.loc[viv["s06a_19"] != 1, "s06a_20"] = 0
viv["s06a_20"] = viv["s06a_20"].fillna(0)

# ==============================================================================
# 3. SEGURIDAD ALIMENTARIA -> puntaje FIES y nivel
# ==============================================================================
print("\n3. SEGURIDAD ALIMENTARIA (FIES)")
preguntas_fies = [f"s07a_0{i}" for i in range(1, 9)]
seg = df_seg[["folio"] + preguntas_fies].copy()
assert seg["folio"].is_unique
# Codigos: 1=Si, 2=No, 3=NS/NR. NS/NR se trata como No (0) - decision documentada.
for c in preguntas_fies:
    seg[c] = (seg[c] == 1).astype(int)
seg["puntaje_inseguridad_alim"] = seg[preguntas_fies].sum(axis=1)
# FIES estandar: 0=segura, 1-3=leve, 4-6=moderada, 7-8=severa
seg["nivel_inseguridad_alim"] = pd.cut(
    seg["puntaje_inseguridad_alim"], bins=[-1, 0, 3, 6, 8], labels=[0, 1, 2, 3]
).astype(int)
seg = seg[["folio", "puntaje_inseguridad_alim", "nivel_inseguridad_alim"]]

# ==============================================================================
# 4. EQUIPAMIENTO -> una fila por hogar, items 1 a 18
# ==============================================================================
print("\n4. EQUIPAMIENTO")
etiquetas_items = meta_eq.value_labels.get(meta_eq.variable_to_label.get("item"), {})
df_eq["posee"] = (df_eq["s08c_1"] == 1).astype(int)  # 1=Si, 2=No
eq = df_eq.pivot(index="folio", columns="item", values="posee").reset_index()
eq.columns = ["folio"] + [f"equip_item_{int(c)}" for c in eq.columns[1:]]
cols_items = [c for c in eq.columns if c.startswith("equip_item_")]
eq["total_activos_hogar"] = eq[cols_items].sum(axis=1)
assert eq["folio"].is_unique

# ==============================================================================
# 5. PERSONA -> agregados a nivel hogar + variables del jefe + etiquetas
# ==============================================================================
print("\n5. PERSONA -> AGREGACION A NIVEL HOGAR")
per = df_per.copy()

# Etiquetas: unicas por hogar; 21 personas sin p0 pero todos los folios tienen valor
agg_etq = per.groupby("folio").agg(
    p0=("p0", "first"),
    pext0=("pext0", "first"),
    tipohogar=("tipohogar", "first"),
)
assert agg_etq["p0"].notna().all() and agg_etq["pext0"].notna().all()

per["es_menor_14"] = (per["s01a_03"] < 14).astype(int)
per["es_adulto_mayor"] = (per["s01a_03"] >= 60).astype(int)
# condact==1 = ocupado (trabajo la semana pasada, s04a_01==1). condact 4/5
# (p_temporal/p_permanente) no trabajaron la semana pasada -> no ocupados.
per["esta_ocupado"] = (per["condact"] == 1).astype(int)
per["es_15mas"] = (per["s01a_03"] >= 15).astype(int)

hogar = per.groupby("folio").agg(
    tamano_hogar=("nro", "count"),
    num_menores_14=("es_menor_14", "sum"),
    num_adultos_mayores=("es_adulto_mayor", "sum"),
    num_ocupados=("esta_ocupado", "sum"),
    anios_estudio_prom_15mas=("aestudio", lambda s: np.nan),  # placeholder, se calcula abajo
).reset_index()

# Promedio de años de estudio solo de 15+ (evita sesgo a la baja por menores)
per15 = per[per["es_15mas"] == 1].groupby("folio")["aestudio"].mean()
hogar = hogar.drop(columns=["anios_estudio_prom_15mas"]).merge(
    per15.rename("anios_estudio_prom_15mas").reset_index(), on="folio", how="left"
)
hogar["anios_estudio_prom_15mas"] = hogar["anios_estudio_prom_15mas"].fillna(0)

hogar["prop_menores_14"] = hogar["num_menores_14"] / hogar["tamano_hogar"]
hogar["prop_adultos_mayores"] = hogar["num_adultos_mayores"] / hogar["tamano_hogar"]
# Dependientes por ocupado: si no hay ocupados, todo el hogar depende de transferencias
hogar["tasa_dependencia_ocupados"] = np.where(
    hogar["num_ocupados"] > 0,
    (hogar["tamano_hogar"] - hogar["num_ocupados"]) / hogar["num_ocupados"],
    hogar["tamano_hogar"].astype(float),
)

# --- Jefe del hogar (s01a_05 == 1; verificado: exactamente 1 por folio) ---
jefes = per[per["s01a_05"] == 1].copy()
assert jefes["folio"].is_unique, "Hay hogares con 0 o 2+ jefes"
mapa_jefe = {
    "s01a_02": "jefe_sexo",
    "s01a_03": "jefe_edad",
    "s01a_09": "jefe_indigena",
    "s01a_10": "jefe_estado_civil",
    "niv_ed": "jefe_nivel_ed",
    "aestudio": "jefe_anios_estudio",
    "condact": "jefe_condact",
    "s04b_12": "jefe_tipo_ocupacion",
    "cob_op": "jefe_grupo_ocup",
}
jefes = jefes[["folio"] + list(mapa_jefe.keys())].rename(columns=mapa_jefe)
# 0 = "no aplica" para jefes sin ocupacion (2.146 hogares) - NO son faltantes
for c in ["jefe_tipo_ocupacion", "jefe_grupo_ocup"]:
    jefes[c] = jefes[c].fillna(0)
jefes["jefe_anios_estudio"] = jefes["jefe_anios_estudio"].fillna(0)

# ==============================================================================
# 6. UNIR POR FOLIO, VALIDAR Y EXPORTAR LOS TRES CSV
# ==============================================================================
print("\n6. UNION POR FOLIO Y EXPORTACION")
base = (viv
        .merge(hogar, on="folio", how="outer", validate="one_to_one", indicator="_m1")
        .merge(jefes, on="folio", how="outer", validate="one_to_one", indicator="_m2")
        .merge(agg_etq, on="folio", how="outer", validate="one_to_one", indicator="_m3")
        .merge(seg, on="folio", how="outer", validate="one_to_one", indicator="_m4")
        .merge(eq, on="folio", how="outer", validate="one_to_one", indicator="_m5"))
for m in ["_m1", "_m2", "_m3", "_m4", "_m5"]:
    print(f"  {m}:", base[m].value_counts().to_dict())
base = base.drop(columns=[c for c in base.columns if c.startswith("_m")])

# Derivadas de hacinamiento y rangos
base["personas_por_dormitorio"] = (base["tamano_hogar"] / base["num_dormitorios"]).round(2)
base["hacinamiento"] = (base["personas_por_dormitorio"] > 2.5).astype(int)
base["jefe_estudio_rango"] = pd.cut(
    base["jefe_anios_estudio"], bins=[-1, 6, 12, 30], labels=[0, 1, 2]
).astype(int)
base["tamano_hogar_rango"] = pd.cut(
    base["tamano_hogar"], bins=[0, 2, 4, 30], labels=[0, 1, 2]
).astype(int)

# Roles de columnas
ID = ["folio"]
PESO = ["factor"]
ETIQ = ["p0", "pext0"]
predictores = [c for c in base.columns if c not in ID + PESO + ETIQ]

# ANTI-FUGA: ningun predictor puede estar en la lista negra
fugas = set(predictores) & set(cols_ingreso)
assert not fugas, f"Variables de ingreso en predictores: {fugas}"

# Sin nulos en predictores (verificar)
nulos = base[predictores].isna().sum()
nulos = nulos[nulos > 0]
if len(nulos):
    print("  [AVISO] Nulos en predictores:\n", nulos)
else:
    print("  [OK] Sin nulos en predictores")

# Orden de columnas
base = base[ID + predictores + PESO + ETIQ]
base.to_csv(os.path.join(PROC, "base_hogares_completa.csv"), index=False)
base[ID + predictores].to_csv(os.path.join(PROC, "X_predictores.csv"), index=False)
base[ID + PESO + ETIQ].to_csv(os.path.join(PROC, "metadatos_y_etiquetas.csv"), index=False)
print(f"  base_hogares_completa.csv: {base.shape}")
print(f"  X_predictores.csv: {len(predictores)} predictores")

# ==============================================================================
# 7. VALIDACIONES FINALES (cifras de control)
# ==============================================================================
print("\n7. VALIDACIONES")
assert len(base) == 12726, f"Se esperaban 12.726 hogares, hay {len(base)}"
assert base["folio"].is_unique
w = lambda d: np.average(d["p0"], weights=d["factor"])
print(f"  N={len(base)} | p0 sin pond={base.p0.mean()*100:.1f}% | p0 pond={w(base)*100:.1f}% | pext0={base.pext0.mean()*100:.1f}%")
cb = base[base["depto"] == 3]
print(f"  Cochabamba (depto=3): n={len(cb)} | pobreza pond={w(cb)*100:.1f}%")
print(f"  area=1 (Urbano): n={(base.area==1).sum()} | area=2 (Rural): n={(base.area==2).sum()}")

# ==============================================================================
# 8. DICCIONARIO PROPIO
# ==============================================================================
print("\n8. DICCIONARIO PROPIO")

def meta_de(col):
    """Devuelve (meta, nombre_col_original) segun el prefijo de la variable."""
    if col.startswith("equip_item_"):
        return meta_eq, f"item={int(col.replace('equip_item_',''))}"
    if col.startswith("jefe_"):
        orig = {"jefe_sexo":"s01a_02","jefe_edad":"s01a_03","jefe_indigena":"s01a_09",
                "jefe_estado_civil":"s01a_10","jefe_nivel_ed":"niv_ed","jefe_anios_estudio":"aestudio",
                "jefe_condact":"condact","jefe_tipo_ocupacion":"s04b_12","jefe_grupo_ocup":"cob_op"}.get(col)
        return (meta_per, orig) if orig else (None, None)
    if col in ("tamano_hogar","num_menores_14","num_adultos_mayores","num_ocupados",
               "anios_estudio_prom_15mas","prop_menores_14","prop_adultos_mayores",
               "tasa_dependencia_ocupados","tipohogar"):
        return meta_per, {"tamano_hogar":"nro","tipohogar":"tipohogar"}.get(col, col)
    if col in ("puntaje_inseguridad_alim","nivel_inseguridad_alim"):
        return meta_seg, "s07a_01"
    if col in ("personas_por_dormitorio","hacinamiento","num_dormitorios",
               "jefe_estudio_rango","tamano_hogar_rango","total_activos_hogar"):
        return None, None
    return meta_viv, col

def etiqueta(col):
    meta, orig = meta_de(col)
    if meta is None or orig is None:
        return ""
    if orig.startswith("item="):
        n = int(orig.split("=")[1])
        return etiquetas_items.get(float(n), etiquetas_items.get(n, f"Item {n}"))
    return str(meta.column_names_to_labels.get(orig, ""))

def codigos(col):
    meta, orig = meta_de(col)
    if meta is None or orig is None or orig.startswith("item="):
        return ""
    ls = meta.variable_to_label.get(orig)
    if ls and ls in meta.value_labels:
        return "; ".join(f"{int(k) if float(k).is_integer() else k}={v}" for k, v in meta.value_labels[ls].items())
    return ""

def tipo_de(col):
    s = base[col].dropna()
    unicos = s.nunique()
    if unicos <= 2:
        return "binaria"
    if unicos <= 15:
        return "categórica"
    return "numérica"

NOMBRES = {
    "folio": ("ID del hogar", "Identificador unico del hogar (UPM-segmento-persona)", "Vivienda/Persona", "texto", "folio = ID; UPM = dos primeros bloques"),
    "factor": ("Factor de expansión", "Peso de expansion muestral del INE", "Vivienda", "numérica", "Solo estadisticas ponderadas; NUNCA predictor"),
    "p0": ("Pobreza por ingreso", "1=Pobre, 0=No pobre (la calcula el INE)", "Persona", "binaria", "Etiqueta principal"),
    "pext0": ("Pobreza extrema", "1=Pobre extremo, 0=No (deriva del mismo ingreso)", "Persona", "binaria", "Etiqueta secundaria; NO predictor (fuga)"),
    "num_dormitorios": ("Dormitorios", "Cuartos usados exclusivamente para dormir (s06a_17; 0 imputado a 1)", "Vivienda", "numérica discreta", "0 del original = sin cuarto exclusivo -> 1"),
    "personas_por_dormitorio": ("Personas por dormitorio", "tamano_hogar / num_dormitorios", "Ingeniería", "numérica", "Hacinamiento si > 2.5"),
    "hacinamiento": ("Hacinamiento", "1 si personas_por_dormitorio > 2.5", "Ingeniería", "binaria", "Derivada"),
    "tamano_hogar_rango": ("Rango tamaño hogar", "0=Bajo(1-2); 1=Medio(3-4); 2=Alto(5+)", "Ingeniería", "ordinal", "Derivada"),
    "jefe_estudio_rango": ("Rango estudio jefe", "0=Bajo(0-6); 1=Medio(7-12); 2=Alto(>12) años", "Ingeniería", "ordinal", "Derivada"),
    "total_activos_hogar": ("Total activos hogar", "Suma de los 18 bienes durables del hogar", "Equipamiento", "numérica discreta", "Derivada"),
    "puntaje_inseguridad_alim": ("Puntaje FIES", "Suma de respuestas afirmativas (0-8) a las 8 preguntas FIES", "Seg. Alimentaria", "ordinal", "NS/NR cuenta como No"),
    "nivel_inseguridad_alim": ("Nivel inseguridad alim.", "0=Segura;1=Leve;2=Moderada;3=Severa (FIES)", "Seg. Alimentaria", "ordinal", "Derivada del puntaje"),
    "tamano_hogar": ("Tamaño del hogar", "Número de personas del hogar", "Persona", "numérica discreta", ""),
    "num_menores_14": ("Menores de 14 años", "Miembros con edad < 14", "Persona", "numérica discreta", ""),
    "prop_menores_14": ("Proporción menores 14", "num_menores_14 / tamano_hogar", "Ingeniería", "numérica", "Derivada"),
    "num_adultos_mayores": ("Adultos mayores", "Miembros con edad >= 60", "Persona", "numérica discreta", ""),
    "prop_adultos_mayores": ("Proporción adultos mayores", "num_adultos_mayores / tamano_hogar", "Ingeniería", "numérica", "Derivada"),
    "num_ocupados": ("Miembros ocupados", "Miembros con condact==1 (trabajó semana pasada)", "Persona", "numérica discreta", ""),
    "tasa_dependencia_ocupados": ("Dependientes por ocupado", "(tamano-ocupados)/ocupados; si 0 ocupados = tamano_hogar", "Ingeniería", "numérica", "Derivada"),
    "anios_estudio_prom_15mas": ("Años estudio prom. 15+", "Promedio de aestudio de miembros >=15 años", "Persona", "numérica", "Excluye menores para no sesgar"),
    "tipohogar": ("Tipo de hogar", "Tipología de hogar del INE (unipersonal, nuclear, extendido...)", "Persona", "categórica", ""),
    "jefe_edad": ("Edad del jefe/a", "Edad en años del jefe/a de hogar", "Persona (Jefe)", "numérica", ""),
    "jefe_condact": ("Condición actividad jefe", "condact del jefe: 0=pensionado?;1=ocupado;2=cesante;3=aspirante;4=temporal;5=permanente", "Persona (Jefe)", "categórica", "Ver decisiones_limpieza"),
    "jefe_tipo_ocupacion": ("Tipo ocupación jefe", "s04b_12 del jefe (1-8); 0=No aplica (sin ocupación)", "Persona (Jefe)", "categórica", "0 = no aplica, NO faltante"),
    "jefe_grupo_ocup": ("Grupo ocupacional jefe", "cob_op del jefe (0-9); 0=No aplica (sin ocupación)", "Persona (Jefe)", "categórica", "0 = no aplica, NO faltante"),
    "jefe_nivel_ed": ("Nivel educativo jefe", "niv_ed del jefe (0=Ninguno ... 6=Otros)", "Persona (Jefe)", "ordinal", ""),
    "jefe_anios_estudio": ("Años estudio jefe", "aestudio del jefe (0-23); NaN->0", "Persona (Jefe)", "numérica discreta", ""),
}

GRUPOS_RED = {
    "num_menores_14": "G1", "prop_menores_14": "G1",
    "num_adultos_mayores": "G4", "prop_adultos_mayores": "G4",
    "jefe_nivel_ed": "G2", "jefe_anios_estudio": "G2", "anios_estudio_prom_15mas": "G2",
    "jefe_estudio_rango": "G2",
    "puntaje_inseguridad_alim": "G3", "nivel_inseguridad_alim": "G3",
    "tamano_hogar": "G5", "tamano_hogar_rango": "G5",
    "num_dormitorios": "G6", "personas_por_dormitorio": "G6", "hacinamiento": "G6",
}

filas = []
for col in base.columns:
    if col in NOMBRES:
        leg, desc, tabla, tipo, cod = NOMBRES[col]
        rol = {"folio": "ID", "factor": "peso", "p0": "etiqueta", "pext0": "etiqueta"}.get(col, "predictor")
        filas.append((col, leg, desc, tabla, tipo, cod, rol, GRUPOS_RED.get(col, "Ninguno")))
    else:
        leg = etiqueta(col)[:80]
        filas.append((col, leg, etiqueta(col), "", tipo_de(col), codigos(col)[:200], "predictor", GRUPOS_RED.get(col, "Ninguno")))

dic = pd.DataFrame(filas, columns=[
    "nombre_original", "nombre_legible", "descripcion", "tabla_origen",
    "tipo_dato", "significado_codigos", "rol", "grupo_redundancia"
])
dic.to_csv(os.path.join(PROC, "diccionario_propio.csv"), index=False, encoding="utf-8-sig")
assert set(dic["nombre_original"]) == set(base.columns)
print(f"  diccionario_propio.csv: {len(dic)} variables")

print("\n" + "=" * 70)
print("PIPELINE COMPLETO")
print("=" * 70)
