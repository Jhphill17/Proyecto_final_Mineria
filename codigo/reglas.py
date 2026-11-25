# -*- coding: utf-8 -*-
"""
Reglas de asociacion - Proyecto final Mineria de Datos (pobreza por ingreso, EH 2025)
Minado manual (sin mlxtend): soporte, confianza y lift para itemsets de 1 a 3 items.
Enfoque A del plan: se comparan reglas con consecuente HOGAR_POBRE y HOGAR_NO_POBRE.

IMPORTANTE: el minado es SIN PONDERAR (sobre la muestra de hogares encuestados).

Uso: python codigo/reglas.py   (desde la raiz del proyecto)
Salida: tablas/reglas_asociacion.csv
"""
import os
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import pyreadstat

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRUDOS = os.path.join(RAIZ, "datos", "crudos")
if not os.path.isdir(CRUDOS):
    CRUDOS = RAIZ
TABLAS = os.path.join(RAIZ, "tablas")
os.makedirs(TABLAS, exist_ok=True)

# ==============================================================================
# 1. ITEMS: bienes del hogar (equipamiento) + servicios/vivienda + etiqueta
# ==============================================================================
print("1. CONSTRUCCION DE ITEMS")

df = pd.read_csv(os.path.join(RAIZ, "datos", "procesados", "base_hogares_completa.csv"),
                 dtype={"folio": str})
N = len(df)

# Nombres legibles de los items de equipamiento (del .sav)
_, meta_eq = pyreadstat.read_sav(os.path.join(CRUDOS, "EH2025_Equipamiento.sav"))
etiquetas_items = meta_eq.value_labels.get(meta_eq.variable_to_label.get("item"), {})

def nombre_item_eq(i):
    lab = etiquetas_items.get(float(i), etiquetas_items.get(i, f"Item {i}"))
    # "5. Refrigerador/freezer...?" -> "Refrigerador/freezer"
    import re
    t = re.sub(r"^\d+\.\s*", "", str(lab))
    return t.rstrip("?").strip()

items = {}  # nombre_legible -> Series booleana

# Equipamiento: los 18 items
for i in range(1, 19):
    col = f"equip_item_{i}"
    items[f"Posee: {nombre_item_eq(i)}"] = df[col].eq(1)

# Vivienda / servicios (dummy explicitas)
items["Vivienda tipo casa"] = df["s06a_01"].eq(1)
items["Vivienda propia"] = df["s06a_02"].isin([1, 2])
items["Agua por cañería de red"] = df["s06a_07"].isin([1, 2])
items["Baño con descarga de agua"] = df["s06a_09"].eq(1)
items["Energía eléctrica"] = df["s06a_12"].eq(1)
items["Cuarto exclusivo para cocinar"] = df["s06a_14"].eq(1)
items["Tiene internet"] = df["s06a_19"].eq(1)
items["Hacinamiento"] = df["hacinamiento"].eq(1)
items["Jefe con estudios superiores"] = df["jefe_estudio_rango"].eq(2)

# Ausencias: la pobreza se asocia con NO tener bienes/servicios.
# Se generan versiones "No/Sin ..." de cada item; el filtro de soporte decide cuales entran.
AUSENCIA = {
    "Vivienda tipo casa": "Vivienda no tipo casa",
    "Vivienda propia": "Vivienda no propia",
    "Agua por cañería de red": "Sin agua por cañería de red",
    "Baño con descarga de agua": "Sin baño con descarga de agua",
    "Energía eléctrica": "Sin energía eléctrica",
    "Cuarto exclusivo para cocinar": "Sin cuarto para cocinar",
    "Tiene internet": "Sin internet",
    "Hacinamiento": "Sin hacinamiento",
    "Jefe con estudios superiores": "Jefe sin estudios superiores",
}
items_presencia = dict(items)
for nombre, serie in items_presencia.items():
    if nombre.startswith("Posee: "):
        aus = "No posee: " + nombre[len("Posee: "):]
    else:
        aus = AUSENCIA.get(nombre, "Sin " + nombre.lower())
    items[aus] = ~serie

# ==============================================================================
# 2. FILTRAR ITEMS POR SOPORTE (descartar casi universales y muy raros)
# ==============================================================================
print("\n2. SOPORTE DE ITEMS")
MIN_SUP, MAX_SUP = 0.05, 0.90
soportes_items = {k: float(v.mean()) for k, v in items.items()}
items_ok = {k: v for k, v in items.items() if MIN_SUP <= soportes_items[k] <= MAX_SUP}
excluidos = {k: soportes_items[k] for k in items if k not in items_ok}
print(f"  Items usados: {len(items_ok)} | excluidos por soporte: {len(excluidos)}")
for k, s in sorted(excluidos.items(), key=lambda x: -x[1]):
    print(f"    excluido ({s:.1%}): {k}")

# Matriz booleana (hogares x items)
X = pd.DataFrame(items_ok)  # bool
n_items = X.shape[1]
print(f"  Matriz: {X.shape}")

# Soporte de la etiqueta
p_pobre = float(df["p0"].mean())
p_nopobre = 1 - p_pobre
print(f"  P(HOGAR_POBRE)={p_pobre:.3f} | P(HOGAR_NO_POBRE)={p_nopobre:.3f}")

# ==============================================================================
# 3. MINADO MANUAL: itemsets frecuentes de tamano 1 a 3 (sin ponderar)
# ==============================================================================
print("\n3. MINADO DE ITEMSETS FRECUENTES (max_len=3, min_support=5%)")
nombres = list(X.columns)
# Transacciones como tuplas de indices de items
trans = [tuple(np.flatnonzero(row)) for row in X.values]

contador = Counter()
for t in trans:
    for k in (1, 2, 3):
        for combo in itertools.combinations(t, k):
            contador[combo] += 1

min_count = MIN_SUP * N
freq = {c: n for c, n in contador.items() if n >= min_count}
print(f"  Itemsets frecuentes: {len(freq)}")

# Soporte de cada itemset frecuente
def soporte_itemset(combo):
    return freq.get(tuple(sorted(combo)), 0) / N

# ==============================================================================
# 4. REGLAS: antecedente (1-3 items) -> HOGAR_POBRE / HOGAR_NO_POBRE
# ==============================================================================
print("\n4. GENERACION DE REGLAS")
# Mascara de hogares pobres / no pobres como sets de indices de fila
idx_pobre = set(df.index[df["p0"] == 1])
idx_nopobre = set(df.index[df["p0"] == 0])

# Para cada itemset frecuente, contamos cuantos hogares del itemset son pobres/no pobres.
# Recorremos transacciones una vez mas, acumulando por itemset.
# Optimizado: para cada itemset frecuente guardamos (n_pobre, n_nopobre).
res = {c: [0, 0] for c in freq}  # [n_pobre, n_nopobre]
for i, t in enumerate(trans):
    es_pobre = i in idx_pobre
    for k in (1, 2, 3):
        for combo in itertools.combinations(t, k):
            if combo in res:
                res[combo][0 if es_pobre else 1] += 1

reglas = []
for combo, (n_pob, n_nopob) in res.items():
    n_ant = n_pob + n_nopob
    if n_ant == 0:
        continue
    sup_ant = n_ant / N
    ant_names = tuple(sorted(nombres[i] for i in combo))
    for cons, n_cons, p_cons, es_pobre in [
        ("HOGAR_POBRE", n_pob, p_pobre, True),
        ("HOGAR_NO_POBRE", n_nopob, p_nopobre, False),
    ]:
        if n_cons == 0:
            continue
        soporte = n_cons / N
        confianza = n_cons / n_ant
        lift = soporte / (sup_ant * p_cons)
        if confianza >= 0.55 and lift > 1.2:
            # P(antecedente | consequente): apoyo del antecedente dentro del grupo de la etiqueta
            sup_en_grupo = n_cons / (N * p_cons)
            reglas.append({
                "antecedente": " + ".join(ant_names),
                "consecuente": cons,
                "soporte": round(soporte, 4),
                "confianza": round(confianza, 4),
                "lift": round(lift, 3),
                "n_hogares": int(n_cons),
                "soporte_en_grupo": round(sup_en_grupo, 4),
                "tam_antecedente": len(combo),
            })

rdf = pd.DataFrame(reglas)
print(f"  Reglas candidatas: {len(rdf)}")

# ==============================================================================
# 5. SELECCION: quitar redundancias (si A->C y A,B->C dicen lo mismo, dejar la mas simple)
# ==============================================================================
print("\n5. SELECCION DE REGLAS NO REDUNDANTES")
seleccion = []
for cons in ["HOGAR_POBRE", "HOGAR_NO_POBRE"]:
    sub = rdf[rdf["consecuente"] == cons].sort_values(
        ["tam_antecedente", "lift"], ascending=[True, False])
    kept_ants = []
    for _, r in sub.iterrows():
        ant_set = set(r["antecedente"].split(" + "))
        # redundante si algun antecedente ya elegido es subconjunto propio
        redundante = any(k < ant_set for k in kept_ants)
        if not redundante:
            kept_ants.append(ant_set)
            seleccion.append(r)

sel = pd.DataFrame(seleccion).sort_values(["consecuente", "lift"], ascending=[True, False])
# Top 25 por consecuente
sel = (sel.groupby("consecuente", group_keys=False)
         .head(25)
         .drop(columns=["tam_antecedente"])
         .reset_index(drop=True))

# Renombrar columnas finales según acuerdo con Luciana/Juan
# soporte_condicional = P(antecedente | consequente): apoyo del antecedente dentro del grupo
sel = sel.rename(columns={"soporte_en_grupo": "soporte_condicional"})
sel.to_csv(os.path.join(TABLAS, "reglas_asociacion.csv"), index=False, encoding="utf-8-sig")
print(f"  reglas_asociacion.csv: {len(sel)} reglas")
print(sel.head(15).to_string(index=False))
print("\nREGLAS COMPLETAS")
