# -*- coding: utf-8 -*-
"""
EDA - Proyecto final Mineria de Datos (pobreza por ingreso, EH 2025 INE)
Genera tablas/ y figuras/ a partir de datos/procesados/base_hogares_completa.csv
Todas las tasas se calculan PONDERADAS con el factor de expansion.
Uso: python codigo/eda.py   (desde la raiz del proyecto)
"""
import os
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(RAIZ, "datos", "procesados")
TABLAS = os.path.join(RAIZ, "tablas")
FIGURAS = os.path.join(RAIZ, "figuras")
os.makedirs(TABLAS, exist_ok=True)
os.makedirs(FIGURAS, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTOS = {1:"Chuquisaca",2:"La Paz",3:"Cochabamba",4:"Oruro",5:"Potosí",6:"Tarija",7:"Santa Cruz",8:"Beni",9:"Pando"}
AREAS = {1:"Urbano",2:"Rural"}

df = pd.read_csv(os.path.join(PROC, "base_hogares_completa.csv"), dtype={"folio": str})
df["pw"] = df["p0"] * df["factor"]

def tasa_por(col, nombre=None):
    """Tasa de pobreza ponderada por categoria de `col`."""
    d = (df.groupby(col, observed=True)
           .agg(hogares_muestra=("folio", "size"),
                hogares_expandidos=("factor", "sum"),
                suma_pw=("pw", "sum"))
           .reset_index())
    d["tasa_pobreza_pond"] = d["suma_pw"] / d["hogares_expandidos"]
    d = d.drop(columns="suma_pw")
    if nombre:
        d = d.rename(columns={col: nombre})
    return d

# ==============================================================================
# 1. DESCRIPTIVAS
# ==============================================================================
print("1. DESCRIPTIVAS")
num_cols = ["tamano_hogar","num_menores_14","num_adultos_mayores","num_ocupados",
            "jefe_edad","jefe_anios_estudio","anios_estudio_prom_15mas",
            "personas_por_dormitorio","num_dormitorios","total_activos_hogar",
            "puntaje_inseguridad_alim"]
desc = df[num_cols].describe(percentiles=[.25,.5,.75]).T[["count","mean","std","min","25%","50%","75%","max"]]
desc.columns = ["n","media","desv_est","min","p25","mediana","p75","max"]
desc["media_pond"] = [np.average(df[c], weights=df["factor"]) for c in desc.index]
desc.to_csv(os.path.join(TABLAS, "eda_descriptivas.csv"), encoding="utf-8-sig")

etiquetas = pd.DataFrame({
    "variable": ["p0","pext0"],
    "tasa_sin_ponderar": [df.p0.mean(), df.pext0.mean()],
    "tasa_ponderada": [np.average(df.p0, weights=df.factor), np.average(df.pext0, weights=df.factor)],
})
etiquetas.to_csv(os.path.join(TABLAS, "eda_etiquetas.csv"), index=False)
print(etiquetas.to_string(index=False))

# ==============================================================================
# 2. POBREZA POR TERRITORIO
# ==============================================================================
print("\n2. POBREZA POR TERRITORIO")
t_da = tasa_por(["depto","area"])
t_da["depto_nombre"] = t_da["depto"].map(DEPTOS)
t_da["area_nombre"] = t_da["area"].map(AREAS)
t_da.to_csv(os.path.join(TABLAS, "eda_pobreza_depto_area.csv"), index=False)

t_dep = tasa_por("depto"); t_dep["depto_nombre"] = t_dep["depto"].map(DEPTOS)
t_dep = t_dep.sort_values("tasa_pobreza_pond", ascending=False)
t_dep.to_csv(os.path.join(TABLAS, "eda_pobreza_depto.csv"), index=False)

t_area = tasa_por("area"); t_area["area_nombre"] = t_area["area"].map(AREAS)
t_area.to_csv(os.path.join(TABLAS, "eda_pobreza_area.csv"), index=False)
print(t_dep.to_string(index=False))
print(t_area.to_string(index=False))

# ==============================================================================
# 3. RELACIONES CON LA POBREZA
# ==============================================================================
print("\n3. RELACIONES CON LA POBREZA")

# Educacion del jefe (rango y años)
t_edu = tasa_por("jefe_estudio_rango")
t_edu["rango"] = t_edu["jefe_estudio_rango"].map({0:"Bajo (0-6)",1:"Medio (7-12)",2:"Alto (>12)"})
t_edu.to_csv(os.path.join(TABLAS, "eda_pobreza_por_educacion_jefe.csv"), index=False)

df["jefe_anios_estudio_r"] = pd.cut(df["jefe_anios_estudio"], bins=[-1,0,3,6,9,12,15,30],
                                  labels=["0","1-3","4-6","7-9","10-12","13-15","16+"])
t_edu2 = tasa_por("jefe_anios_estudio_r")
t_edu2.to_csv(os.path.join(TABLAS, "eda_pobreza_por_anios_estudio_jefe.csv"), index=False)

# Equipamiento: total de activos en rangos + items clave
df["activos_rango"] = pd.cut(df["total_activos_hogar"], bins=[-1,4,8,12,18],
                             labels=["0-4","5-8","9-12","13-18"])
t_act = tasa_por("activos_rango")
t_act.to_csv(os.path.join(TABLAS, "eda_pobreza_por_activos.csv"), index=False)

items_clave = ["equip_item_5","equip_item_13","equip_item_14","equip_item_15","equip_item_17","equip_item_18","equip_item_7"]
filas_items = []
for it in items_clave:
    r = tasa_por(it)
    r.insert(0, "item", it)
    filas_items.append(r)
t_items = pd.concat(filas_items, ignore_index=True)
t_items.to_csv(os.path.join(TABLAS, "eda_pobreza_por_equipamiento.csv"), index=False)

# Inseguridad alimentaria
t_fies = tasa_por("nivel_inseguridad_alim")
t_fies["nivel"] = t_fies["nivel_inseguridad_alim"].map({0:"Segura",1:"Leve",2:"Moderada",3:"Severa"})
t_fies.to_csv(os.path.join(TABLAS, "eda_pobreza_por_inseguridad_alim.csv"), index=False)

# Vivienda: variables s06a principales
for v, nom in [("s06a_01","tipo_vivienda"),("s06a_02","tenencia_vivienda"),("s06a_07","procedencia_agua"),
               ("s06a_09","tipo_bano"),("s06a_12","energia_electrica"),("s06a_19","internet")]:
    t = tasa_por(v)
    t.to_csv(os.path.join(TABLAS, f"eda_pobreza_por_{nom}.csv"), index=False)

# Hacinamiento y tamano de hogar
t_hac = tasa_por("hacinamiento")
t_hac.to_csv(os.path.join(TABLAS, "eda_pobreza_por_hacinamiento.csv"), index=False)
t_tam = tasa_por("tamano_hogar_rango")
t_tam.to_csv(os.path.join(TABLAS, "eda_pobreza_por_tamano_hogar.csv"), index=False)

print("  Tablas de relaciones exportadas.")

# ==============================================================================
# 4. CORRELACIONES (Spearman, robusta a atipicos)
# ==============================================================================
print("\n4. CORRELACIONES")
num_all = [c for c in df.columns if df[c].dtype.kind in "if" and c not in ["factor","p0","pext0","pw","jefe_anios_estudio_r"]]
corr = df[num_all].corr(method="spearman")
corr.to_csv(os.path.join(TABLAS, "eda_correlaciones.csv"), encoding="utf-8-sig")
pares = []
for i, a in enumerate(num_all):
    for b in num_all[i+1:]:
        v = corr.loc[a, b]
        if abs(v) > 0.5:
            pares.append((a, b, round(v, 3)))
pares.sort(key=lambda x: -abs(x[2]))
print("  Pares |rho|>0.5:")
for p in pares[:20]:
    print("   ", p)

# ==============================================================================
# 5. FIGURAS PNG (200 dpi)
# ==============================================================================
print("\n5. FIGURAS")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})

def fig_barras_horizontal(t, xcol, ycol, titulo, xlabel, fname, color="#4C78A8"):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35*len(t))))
    ax.barh(t[xcol].astype(str), t[ycol]*100, color=color)
    ax.set_xlabel(xlabel); ax.set_title(titulo)
    for i, (v, n) in enumerate(zip(t[ycol]*100, t["hogares_muestra"])):
        ax.text(v+0.3, i, f"{v:.1f}% (n={int(n)})", va="center", fontsize=8)
    ax.set_xlim(0, max(t[ycol]*100)*1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS, fname), dpi=200)
    plt.close(fig)

fig_barras_horizontal(t_dep, "depto_nombre", "tasa_pobreza_pond",
    "Pobreza por ingreso ponderada por departamento (EH 2025)",
    "Pobreza (% de hogares, ponderado)", "eda_01_pobreza_depto.png", "#4C78A8")

fig, ax = plt.subplots(figsize=(6,4))
t_area_p = t_area.set_index("area_nombre")["tasa_pobreza_pond"]*100
ax.bar(t_area_p.index, t_area_p.values, color=["#4C78A8","#F58518"])
for i,(k,v) in enumerate(t_area_p.items()):
    n = int(t_area[t_area.area_nombre==k]["hogares_muestra"].iloc[0])
    ax.text(i, v+0.8, f"{v:.1f}%\nn={n}", ha="center", fontsize=9)
ax.set_ylabel("Pobreza (% hogares, ponderado)"); ax.set_title("Pobreza por área (EH 2025)")
fig.tight_layout(); fig.savefig(os.path.join(FIGURAS,"eda_02_pobreza_area.png"), dpi=200); plt.close(fig)

fig_barras_horizontal(t_edu, "rango", "tasa_pobreza_pond",
    "Pobreza según nivel de estudio del jefe/a",
    "Pobreza (% de hogares, ponderado)", "eda_03_pobreza_educacion_jefe.png", "#54A24B")

fig_barras_horizontal(t_fies, "nivel", "tasa_pobreza_pond",
    "Pobreza según inseguridad alimentaria (FIES)",
    "Pobreza (% de hogares, ponderado)", "eda_04_pobreza_inseguridad_alim.png", "#E45756")

fig_barras_horizontal(t_act, "activos_rango", "tasa_pobreza_pond",
    "Pobreza según número de bienes durables del hogar",
    "Pobreza (% de hogares, ponderado)", "eda_05_pobreza_activos.png", "#72B7B2")

# Heatmap depto x area
piv = t_da.pivot(index="depto_nombre", columns="area_nombre", values="tasa_pobreza_pond")*100
fig, ax = plt.subplots(figsize=(7,5))
im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.values[i,j]:.1f}", ha="center", va="center", fontsize=8)
ax.set_title("Pobreza ponderada (%) por departamento y área")
fig.colorbar(im, label="Pobreza %"); fig.tight_layout()
fig.savefig(os.path.join(FIGURAS,"eda_06_pobreza_depto_area.png"), dpi=200); plt.close(fig)

print("  Figuras exportadas a figuras/")
print("\nEDA COMPLETO")
