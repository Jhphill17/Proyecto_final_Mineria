# -*- coding: utf-8 -*-
"""
Modelado de pobreza por ingreso - Proyecto final Mineria de Datos (EH 2025)
Implementa las tareas 3.1-3.4, 4.1 y 4.3 del plan de Juan.
Semilla 42, validacion agrupada por UPM, preprocesamiento dentro del Pipeline.

Uso: python codigo/modelado.py   (desde la raiz del proyecto)
Salidas: resultados/*.csv, resultados/figuras/*.png, parametros.json
"""
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (StratifiedGroupKFold, cross_validate,
                                      cross_val_predict, GridSearchCV, RandomizedSearchCV)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             precision_score, f1_score, balanced_accuracy_score,
                             brier_score_loss, precision_recall_curve, roc_curve,
                             confusion_matrix)
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)

SEED = 42
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(RAIZ, "datos", "procesados")
RES = os.path.join(RAIZ, "resultados")
FIG = os.path.join(RES, "figuras")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ==============================================================================
# 1. DATOS
# ==============================================================================
log("1. CARGA DE DATOS")
X = pd.read_csv(os.path.join(PROC, "X_predictores.csv"), dtype={"folio": str})
M = pd.read_csv(os.path.join(PROC, "metadatos_y_etiquetas.csv"), dtype={"folio": str})
assert X["folio"].is_unique and (X["folio"].values == M["folio"].values).all()
df = X.merge(M[["folio", "factor", "p0", "pext0"]], on="folio", validate="one_to_one")

FEATURES = [c for c in X.columns if c != "folio"]
assert not {"p0", "pext0", "factor", "folio"} & set(FEATURES)
log(f"  {len(df)} hogares x {len(FEATURES)} predictores")

# Tipos desde el diccionario
dic = pd.read_csv(os.path.join(PROC, "diccionario_propio.csv"))
trat = dict(zip(dic.nombre_original, dic.tratamiento_recomendado if "tratamiento_recomendado" in dic else dic.tipo_dato))
# diccionario_propio no tiene tratamiento_recomendado; derivar de tipo_dato
NOMINALES = [c for c in FEATURES if dic.set_index("nombre_original").loc[c, "tipo_dato"] in
             ("binaria", "categórica", "ordinal")]
NUMERICAS = [c for c in FEATURES if c not in NOMINALES]
# NOTA: no se convierten a texto; OneHotEncoder trata cada valor numerico como categoria
# y asi depto/area quedan numericos para exportar y para los subgrupos.
log(f"  nominales: {len(NOMINALES)} | numericas: {len(NUMERICAS)}")

# UPM y particion 80/20
df["upm"] = df["folio"].str.split("-").str[:2].str.join("-")
y = df["p0"].astype(int)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr, te = next(sgkf.split(df, y, groups=df["upm"]))
assert set(df["upm"].iloc[tr]).isdisjoint(df["upm"].iloc[te])
Xd = df[FEATURES]
w = df["factor"]
log(f"  train {len(tr)} | test {len(te)} | UPMs train {df['upm'].iloc[tr].nunique()} test {df['upm'].iloc[te].nunique()}")

# ==============================================================================
# 2. PREPROCESAMIENTO Y METRICAS
# ==============================================================================
def prep(escalar):
    num = StandardScaler() if escalar else "passthrough"
    return ColumnTransformer(
        [("nom", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                               sparse_output=False), NOMINALES),
         ("num", num, NUMERICAS)], sparse_threshold=0)

SCORING = {"auc": "roc_auc", "auc_pr": "average_precision", "recall": "recall",
           "precision": "precision", "f1": "f1", "bal_acc": "balanced_accuracy"}

def cv_report(model, Xd_, yd_, groups, k=5):
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    r = cross_validate(model, Xd_, yd_, groups=groups, cv=cv, scoring=SCORING)
    return pd.Series({m: r[f"test_{m}"].mean() for m in SCORING})

def test_report(model, Xtr_, ytr_, Xte_, yte_, wte_, thr=0.5):
    model.fit(Xtr_, ytr_)
    p = model.predict_proba(Xte_)[:, 1]
    yh = (p >= thr).astype(int)
    return pd.Series({"auc": roc_auc_score(yte_, p),
                      "auc_pond": roc_auc_score(yte_, p, sample_weight=wte_),
                      "auc_pr": average_precision_score(yte_, p),
                      "recall": recall_score(yte_, yh), "precision": precision_score(yte_, yh),
                      "f1": f1_score(yte_, yh), "bal_acc": balanced_accuracy_score(yte_, yh),
                      "brier": brier_score_loss(yte_, p)})

# ==============================================================================
# 3. MODELOS: busqueda de hiperparametros (3 pliegues agrupados, dentro de train)
# ==============================================================================
log("2. BUSQUEDA DE HIPERPARAMETROS")
cv3 = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
Xtr, ytr, gtr = Xd.iloc[tr], y.iloc[tr], df["upm"].iloc[tr]

resultados = {}   # nombre -> (pipeline, cv_score)

def buscar(nombre, pipe, espacio, n_iter=None):
    for cw in [None, "balanced"]:
        p = pipe
        p.set_params(clf__class_weight=cw)
        if n_iter:
            gs = RandomizedSearchCV(p, espacio, n_iter=n_iter, scoring="roc_auc",
                                    cv=cv3, random_state=SEED, n_jobs=-1)
        else:
            gs = GridSearchCV(p, espacio, scoring="roc_auc", cv=cv3, n_jobs=-1)
        gs.fit(Xtr, ytr, groups=gtr)
        tag = f"{nombre}_{'bal' if cw else 'sin'}"
        resultados[tag] = (gs.best_estimator_, gs.best_score_)
        log(f"  {tag}: best={gs.best_score_:.4f} params={gs.best_params_}")

# Logistica
logi = Pipeline([("prep", prep(True)),
                 ("clf", LogisticRegression(penalty="l2", max_iter=2000, random_state=SEED))])
buscar("logistica", logi, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0]})

# Arbol
arbol = Pipeline([("prep", prep(False)),
                  ("clf", DecisionTreeClassifier(random_state=SEED))])
buscar("arbol", arbol, {"clf__max_depth": [3, 4, 5, 6, 8, 10],
                        "clf__min_samples_leaf": [10, 30, 60]})

# Random Forest
rf = Pipeline([("prep", prep(False)),
               ("clf", RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED))])
buscar("rf", rf, {"clf__max_depth": [6, 8, 12, None], "clf__min_samples_leaf": [5, 10, 20],
                  "clf__max_features": ["sqrt", 0.3, 0.5]}, n_iter=6)

# Gradient Boosting
gb = Pipeline([("prep", prep(False)),
               ("clf", HistGradientBoostingClassifier(early_stopping=True, random_state=SEED))])
buscar("gb", gb, {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_leaf_nodes": [8, 15, 31],
                  "clf__min_samples_leaf": [20, 50, 100], "clf__l2_regularization": [0, 1, 5]},
      n_iter=8)

# ==============================================================================
# 4. COMPARACION: CV 5 pliegues + prueba 20%
# ==============================================================================
log("3. COMPARACION DE MODELOS")
filas = []
for tag, (modelo, _) in resultados.items():
    cv = cv_report(modelo, Xd, y, df["upm"], k=5)
    te_r = test_report(modelo, Xtr, ytr, Xd.iloc[te], y.iloc[te], w.iloc[te], thr=0.5)
    fila = {"modelo": tag, "cv_auc": cv["auc"], "cv_auc_pr": cv["auc_pr"],
            "cv_recall": cv["recall"], "cv_precision": cv["precision"],
            "cv_f1": cv["f1"], "cv_bal_acc": cv["bal_acc"],
            "test_auc": te_r["auc"], "test_auc_pond": te_r["auc_pond"],
            "test_auc_pr": te_r["auc_pr"], "test_recall": te_r["recall"],
            "test_precision": te_r["precision"], "test_f1": te_r["f1"],
            "test_bal_acc": te_r["bal_acc"], "test_brier": te_r["brier"]}
    filas.append(fila)
    log(f"  {tag}: CV AUC={cv['auc']:.3f} | test AUC={te_r['auc']:.3f}")

metricas = pd.DataFrame(filas).sort_values("cv_auc", ascending=False)
metricas.to_csv(os.path.join(RES, "metricas_modelos.csv"), index=False)
print(metricas[["modelo", "cv_auc", "test_auc", "test_recall", "test_precision", "test_f1"]].to_string(index=False))

# Eleccion: mayor AUC CV; desempate por AUC-PR CV
mejor_tag = metricas.iloc[0]["modelo"]
modelo = resultados[mejor_tag][0]
log(f"  MODELO ELEGIDO: {mejor_tag}")

# ==============================================================================
# 5. PREDICCIONES FUERA DE PLIEGUE, UMBRAL Y CALIBRACION
# ==============================================================================
log("4. OOF, UMBRAL Y CALIBRACION")
cvk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = cross_val_predict(modelo, Xd, y, groups=df["upm"], cv=cvk,
                        method="predict_proba", n_jobs=-1)[:, 1]
log(f"  AUC fuera de pliegue (todos): {roc_auc_score(y, oof):.4f}")

pr, rc, th = precision_recall_curve(y.iloc[tr], oof[tr])
f1s = 2 * pr * rc / (pr + rc + 1e-12)
UMBRAL = float(th[np.argmax(f1s[:-1])])
log(f"  Umbral F1-optimo (train): {UMBRAL:.3f}")

pred_pobre = (oof >= UMBRAL).astype(int)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y)
prob_cal = iso.predict(oof)
log(f"  media cruda={oof.mean():.3f} | calibrada={prob_cal.mean():.3f} | real={y.mean():.3f} | ponderada={np.average(y, weights=w):.3f}")
log(f"  umbral calibrado equivalente: {iso.predict([UMBRAL])[0]:.3f}")

# Metricas finales en prueba con umbral
p_te = modelo.predict_proba(Xd.iloc[te])[:, 1]
yh_te = (p_te >= UMBRAL).astype(int)
cm = confusion_matrix(y.iloc[te], yh_te)
log(f"  Test: TP={cm[1,1]} FN={cm[1,0]} FP={cm[0,1]} TN={cm[0,0]} | AUC={roc_auc_score(y.iloc[te], p_te):.3f}")

# Exportar hogares_resultados.csv
out = df[["folio", "depto", "area", "factor", "p0", "pext0"]].copy()
out["prob_pobreza"] = prob_cal
out["pred_pobre"] = pred_pobre
out.to_csv(os.path.join(RES, "hogares_resultados.csv"), index=False)
log("  hogares_resultados.csv exportado")

# Curva de calibracion por deciles (para figura 05)
df_cal = pd.DataFrame({"p": prob_cal, "y": y, "w": w})
df_cal["dec"] = pd.qcut(df_cal["p"], 10, duplicates="drop")
cal_tab = df_cal.groupby("dec", observed=True).apply(
    lambda g: pd.Series({"p_media": np.average(g["p"], weights=g["w"]),
                         "y_pond": np.average(g["y"], weights=g["w"]),
                         "n": len(g)}), include_groups=False).reset_index()
cal_tab.to_csv(os.path.join(RES, "calibracion_deciles.csv"), index=False)

# ==============================================================================
# 6. IMPORTANCIA, ODDS RATIOS, SENSIBILIDAD, SUBGRUPOS
# ==============================================================================
log("5. IMPORTANCIA Y SENSIBILIDAD")
pi = permutation_importance(modelo, Xd.iloc[te], y.iloc[te], scoring="roc_auc",
                            n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"variable": FEATURES, "caida_auc": pi.importances_mean,
                    "sd": pi.importances_std}).sort_values("caida_auc", ascending=False)
imp.to_csv(os.path.join(RES, "importancia_variables.csv"), index=False)
log("  top importancia: " + ", ".join(imp.head(5).variable))

# Odds ratios de la logistica (mejor C, sin pesos)
log_best = resultados[[k for k in resultados if k.startswith("logistica") and k.endswith("sin")][0]][0]
log_best.fit(Xtr, ytr)
nombres = log_best.named_steps["prep"].get_feature_names_out()
coef = log_best.named_steps["clf"].coef_[0]
pd.DataFrame({"termino": nombres, "coef": coef, "odds_ratio": np.exp(coef)}) \
  .sort_values("odds_ratio", ascending=False).to_csv(
      os.path.join(RES, "odds_ratios_logistica.csv"), index=False)

# Sensibilidad por grupos de variables
def quitar(cols):
    feats = [c for c in FEATURES if c not in cols]
    nom = [c for c in NOMINALES if c in feats]
    num = [c for c in NUMERICAS if c in feats]
    ct = ColumnTransformer([("nom", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                                  sparse_output=False), nom),
                            ("num", "passthrough", num)], sparse_threshold=0)
    return Pipeline([("prep", ct),
                     ("clf", HistGradientBoostingClassifier(
                         learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
                         l2_regularization=1, early_stopping=True, random_state=SEED,
                         class_weight="balanced"))])

GRUPOS = {
    "todos": [],
    "sin_inseguridad_alim": ["puntaje_inseguridad_alim", "nivel_inseguridad_alim"],
    "sin_territorio": ["depto", "area"],
    "sin_ambos": ["puntaje_inseguridad_alim", "nivel_inseguridad_alim", "depto", "area"],
    "sin_equipamiento": [c for c in FEATURES if c.startswith("equip_item_")],
}
sens = []
for gname, cols in GRUPOS.items():
    m = quitar(cols)
    cv = cv_report(m, Xd, y, df["upm"], k=5)
    sens.append({"grupo": gname, "n_vars": len(FEATURES) - len(cols), "cv_auc": cv["auc"]})
    log(f"  sensibilidad {gname}: AUC={cv['auc']:.3f}")
pd.DataFrame(sens).to_csv(os.path.join(RES, "sensibilidad_variables.csv"), index=False)

# Subgrupos en prueba
sub = []
for nombre, mascara in [("cochabamba", df["depto"] == 3), ("resto_pais", df["depto"] != 3),
                        ("urbano", df["area"] == 1), ("rural", df["area"] == 2)]:
    idx = np.where(mascara.values)[0]
    idx_te = [i for i in idx if i in set(te)]
    if len(idx_te) < 50 or len(set(y.iloc[idx_te])) < 2:
        continue
    p = modelo.predict_proba(Xd.iloc[idx_te])[:, 1]
    sub.append({"subgrupo": nombre, "n": len(idx_te),
                "pobreza_pond": np.average(y.iloc[idx_te], weights=w.iloc[idx_te]),
                "auc": roc_auc_score(y.iloc[idx_te], p)})
pd.DataFrame(sub).to_csv(os.path.join(RES, "desempeno_subgrupos.csv"), index=False)
print(pd.DataFrame(sub).to_string(index=False))

# ==============================================================================
# 7. CLUSTERING
# ==============================================================================
log("6. CLUSTERING")
VARS_CLUSTER = ["tamano_hogar", "prop_menores_14", "prop_adultos_mayores",
                "num_ocupados", "tasa_dependencia_ocupados", "jefe_edad",
                "jefe_anios_estudio", "anios_estudio_prom_15mas", "jefe_nivel_ed",
                "jefe_condact", "jefe_tipo_ocupacion", "jefe_grupo_ocup",
                "puntaje_inseguridad_alim", "nivel_inseguridad_alim", "hacinamiento",
                "num_dormitorios", "personas_por_dormitorio", "depto", "area",
                "tipohogar", "s06a_01", "s06a_02", "s06a_03", "s06a_05", "s06a_06",
                "s06a_07", "s06a_09", "s06a_12", "s06a_15", "s06a_19", "s06a_20"] + \
               [f"equip_item_{i}" for i in range(1, 19)]
VARS_CLUSTER = [c for c in VARS_CLUSTER if c in df.columns]
# nominales del cluster a texto->codigos? KMeans necesita numerico: usar codigos crudos
Zdf = df[VARS_CLUSTER].copy()
for c in VARS_CLUSTER:
    if Zdf[c].dtype == object:
        Zdf[c] = Zdf[c].astype("category").cat.codes
Zdf["personas_por_dormitorio"] = Zdf["personas_por_dormitorio"].clip(upper=6)
from sklearn.preprocessing import StandardScaler as SS
Z = SS().fit_transform(Zdf)
Zp = PCA(n_components=0.90, random_state=SEED).fit_transform(Z)
log(f"  {len(VARS_CLUSTER)} vars -> PCA {Zp.shape[1]} componentes (90% varianza)")

filas_k = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Zp)
    filas_k.append({"k": k, "inercia": km.inertia_,
                    "silueta": silhouette_score(Zp, km.labels_, sample_size=5000, random_state=SEED),
                    "calinski": calinski_harabasz_score(Zp, km.labels_),
                    "davies": davies_bouldin_score(Zp, km.labels_)})
    log(f"  k={k}: silueta={filas_k[-1]['silueta']:.3f} calinski={filas_k[-1]['calinski']:.0f}")
pd.DataFrame(filas_k).to_csv(os.path.join(RES, "seleccion_k_clustering.csv"), index=False)

K = 3
km3 = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Zp)
df["cluster"] = km3.labels_
ari_sem = adjusted_rand_score(km3.labels_,
    KMeans(K, n_init=20, random_state=7).fit(Zp).labels_)
rng = np.random.default_rng(SEED)
idx_sub = rng.choice(len(Zp), 3000, replace=False)
ari_ward = adjusted_rand_score(km3.labels_[idx_sub],
    AgglomerativeClustering(K, linkage="ward").fit_predict(Zp[idx_sub]))
log(f"  k={K} | ARI semillas={ari_sem:.2f} | ARI vs Ward={ari_ward:.2f}")

# Perfiles ponderados
pw = df["p0"] * df["factor"]
perf = df.groupby("cluster").agg(hogares_exp=("factor", "sum"), suma_pw=("pw" if "pw" in df else "factor", "sum"), n=("folio", "size"))
df["pw"] = pw
perf = df.groupby("cluster").agg(hogares_exp=("factor", "sum"), suma_pw=("pw", "sum"), n=("folio", "size"))
perf["pct_hogares_pond"] = perf["hogares_exp"] / perf["hogares_exp"].sum()
perf["pobreza_pond"] = perf["suma_pw"] / perf["hogares_exp"]
perf.to_csv(os.path.join(RES, "perfiles_clusters.csv"))
print(perf.to_string())

# Medias ponderadas por cluster para describir perfiles
medias = []
VARS_PERFIL_NUM = ["tamano_hogar", "prop_menores_14", "jefe_edad", "jefe_anios_estudio",
                   "anios_estudio_prom_15mas", "total_activos_hogar", "hacinamiento",
                   "num_dormitorios", "puntaje_inseguridad_alim", "tasa_dependencia_ocupados"]
VARS_PERFIL_CAT = ["jefe_condact", "jefe_tipo_ocupacion", "jefe_grupo_ocup", "tipohogar"]
for cl in range(K):
    sub = df[df["cluster"] == cl]
    for v in VARS_PERFIL_NUM:
        if v in sub.columns and pd.api.types.is_numeric_dtype(sub[v]):
            medias.append({"cluster": cl, "variable": v,
                           "media_pond": np.average(sub[v], weights=sub["factor"])})
    # proporciones ponderadas de depto y area
    for v in ["depto", "area"]:
        if v in sub.columns:
            for val in sorted(sub[v].dropna().unique()):
                m = (sub[v] == val).astype(float)
                medias.append({"cluster": cl, "variable": f"{v}={val}",
                               "media_pond": np.average(m, weights=sub["factor"])})
    # moda ponderada de variables categoricas clave
    for v in VARS_PERFIL_CAT:
        if v in sub.columns:
            vc = sub.groupby(v)["factor"].sum()
            top = vc.idxmax()
            medias.append({"cluster": cl, "variable": f"{v}_modal",
                           "media_pond": float(vc.max() / vc.sum()),
                           "nivel_modal": str(top)})
pd.DataFrame(medias).to_csv(os.path.join(RES, "perfiles_equipamiento.csv"), index=False)

# Actualizar hogares_resultados con cluster
out2 = pd.read_csv(os.path.join(RES, "hogares_resultados.csv"), dtype={"folio": str})
out2["cluster"] = df.set_index("folio").loc[out2["folio"], "cluster"].values
out2.to_csv(os.path.join(RES, "hogares_resultados.csv"), index=False)

# ==============================================================================
# 8. FIGURAS
# ==============================================================================
log("7. FIGURAS")
# 01 ROC/PR del modelo elegido en prueba
fpr, tpr, _ = roc_curve(y.iloc[te], p_te)
prec, rec, _ = precision_recall_curve(y.iloc[te], p_te)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y.iloc[te], p_te):.3f}")
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set_xlabel("Tasa de falsos positivos"); ax[0].set_ylabel("Sensibilidad")
ax[0].set_title("Curva ROC (prueba)"); ax[0].legend()
ax[1].plot(rec, prec, label=f"AP={average_precision_score(y.iloc[te], p_te):.3f}")
ax[1].axhline(y.iloc[te].mean(), ls="--", color="gray")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precisión")
ax[1].set_title("Curva Precision-Recall (prueba)"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "01_roc_pr.png")); plt.close(fig)

# 02 Importancia por permutacion
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.head(15)["variable"][::-1], imp.head(15)["caida_auc"][::-1])
ax.set_xlabel("Caída del AUC al permutar"); ax.set_title("Importancia de variables (permutación)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_importancia.png")); plt.close(fig)

# 03 Dependencia parcial (top 4 variables numericas)
top_vars = [v for v in imp.head(12)["variable"] if v in NUMERICAS][:4]
if top_vars:
    idxs = [FEATURES.index(v) for v in top_vars]
    fig, axes = plt.subplots(1, len(top_vars), figsize=(4 * len(top_vars), 3.5))
    if len(top_vars) == 1:
        axes = [axes]
    for ax, v, ix in zip(axes, top_vars, idxs):
        pdp = partial_dependence(modelo, Xd.iloc[te].astype(float), features=[ix], kind="average")
        curva = np.asarray(pdp["average"]).reshape(-1)
        ax.plot(pdp["grid_values"][0], curva)
        ax.set_title(v); ax.set_xlabel(v); ax.set_ylabel("Efecto parcial")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "03_dependencia_parcial.png")); plt.close(fig)

# 05 Calibracion
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.plot(cal_tab["p_media"], cal_tab["y_pond"], "o-")
ax.set_xlabel("Probabilidad media predicha (ponderada)"); ax.set_ylabel("Pobreza real (ponderada)")
ax.set_title("Calibración por deciles"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "05_calibracion.png")); plt.close(fig)

# 06 Seleccion de k
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(filas_k_df := pd.DataFrame(filas_k)["k"], pd.DataFrame(filas_k)["silueta"], "o-")
ax.set_xlabel("k"); ax.set_ylabel("Silueta"); ax.set_title("Selección de k (clustering)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "06_seleccion_k.png")); plt.close(fig)

# 08/09 Perfiles y clusters PCA
fig, ax = plt.subplots(figsize=(7, 4))
for cl in range(K):
    m = df["cluster"] == cl
    ax.scatter(Zp[m, 0], Zp[m, 1], s=3, alpha=0.4, label=f"Cluster {cl} ({perf.loc[cl,'pct_hogares_pond']*100:.0f}%)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Clusters en espacio PCA")
ax.legend(markerscale=5); fig.tight_layout()
fig.savefig(os.path.join(FIG, "09_clusters_pca.png")); plt.close(fig)

# 08 perfiles: pobreza por cluster
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(perf.index.astype(str), perf["pobreza_pond"] * 100)
ax.axhline(np.average(y, weights=w) * 100, ls="--", color="red", label="Promedio nacional")
ax.set_xlabel("Cluster"); ax.set_ylabel("Pobreza ponderada (%)"); ax.set_title("Pobreza por perfil de hogar")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "08_perfiles_pobreza.png")); plt.close(fig)

# ==============================================================================
# 9. PARAMETROS Y RESUMEN
# ==============================================================================
params = {
    "semilla": SEED, "n_hogares": int(len(df)), "n_predictores": len(FEATURES),
    "n_nominales": len(NOMINALES), "n_numericas": len(NUMERICAS),
    "umbral": UMBRAL, "modelo_elegido": mejor_tag,
    "auc_oof": float(roc_auc_score(y, oof)),
    "media_prob_cruda": float(oof.mean()), "media_prob_calibrada": float(prob_cal.mean()),
    "pobreza_real": float(y.mean()), "pobreza_ponderada": float(np.average(y, weights=w)),
    "k_clusters": K, "ari_semillas": float(ari_sem), "ari_vs_ward": float(ari_ward),
}
with open(os.path.join(RES, "parametros.json"), "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2, ensure_ascii=False)
log("PARAMETROS: " + json.dumps(params, ensure_ascii=False))
log("MODELADO COMPLETO")
