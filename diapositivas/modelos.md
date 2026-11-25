# Diapositivas de modelos — Proyecto final Minería de Datos
> Contenido para 5 diapositivas (plantilla de Luciana). Cifras redondeadas; fuente en notas del orador.

## Diapositiva 1 · Validación: ¿por qué agrupar por UPM?
- 12.726 hogares en 1.088 grupos de muestreo (UPM = dos primeros bloques del `folio`).
- Hogares del mismo segmento se parecen → validación aleatoria daría resultados optimistas.
- `StratifiedGroupKFold` (5 pliegues): ninguna UPM aparece en entrenamiento y prueba a la vez.
- Partición 80/20 + CV agrupada; semilla 42; todo reproducible (`codigo/modelado.py`).
- *Fuente: `resultados/parametros.json`.*

## Diapositiva 2 · Comparación de modelos
- 4 familias × con/sin pesos de clase: logística, árbol, Random Forest, Gradient Boosting.
- Mejor: **Gradient Boosting balanceado — AUC CV 0,86 / prueba 0,84** (ponderado 0,85).
- Los ensambles mejoran solo ~0,01 sobre la logística → la pobreza se explica en gran parte de forma casi lineal.
- *Fuente: `resultados/metricas_modelos.csv`, figura `01_roc_pr.png`.*

## Diapositiva 3 · Variables influyentes
- Importancia por permutación (caída del AUC):
  1. `tasa_dependencia_ocupados` (0,085 — muy por encima del resto)
  2. `jefe_tipo_ocupacion`, `jefe_grupo_ocup`
  3. `anios_estudio_prom_15mas`, `depto`, `total_activos_hogar`, `tamano_hogar`
- Odds ratios: más tamaño del hogar y más menores → más riesgo; más ocupados y más estudio → menos riesgo.
- *Fuente: `resultados/importancia_variables.csv`, `odds_ratios_logistica.csv`, figura `02_importancia.png`.*

## Diapositiva 4 · Perfiles de hogares (clustering)
- K-Means sobre 49 variables (PCA 90 % varianza, 31 componentes), k = 3 estable (ARI 1,0 entre semillas; silueta 0,10 → tendencias, no grupos nítidos).
- **Cluster 0** "Hogares pequeños de adultos mayores": 25 % de hogares, pobreza 45 % (jefe de 62 años, 5 años de estudio, 67 % rural).
- **Cluster 1** "Familias numerosas jóvenes": 32 %, pobreza **58 %** (4,3 personas, 37 % menores, hacinamiento 53 %).
- **Cluster 2** "Hogares urbanos con más estudios y bienes": 44 %, pobreza 19 % (13,4 años de estudio del jefe, 94 % urbano).
- *Fuente: `resultados/perfiles_clusters.csv`, `perfiles_equipamiento.csv`, figuras `08_perfiles_pobreza.png`, `09_clusters_pca.png`.*

## Diapositiva 5 · Límites y mensajes clave
- AUC ≈ 0,84 en datos no vistos: se identifica la pobreza sin usar ingreso (proxy means test).
- `prob_pobreza` calibrada (isotónica) a escala del 38 % real — es una estimación, no un diagnóstico.
- Umbral 0,488: detecta 75 % de los pobres (recall) con precisión 63 %.
- Limitaciones: etiqueta monetaria, subdeclaración del ingreso, datos transversales (no causalidad), perfiles solapados.
- *Fuente: `resultados/parametros.json`, `calibracion_deciles.csv`, `hogares_resultados.csv`.*
