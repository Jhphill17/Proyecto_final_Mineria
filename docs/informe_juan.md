# Informe de avance — Juan
**Proyecto:** Factores asociados a la pobreza por ingreso en hogares bolivianos (EH 2025)
**Periodo:** 16/11 – 25/11/2026 · **Estado al 25/11/2026**

---

## Resumen

Completé el **100 % de mis entregables técnicos** del cronograma (modelado, clustering, exportación de resultados, sección de modelos del informe y diapositivas). Lo único pendiente es la **validación del dashboard** (30/11), que depende del trabajo de Luciana, y la revisión cruzada de Vania.

## Tareas realizadas

### Semana 1

| Tarea | Qué hice | Evidencia |
|---|---|---|
| **3.1** Código base (pipeline, CV, métricas) | Escribí `eh2025.py` (23/11) y luego `pipeline_datos.py` completo: carga de los 4 `.sav`, limpieza, recodificación, agregación a nivel hogar, anti-fuga de variables de ingreso, exportación de `base_hogares_completa.csv`, `X_predictores.csv`, `metadatos_y_etiquetas.csv` y `diccionario_propio.csv`. | `codigo/pipeline_datos.py`, `datos/procesados/` |
| **3.2** Regresión logística | Implementada con `class_weight` balanceado/sin balancear, `C` en {0.01–1.0}, L2. AUC CV 0,851 / test 0,831. | `codigo/modelado.py`, `resultados/metricas_modelos.csv` |
| **3.3** Árbol, Random Forest y Gradient Boosting con desbalance | Las 3 familias × con/sin `class_weight="balanced"`, con búsqueda de hiperparámetros por `StratifiedGroupKFold` (5 pliegues, agrupado por UPM). Mejor: GB balanceado (CV 0,856 / test 0,838). | `codigo/modelado.py`, `resultados/metricas_modelos.csv` |
| **3.4** Comparar modelos, importancia de variables | Comparación de las 8 variantes (AUC, precision-recall, Brier); importancia por **permutación** (10 repeticiones); odds ratios de la logística; dependencia parcial. ⚠️ Usé permutación en lugar de SHAP — pendiente acordar con el docente. | `resultados/metricas_modelos.csv`, `importancia_variables.csv`, `odds_ratios_logistica.csv`, `figuras/01_roc_pr.png`, `02_importancia.png`, `03_dependencia_parcial.png` |
| **4.1** Clustering y perfiles | K-Means sobre 49 variables del hogar (PCA al 90 % varianza = 31 componentes), selección de k con silueta/Calinski-Harabasz/Davies-Bouldin, k=3 estable (ARI 1,0 entre semillas). Perfiles ponderados descritos. | `resultados/seleccion_k_clustering.csv`, `perfiles_clusters.csv`, `perfiles_equipamiento.csv`, `figuras/06_seleccion_k.png`, `08_perfiles_pobreza.png`, `09_clusters_pca.png` |
| **4.3** Exportar resultados (H2) | `hogares_resultados.csv` con `folio`, `depto`, `area`, `factor`, `p0`, `pext0`, `cluster`, `prob_pobreza` **calibrada** (isotónica, media 0,379 = tasa real) y `pred_pobre` (umbral F1-óptimo 0,488). Más 9 CSV de apoyo (métricas, importancia, perfiles, sensibilidad, subgrupos, calibración, odds ratios). | `resultados/hogares_resultados.csv` + `tablas/` |
| **2.2** Sincronización de equipo | Participé en el reparto y ajustes (documentado en chat del equipo). | — |

### Semana 2

| Tarea | Qué hice | Evidencia |
|---|---|---|
| **6.6** Sección de modelos del informe | Redacté `seccion_modelos.tex` completa: diseño experimental (validación agrupada por UPM), comparación de modelos, umbral y calibración, variables influyentes, sensibilidad, subgrupos, perfiles, reglas de asociación y limitaciones. Sin preámbulo, lista para `\input`. | `informe/seccion_modelos.tex` |
| **7.2** Diapositivas de modelos | Redacté `diapositivas/modelos.md`: 5 láminas (validación por UPM, comparación, variables influyentes, perfiles, límites) con cifras y fuentes para la plantilla de Luciana. | `diapositivas/modelos.md` |
| **2.4** Cierre EDA/modelos (parcial) | Los mensajes clave están integrados en `seccion_modelos.tex` y en las diapositivas; falta la sesión de cierre formal del equipo. | — |

## Resultados clave (mis números)

- **Pobreza:** 37,8 % de hogares (ponderada con `factor`).
- **Modelo elegido:** Gradient Boosting con pesos balanceados — **AUC CV agrupada 0,856 / prueba 0,838** (0,845 ponderado), Brier 0,164.
- **Predictor dominante:** `tasa_dependencia_ocupados` (caída de AUC 0,085 por permutación), seguido de la ocupación del jefe, territorio y educación.
- **Calibración:** `prob_pobreza` isotónica con media 0,379 = tasa real exacta.
- **Umbral:** 0,488 → recall 0,75 / precisión 0,63 en prueba.
- **Perfiles:** 3 clusters — familias numerosas jóvenes (58 % pobreza), hogares pequeños de adultos mayores (45 %), hogares urbanos con más estudios y bienes (19 %).
- **Sensibilidad:** sin inseguridad alimentaria 0,853 · sin territorio 0,852 · sin equipamiento 0,855 → ningún grupo imprescindible.
- **Subgrupos:** Cochabamba 0,837 vs resto 0,838 · urbano 0,835 vs rural 0,831.

## Pendiente (no depende de mí)

| Tarea | Fecha | Bloqueo |
|---|---|---|
| **5.3** Validar dashboard | 30/11 | Esperando el Power BI de Luciana (5.1–5.2) |
| **3.5** Revisión cruzada | 30/11 | Es de Vania, pero debo estar disponible para sustentar reproducibilidad |
| **H3, H5** | 30/11 | Dependen del dashboard y la entrega final |
| Decisión SHAP vs permutación | Antes del 30/11 | Acordar con docente/equipo |

## Notas para la defensa

1. **EH 2025, no EH 2023**: el cronograma decía EH 2023 pero los datos disponibles son EH 2025; está documentado en `docs/acuerdo_variable_objetivo.md` y debe mencionarse en el informe.
2. **Validación agrupada por UPM**: todo el modelado usa `StratifiedGroupKFold` con la UPM (segmento de muestreo) como grupo — es el punto metodológico más defendible.
3. **Anti-fuga**: ningún predictor de ingreso; `p0`, `pext0`, `factor`, `folio` excluidos; lista negra en `docs/lista_negra_ingreso.csv`.
4. **Reproducibilidad**: `python codigo/modelado.py` regenera todo con semilla 42 en ~6 minutos.
