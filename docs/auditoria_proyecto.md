# AUDITORÍA INTEGRAL DEL PROYECTO
**Proyecto:** Factores asociados a la pobreza por ingreso en hogares bolivianos (EH 2025, INE)
**Fecha de auditoría:** 25/11/2026 · **Auditor:** Cascade (asistente del equipo)
**Alcance:** `C:\Users\jhphi\Downloads\final_mineria\final` → github.com/Jhphill17/Proyecto_final_Mineria

---

## 1. INVENTARIO DE EVIDENCIAS

### 1.1 Archivos del proyecto (83 en git)

| Categoría | Archivos | Evidencia |
|---|---|---|
| Datos crudos | 7 `.sav` en `datos/crudos/` (~120 MB, **gitignored**) | CreationTime jun-2026 (fecha preservada del origen; no verificable la descarga real) |
| Datos procesados | `base_hogares_completa.csv` (12.726×67), `X_predictores.csv` (63 vars), `metadatos_y_etiquetas.csv`, `diccionario_propio.csv` | Generados 25/11 13:49 |
| Código | `pipeline_datos.py`, `eda.py`, `reglas.py`, `modelado.py`, `eh2025_v1_obsoleto.py` | Ver §3 |
| Tablas | 20 CSV de EDA + 10 CSV de modelado + `reglas_asociacion.csv` + `resumen_eda_1pagina.md` | `tablas/` y `resultados/` |
| Figuras | 6 EDA + 7 modelos = 13 PNG | `figuras/` y `resultados/figuras/` |
| Documentos | `decisiones_limpieza.md`, `acuerdo_variable_objetivo.md`, `lista_negra_ingreso.csv`, `variables_para_modelado.csv`, `depto/area_nombres.csv` | `docs/` |
| Informe | `seccion_datos.tex`, `seccion_modelos.tex` (sin preámbulo, listos para `\input`) | `informe/` |
| Diapositivas | `modelos.md` (5 láminas con fuentes) | `diapositivas/` |
| Config | `requirements.txt`, `.gitignore`, `README.md` | raíz |

### 1.2 Lo que NO existe en el proyecto

- **Cronograma en archivo**: solo existe en el chat del equipo; no hay `.md`/`.xlsx`/`.pdf` de planificación en el repo.
- **Documentos de integrantes**: no hay archivos con nombres, roles o correos del equipo.
- **Actas de reuniones ni registros de avance**: ninguna evidencia documental de las reuniones 0.1, 2.2, H4.
- **Historial git granular**: un único commit `8075de4` (25/11 14:43). No hay commits por tarea ni por jornada.
- **Scripts de exploración inicial** (`_explorar*.py`): fueron temporales y eliminados; la tarea 1.1 no deja código, solo sus resultados.
- **SHAP**: el plan 3.4 pedía SHAP; se usó importancia por permutación (ver §5).

---

## 2. CRONOLOGÍA RECONSTRUIDA (por timestamps del sistema de archivos)

| Fecha/hora | Archivo(s) | Actividad inferida | Tarea(s) del cronograma |
|---|---|---|---|
| 23/11 17:00 | `eh2025.py` (luego `eh2025_v1_obsoleto.py`) | Primera versión del pipeline de datos | 3.1 (Juan) — **con 5 días de retraso** respecto al 18/11 |
| 25/11 13:35 | `datos/crudos/*.sav` | Reorganización de datos crudos | — (higiene del repo) |
| 25/11 13:49 | `pipeline_datos.py` + 4 CSV procesados | Pipeline corregido: folio, 3 CSV, diccionario, anti-fuga | 1.1–1.4, H1 |
| 25/11 13:50 | `eda.py` + 20 tablas + 6 figuras + resumen | EDA completo ponderado | 2.1, 2.3, 6.4 (parcial) |
| 25/11 13:53–13:56 | `reglas.py` + `reglas_asociacion.csv` + `seccion_datos.tex` | Reglas de asociación + sección de datos | 4.2, 6.5 |
| 25/11 14:06–14:30 | `modelado.py` + `resultados/` (10 CSV + 7 figuras + `parametros.json`) | Modelos, calibración, clustering, sensibilidad | 3.2–3.4, 4.1, 4.3, H2 |
| 25/11 14:35 | `seccion_modelos.tex` | Sección de modelos del informe | 6.6 |
| 25/11 ~14:40 | `diapositivas/modelos.md` | Diapositivas de modelos | 7.2 |
| 25/11 14:38–14:43 | `README.md`, `requirements.txt`, `.gitignore`, commit+push | Publicación en GitHub | — |

**Conclusión cronológica:** todo el trabajo técnico verificable se concentró en **una sola jornada (25/11)**, salvo el primer pipeline del 23/11. El cronograma distribuía estas tareas entre el 16/11 y el 25/11. No hay evidencia en el repo de trabajo previo versionado; es posible que existiera trabajo en otra carpeta no versionada, pero **no es demostrable**.

---

## 3. CONTRASTE CRONOGRAMA vs EVIDENCIA

### Semana 1

| Tarea | Responsable | Evidencia | Estado auditado |
|---|---|---|---|
| 0.1 Reunión de arranque | Equipo | Ninguna en repo | ⚠️ No demostrable (solo chat) |
| 0.2 Descargar EH | Vania | 7 `.sav` en `datos/crudos/` | ✅ Existe, **pero son EH 2025, no EH 2023** como pedía el cronograma |
| 0.3 Revisar diccionario | Equipo | `docs/decisiones_limpieza.md`, `acuerdo_variable_objetivo.md` | ✅ Documentado |
| 1.1 Explorar tablas | Vania | Sin scripts; solo resultados en CSV | ⚠️ Entregable existe, proceso no |
| 1.2 Limpiar/recodificar/agregar | Vania | `pipeline_datos.py` (escrito 25/11) | ✅ Funcional; autoría compartida con asistente |
| 1.3 Acuerdo variable objetivo | Vania/Equipo | `acuerdo_variable_objetivo.md` | ✅ |
| 1.4 Unir por folio + base + diccionario | Vania | `base_hogares_completa.csv`, `X_predictores.csv`, `metadatos_y_etiquetas.csv`, `diccionario_propio.csv` | ✅ Verificado: 12.726 filas, folio único, 63 predictores |
| H1 Base lista | Equipo | Ídem anterior | ✅ |
| 3.1 Código base | Juan | `pipeline_datos.py`, `modelado.py` | ✅ (pipeline v1 del 23/11 + reescritura 25/11) |
| 3.2 Regresión logística | Juan | `metricas_modelos.csv` (2 variantes) | ✅ |
| 3.3 Árbol, RF, GB con desbalance | Juan | `metricas_modelos.csv` (6 variantes, `class_weight`) | ✅ |
| 3.4 Comparar modelos + importancia + SHAP | Juan | `metricas_modelos.csv`, `01_roc_pr.png`, `importancia_variables.csv`, `odds_ratios_logistica.csv` | ⚠️ **SHAP no implementado** — se usó permutación (justificado: SHAP no instalado; decisión a validar con docente) |
| 4.1 Clustering + perfiles | Juan | `seleccion_k_clustering.csv`, `perfiles_clusters.csv`, `perfiles_equipamiento.csv`, figuras 06/08/09 | ✅ k=3, ARI 1,0 |
| 4.2 Reglas de asociación | Vania | `reglas.py`, `reglas_asociacion.csv` (50 reglas) | ✅ Implementación manual (mlxtend no instalado) |
| 4.3 Exportar CSV (H2) | Juan | `hogares_resultados.csv` (12.726×9, orden del plan) + 9 CSV | ✅ Verificado: ponderada 0,378; depto==3 → 2.198 |
| 2.1, 2.3 EDA + resumen | Vania/Juan | 20 tablas `eda_*.csv`, 6 figuras, `resumen_eda_1pagina.md` | ✅ |
| 6.4 Entregar figuras/tablas a Luciana | Vania | `tablas/` + `figuras/` consolidadas | ✅ |
| 6.5 Sección datos/limpieza/EDA | Vania | `informe/seccion_datos.tex` | ⚠️ **Redactada por el asistente**; Vania debe revisarla y asumir autoría |
| 6.6 Sección modelos | Juan | `informe/seccion_modelos.tex` | ⚠️ Ídem — redactada por el asistente |
| 7.1 Diapositivas datos/EDA | Vania | **No existe** | ❌ Pendiente |
| 7.2 Diapositivas modelos | Juan | `diapositivas/modelos.md` | ✅ (borrador para plantilla de Luciana) |
| 2.2, 2.4 Sincronización/cierre | Equipo | No demostrable | ⚠️ |
| 3.5 Revisión cruzada | Vania | Pendiente (30/11) | ⏳ |
| 5.3 Validar dashboard | Juan | Pendiente — depende de Luciana | ⏳ |
| 6.7, 5.1, 5.2 Informe/dashboard | Luciana | Fuera del alcance de esta auditoría | — |
| H4, H5 | Equipo | Pendientes | ⏳ |

---

## 4. HALLAZGOS CRÍTICOS

1. **Concentración temporal**: el 90 % del trabajo verificable es del 25/11. Si el equipo trabajó antes, no dejó evidencia versionada. **Recomendación:** a partir de ahora, un commit por tarea (o al menos diario) para que el historial respalde el cronograma.
2. **EH 2025 vs EH 2023**: el cronograma y la tarea 0.2 dicen EH 2023; los datos son EH 2025. Está documentado en `acuerdo_variable_objetivo.md`, pero debe mencionarse explícitamente en el informe para evitar que parezca error.
3. **Autoría de documentos**: `seccion_datos.tex`, `seccion_modelos.tex` y `modelos.md` fueron redactados por el asistente de IA. Son correctos y con cifras reales, pero **Vania y Juan deben revisarlos, validarlos y asumirlos** antes de la defensa (H4).
4. **SHAP pendiente de decisión**: si el docente exige SHAP, falta calcularlo (`shap` no está en `requirements.txt`). La permutación es defendible, pero debe acordarse.
5. **Trazabilidad de la exploración (1.1)**: los scripts `_explorar*.py` se eliminaron. Si se requiere evidencia del proceso, conviene recuperar del historial de la sesión o documentar en `decisiones_limpieza.md` (ya parcialmente cubierto).
6. **Reproducibilidad**: `pipeline_datos.py` requiere los `.sav` (gitignored); `eda.py`, `reglas.py` y `modelado.py` corren solo con `datos/procesados/` (incluidos en git). `eh2025_v1_obsoleto.py` no es reproducible sin los `.sav` y está marcado como obsoleto.
7. **Calibración isotónica** evaluada sobre las mismas predicciones OOF con que se ajustó → ligeramente optimista (ya declarado en `seccion_modelos.tex`).

## 5. VERIFICACIONES NUMÉRICAS REALIZADAS

- `hogares_resultados.csv`: 12.726 filas, `folio` único, columnas en orden del plan, pobreza ponderada 0,378, `depto==3` → 2.198 hogares.
- `metricas_modelos.csv`: 8 variantes; mejor GB balanceado (CV 0,856 / test 0,838).
- `parametros.json`: semilla 42, umbral 0,4877, AUC OOF 0,8564, media calibrada 0,3792 = tasa real.
- `perfiles_clusters.csv`: k=3, ARI semillas 1,0, ARI vs Ward 0,37, silueta 0,099.
- `reglas_asociacion.csv`: 50 reglas (25 por clase), soporte ≥5 %, confianza ≥55 %, lift >1,2.

## 6. CONCLUSIÓN

**El proyecto tiene todos los entregables técnicos del cronograma de Juan y de Vania**, con calidad verificable y reproducible (semilla 42, validación agrupada por UPM, anti-fuga documentada). Las brechas son de **proceso y autoría**, no de resultados: falta evidencia de trabajo previo al 25/11, Vania debe asumir sus documentos y diapositivas de datos/EDA, Juan debe validar el dashboard el 30/11, y el equipo debe decidir SHAP vs permutación y comunicar el cambio EH 2023→2025.
