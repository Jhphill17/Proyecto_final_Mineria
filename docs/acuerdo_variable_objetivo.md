# Acuerdo de variable objetivo y roles de columnas

- **Unidad:** hogar (una fila por `folio`). 12.726 hogares.
- **Variable objetivo principal:** `p0` (pobreza por ingreso, la calcula el INE). `pext0` = etiqueta secundaria (pobreza extrema; **no se predice** en el trabajo principal).
- **Roles:** `folio` = ID · `factor` = peso de expansión (solo estadísticas ponderadas) · `p0`, `pext0` = etiquetas · resto = predictores (63 variables).
- **Regla anti-fuga:** ningún predictor de ingreso ni derivado de ingreso (lista negra en `docs/lista_negra_ingreso.csv`). `pext0` tampoco es predictor. Ante la duda decide el diccionario del INE.
- **Validación:** agrupada por UPM (dos primeros bloques del `folio`, p. ej. `111-000005`).
- **Tasas y tamaños:** siempre con `factor`. Métricas de modelos: sin ponderar (más AUC ponderado).
- **Archivos:** `base_hogares_completa.csv`, `X_predictores.csv`, `metadatos_y_etiquetas.csv`, `diccionario_propio.csv` en `datos/procesados/`.
- **Nota:** los datos son **EH 2025** (no 2023 como decía el plan original); el informe y diapositivas deben actualizarse.

Visto bueno: Juan [ ] · Luciana [ ] · Vania [x]   Fecha: 25 nov 2026
