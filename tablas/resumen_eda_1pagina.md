# Resumen del EDA en 1 página — Pobreza por ingreso, EH 2025 (INE)

> Para Juan (cierre 2.4) y Luciana (dashboard). Todas las tasas son **ponderadas** con `factor`.
> Archivos fuente en `tablas/` y `figuras/`.

## Hallazgos (cada cifra sale de un archivo)

1. **La pobreza por ingreso alcanza al 37,8 % de los hogares (ponderada); la extrema, 13,9 % ponderada (13,6 % sin ponderar).** `[tablas/eda_etiquetas.csv]`
2. **Cochabamba (37,1 %) está por debajo del promedio nacional (37,8 %) pero muy cerca; el departamento más pobre es Potosí (50,1 %) y el menos, Santa Cruz (25,9 %).** `[tablas/eda_pobreza_depto.csv]`
3. **La pobreza es mucho mayor en el área rural (50,9 %) que en la urbana (31,8 %).** `[tablas/eda_pobreza_area.csv]`
4. **A más años de estudio del jefe, menor pobreza: 59,6 % (sin estudios) vs 15,8 % (16+ años); con estudios superiores (>12) baja a 19,9 %.** `[tablas/eda_pobreza_por_anios_estudio_jefe.csv]`, `[tablas/eda_pobreza_por_educacion_jefe.csv]`
5. **Los hogares con hacinamiento (>2,5 personas/dormitorio) tienen 60,1 % de pobreza vs 32,6 % sin hacinamiento; y a menos bienes durables, más pobreza: 52,4 % con 0-4 activos vs 5,3 % con 13-18.** `[tablas/eda_pobreza_por_hacinamiento.csv]`, `[tablas/eda_pobreza_por_activos.csv]`
6. **La inseguridad alimentaria se asocia con más pobreza (29,5 % segura → 53,1 % moderada), pero es probablemente consecuencia y no causa: el equipo decide si entra como predictor o solo descriptiva.** `[tablas/eda_pobreza_por_inseguridad_alim.csv]`

## Datos clave

- 12.726 hogares · 63 predictores · sin nulos · `folio` como ID.
- Desbalance moderado: 37,9 % pobres sin ponderar → no requiere submuestreo agresivo.
- Correlaciones >0,85 confirman los grupos redundantes G1-G6 (ver `diccionario_propio.csv`, columna `grupo_redundancia`).

## Reglas de asociación

`tablas/reglas_asociacion.csv`: 50 reglas (25 → `HOGAR_POBRE`, 25 → `HOGAR_NO_POBRE`), minadas **sin ponderar** con soporte ≥5 %, confianza ≥0,55, lift >1,2, itemsets de hasta 3 ítems (presencia y ausencia de bienes/servicios). Columnas: `antecedente, consecuente, soporte, confianza, lift, n_hogares, soporte_condicional` (=P(antecedente|consecuente)). Ejemplo: *"No posee cocina + sin baño con descarga → hogar pobre; confianza 0,65, lift 1,71"*. Script: `codigo/reglas.py`.

## Limitaciones

- Etiqueta monetaria (ingreso per cápita vs línea de pobreza del INE); subdeclaración del ingreso.
- Datos transversales: las asociaciones no implican causalidad.
- FIES puede ser consecuencia de la pobreza.
- `condact` 4/5 (temporal/permanente) no contados como ocupados — pendiente confirmar con diccionario INE.
