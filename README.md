# Factores asociados a la pobreza por ingreso en los hogares bolivianos

**Predicción y perfiles de hogares mediante minería de datos** — Proyecto final, curso de Minería de Datos, Universidad del Valle (Cochabamba, Bolivia). Docente: Cecilia Reyes Trigo.

Fuente: **Encuesta de Hogares (EH) 2025 del INE de Bolivia** (catálogo ANDA). Unidad de análisis: el hogar (12.726 hogares).

## Estructura

```
├── datos/
│   ├── crudos/            # .sav del INE (NO en git; ver "Datos" abajo)
│   └── procesados/        # base_hogares_completa.csv, X_predictores.csv,
│                          # metadatos_y_etiquetas.csv, diccionario_propio.csv
├── codigo/
│   ├── pipeline_datos.py  # .sav -> CSV procesados (limpieza, agregación, anti-fuga)
│   ├── eda.py             # análisis exploratorio -> tablas/eda_*.csv + figuras/
│   ├── reglas.py          # reglas de asociación -> tablas/reglas_asociacion.csv
│   ├── modelado.py        # modelos supervisados + clustering + calibración -> resultados/
│   └── eh2025_v1_obsoleto.py  # versión anterior del pipeline (solo referencia)
├── resultados/            # CSV del modelado + figuras/ + parametros.json
├── tablas/                # tablas finales para el dashboard (Luciana)
├── figuras/               # figuras finales (modelos + EDA)
├── docs/                  # decisiones_limpieza.md, acuerdo_variable_objetivo.md,
│                          # lista_negra_ingreso.csv, variables_para_modelado.csv,
│                          # depto_nombres.csv, area_nombres.csv
├── informe/               # seccion_datos.tex, seccion_modelos.tex
├── diapositivas/          # modelos.md (contenido para la plantilla)
└── requirements.txt
```

## Datos

Los `.sav` del INE **no están en el repositorio** (~120 MB). Descargarlos del catálogo ANDA: https://anda.ine.gob.bo/index.php/catalog/108 y dejarlos en `datos/crudos/`. Los CSV procesados sí están incluidos, por lo que `eda.py`, `reglas.py` y `modelado.py` corren sin los `.sav`.

## Cómo reproducir

```bash
pip install -r requirements.txt

# 1) Solo si tienes los .sav en datos/crudos/ (regenera datos/procesados/):
python codigo/pipeline_datos.py

# 2) EDA, reglas y modelado (usan datos/procesados/):
python codigo/eda.py
python codigo/reglas.py
python codigo/modelado.py        # ~6 min, semilla 42
```

## Resultados principales

- **Pobreza por ingreso:** 37,8 % de hogares (ponderada); rural 50,9 % vs urbano 31,8 %; Potosí el departamento más pobre (50,1 %).
- **Modelo elegido:** Gradient Boosting con pesos balanceados — **AUC CV agrupada 0,856 / prueba 0,838** (0,845 ponderado), validación agrupada por UPM (segmento de muestreo), semilla 42.
- **Predictor dominante:** `tasa_dependencia_ocupados`, seguido de la ocupación del jefe, el territorio y la educación.
- **`prob_pobreza` calibrada** (isotónica) a la tasa real del 38 %; umbral 0,488.
- **3 perfiles de hogar** (K-Means, tendencias): familias numerosas jóvenes (58 % pobreza), hogares pequeños de adultos mayores (45 %) y hogares urbanos con más estudios y bienes (19 %).
- **50 reglas de asociación** (presencia/ausencia de bienes y servicios → pobreza / no pobreza).

## Reglas del proyecto

- Ningún predictor de ingreso, `p0`, `pext0`, `factor` ni `folio` (lista negra en `docs/`).
- Validación siempre agrupada por UPM; semilla 42; preprocesamiento dentro del `Pipeline`.
- Tasas poblacionales ponderadas con `factor`; métricas de modelo sin ponderar (+ AUC ponderado).
