# Decisiones de limpieza y construcción de la base — EH 2025 (INE)

> Registro de cada decisión de limpieza/recodificación: qué, por qué y cuántos hogares afecta.
> Fuente: Encuesta de Hogares **2025** del INE de Bolivia (catálogo ANDA). Archivos `.sav` leídos con `pyreadstat` (se conservan etiquetas de valores).
> Script reproducible: `codigo/pipeline_datos.py`.

## 0. Archivos crudos (solo lectura)

| Archivo | Contenido | Filas |
|---|---|---|
| `EH2025_Persona.sav` | Características de las personas + etiquetas de pobreza | ~35 mil personas |
| `EH2025_Vivienda_1.sav` | Características de la vivienda (1 fila/hogar) | 12.726 |
| `EH2025_Seguridad_Alimentaria.sav` | FIES (8 preguntas, 1 fila/hogar) | 12.726 |
| `EH2025_Equipamiento.sav` | Bienes del hogar (18 ítems × hogares, formato largo) | 229.068 |
| `EH2025_Gastos_Alimentarios.sav` | **NO SE USA** — gasto alimentario (proxy de ingreso, fuga) | — |
| `EH2025_Gastos_No_Alimentarios.sav` | **NO SE USA** — gasto no alimentario (proxy de ingreso, fuga) | — |
| `EH2025_Discriminacion.sav` | **NO SE USA** — módulo de discriminación, fuera del alcance | — |

## 1. Unidad de análisis y claves

- **Unidad:** el hogar (una fila por `folio`, formato `111-000005-A-0011`). Los dos primeros bloques (`111-000005`) son la UPM/segmento de muestreo (~1.088 grupos).
- Las 4 tablas usadas comparten **exactamente los mismos 12.726 folios** (verificado con `indicator=True`: todas las uniones dan `both=12726`). **No se excluyó ningún hogar** por falta de tablas.
- `folio` se lee como texto (`str.strip()`).
- **Jefe del hogar:** `s01a_05 == 1` (JEFE O JEFA DEL HOGAR). Verificado: exactamente 1 jefe por folio, 0 hogares sin jefe, 0 duplicados.

## 2. Etiquetas de pobreza

- `p0` = pobreza por ingreso (1=Pobre, 0=No pobre), calculada por el INE. `pext0` = pobreza extrema.
- **21 personas (en 16 folios) tienen `p0`/`pext0` nulo**, pero todos esos hogares tienen otros miembros con valor → se toma el primer valor no nulo por `folio`. **Afecta a 0 hogares** (todos conservan su etiqueta).
- `pext0` **no es predictor** (fuga de datos: deriva del mismo ingreso).
- `factor` = factor de expansión muestral (idéntico en las 4 tablas, verificado). **Solo estadísticas ponderadas; nunca predictor.**

## 3. Decisiones de recodificación

| Variable | Decisión | Motivo | Hogares afectados |
|---|---|---|---|
| `s06a_17` (dormitorios) | `0 → 1` en `num_dormitorios` | 0 = el hogar no reporta cuarto usado *exclusivamente* para dormir (duerme en cuartos de uso múltiple); se asume mínimo 1 espacio para dormir | 1.046 |
| `s06a_10` (desagüe) | `NaN → 0` | Solo aplica a ciertos tipos de baño; 0 = "no aplica" | 3.230 |
| `s06a_11` (baño compartido) | `NaN → 0` | Ídem | 960 |
| `s06a_20` (tipo internet) | `NaN → 0` cuando `s06a_19 != 1`; resto `NaN → 0` | 0 = "sin conexión a internet" | 3.298 |
| FIES `s07a_01..08` | `1 (Sí) → 1`; `2 (No)` y `3 (NS/NR) → 0` | NS/NR se trata como respuesta negativa (criterio conservador FIES) | pocos casos |
| `jefe_tipo_ocupacion` (`s04b_12`) | `NaN → 0` | 0 = "no aplica": jefe sin ocupación. **No son faltantes** | 2.146 |
| `jefe_grupo_ocup` (`cob_op`) | `NaN → 0` | Ídem | 2.146 |
| `jefe_anios_estudio` (`aestudio`) | `NaN → 0` | Sin dato de escolaridad = 0 años (jefes sin NaN en la práctica) | 0 |
| `condact` (ocupación) | Ocupado = `condact == 1` | Es el único grupo que trabajó la semana pasada (`s04a_01==1` en 19.012/19.769). `condact` 4/5 (`p_temporal`/`p_permanente`) reportan `s04a_01==2` (no trabajaron) → **no se cuentan como ocupados**. Pendiente confirmar con diccionario INE | — |
| `equip_item_*` | `s08c_1==1 (Sí) → 1`; `==2 (No) → 0` | Cada hogar tiene exactamente 18 ítems | — |
| Ítems de equipamiento | **Se incluyen los 18** (a diferencia de la versión anterior, que excluía celular/radio/minicomponente/bicicleta) | Son predictores legítimos; la relevancia la decide el modelo | — |

## 4. Variables construidas

- `tamano_hogar`, `num_menores_14`, `num_adultos_mayores`, `num_ocupados` (conteos por `folio`).
- `prop_menores_14`, `prop_adultos_mayores` (proporciones).
- `tasa_dependencia_ocupados` = (tamano − ocupados)/ocupados; **si ocupados = 0 → `tamano_hogar`** (todo el hogar depende de transferencias).
- `anios_estudio_prom_15mas` = promedio de `aestudio` solo de miembros ≥15 años (evita sesgo a la baja por menores).
- `num_dormitorios`, `personas_por_dormitorio` = tamano/num_dormitorios, `hacinamiento` = ratio > 2.5.
- `jefe_estudio_rango` (0-6 / 7-12 / >12 años), `tamano_hogar_rango` (1-2 / 3-4 / 5+).
- `puntaje_inseguridad_alim` (0-8) y `nivel_inseguridad_alim` (FIES: 0 segura, 1 leve, 2 moderada, 3 severa).
- `total_activos_hogar` = suma de los 18 bienes.
- Del jefe: `jefe_sexo`, `jefe_edad`, `jefe_indigena`, `jefe_estado_civil`, `jefe_nivel_ed` (`niv_ed` 0-6), `jefe_anios_estudio`, `jefe_condact`, `jefe_tipo_ocupacion`, `jefe_grupo_ocup`.
- `tipohogar` (tipología de hogar del INE).

## 5. Exclusiones y advertencias

- **Ningún predictor de ingreso ni derivado** (ver `docs/lista_negra_ingreso.csv`: 99 variables de ingreso/gasto detectadas por etiqueta + tablas de gastos excluidas).
- **21 jefes menores de 18 años** (2 con ≤15): se conservan, no se borran (son jefes declarados por el INE).
- Edades: 0-98 años (sin valores imposibles). `aestudio`: 0-23.
- `jefe_grupo_ocup` = `cob_op` (grupo ocupacional); `caeb_op` (actividad económica, 21 categorías) está disponible pero **no se incluyó** (granularidad alta, solo para ocupados).
- **Limitación FIES:** la inseguridad alimentaria puede ser consecuencia (no causa) de la pobreza; el equipo decide si entra como predictor o solo descriptiva.

## 6. Cifras de control (verificadas el 25 nov)

- 12.726 hogares · 67 columnas en `base_hogares_completa.csv` · 63 predictores.
- Pobreza: **37,9 % sin ponderar / 37,8 % ponderada** · extrema **13,6 %**.
- Cochabamba (`depto=3`): **2.198 hogares, 37,1 % ponderada**.
- Área: urbano (`area=1`) n=9.611 → 31,8 %; rural (`area=2`) n=3.115 → 50,9 %.
- Códigos confirmados: `depto` 1=Chuquisaca … 3=Cochabamba … 9=Pando; `area` 1=Urbano, 2=Rural.
