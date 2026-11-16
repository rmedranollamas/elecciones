# Elecciones 29-N (2026) — Predictor Probabilístico del Congreso (LOREG Art. 163)

Sistema estadístico completo para la predicción probabilística de escaños al Congreso de los Diputados en las **elecciones generales de España del 29 de noviembre de 2026 (29-N)**, calibrado fuera de muestra sobre las cinco convocatorias generales de 2015 a 2023 (`260` carreras provinciales · `1.750` escaños) y compilado en un tablero editorial autónomo (`docs/index.html`).

---

## Características principales

- **Motor exacto LOREG Art. 163 (`dhondt.py`)**:
  - Barrera legal del **3%** sobre voto válido provincial (**candidaturas + voto en blanco**), regla de mayoría simple (`0%`) en **Ceuta (51)** y **Melilla (52)**, y desempate legal estricto por mayor número total de votos (`Art. 163.1.d`).
  - Verificado con **0 discrepancias sobre 1.750 escaños históricos** (`2015-D`, `2016-J`, `2019-A`, `2019-N`, `2023-J`) y `0` discrepancias en las elecciones europeas de junio de 2024 (`BOE-A-2024-15344`).
- **Ingesta oficial y demoscópica (`fetch`)**:
  - Catálogo 2026 actualizado (`BOE-A-2026-20742` y proclamación de candidaturas del 3-N `BOE-A-2026-23103`): **Madrid 38 escaños**, **Cádiz 8 escaños**, máscara binaria exacta $M_{c,p} \in \{0,1\}$ (p. ej. `UPN_PP` en Navarra y exclusión de candidaturas no proclamadas).
  - Parser de tablas MediaWiki de Wikipedia (`national_2015_2023.csv`, `national_2026.csv`, `regional_2026.csv`), parser de tabulaciones del **CIS Preelectoral #3584 (`N = 45.074`)** con desesgado IPF (`cis_3584_provincial.csv`), verificación cruzada *Brújula 29N* y soporte de parches manuales (`data/polls/manual_polls_2026.csv`).
- **Modelado estadístico en cuatro etapas**:
  - **Etapa 1 — Agregación nacional y sesgos de casa**:
    - Índice histórico de casas encuestadoras (`pollster_ratings.csv`: MAE encogido por Empirical Bayes, grados `A/B/C`, detección de *herding* y efectos de casa anclados por resultados oficiales).
    - **Etapa 1A (`kernel`)**: promedio ponderado por grado, decaimiento exponencial de recencia (vida media de campaña/no-campaña), tope de tamaño muestral efectivo (`N_eff`), penalización por solapamiento en *tracking* diarios, ajuste centrado de *house effects* a lo Jackman y conmutador `--exclude-cis-headline`.
    - **Etapa 1B (`state_space`)**: filtro/suavizador dinámico en espacio de razón logarítmica aditiva (**ALR**) con propagación de deriva por paseo aleatorio durante la veda electoral (24–28 nov).
    - **Ensemble calibrado (`ensemble`)**: combinación lineal convexa calibrada mediante backtest LOOCV (`0.25 Kernel + 0.75 Espacio-Estado`).
  - **Etapa 2 — Proyección provincial IPF (`provincial.py`)**:
    - Prior compuesto provincial, matrices de transferencia de coaliciones (`UPN–PP`, `FA`/`Podemos`/`Compromís-FV`/`UPC`/`Ara Més`/`Adelante`, `SALF`, `AC`) estrictamente enmascaradas por $M_{c,p}$, desplazamiento *swing* logit multinomial + rastrillado IPF ponderado por participación, y encogimiento *Empirical Bayes* con encuestas autonómicas/provinciales y el CIS #3584.
  - **Etapa 3 — Simulador Monte Carlo jerárquico Student-$t$ (`covariance.py`, `simulator.py`)**:
    - `50.000` simulaciones en `< 1.5s` (`~1.1–1.3s` en CPU estándar) mediante vectorización comprimida sobre celdas activas ($M_{c,p}=1$).
    - Descomposición jerárquica de errores en log-razón (**Nacional + CCAA + Provincial**) con colas pesadas Student-$t$ ($\nu = 4.5$), choques correlacionados de bloque izquierda/derecha, covariancia posterior de Etapa 1 y variación estocástica de participación y voto en blanco.
    - Clasificación de escaños en niveles de certidumbre: **Seguro** ($\ge 99\%$), **Probable** ($75\text{–}99\%$), **En juego** ($25\text{–}75\%$), **Remoto** ($5\text{–}25\%$).
  - **Etapa 4 — Investidura Art. 99 CE (`investiture.py`)**:
    - Probabilidades de **1.ª votación** ($\ge 176$ escaños, mayoría absoluta), **2.ª votación** (mayoría simple estricta $\text{Sí} > \text{No}$ con reglas explícitas de abstención) y riesgo de bloqueo institucional / repetición electoral (**Art. 99.5 CE**).
- **Tablero editorial cero-backend (`docs/index.html`)**:
  - Hemiciclo interactivo SVG de 350 escaños, cartograma de las 52 circunscripciones con inspector D'Hondt por provincia, series temporales de 100 días, simulador en vivo *"What-If"* sobre las 52 circunscripciones en el navegador y constructor interactivo de pactos de investidura.

---

## Calibración histórica fuera de muestra (LOOCV 2015–2023)

Evaluado de manera estricta fuera de muestra dejando fuera cada elección completa (`260` carreras provinciales · `1.750` escaños · horizontes `T-15d`, `T-6d`, `T-1d`):

| Motor | MAE Voto Nac. (pp) | MAE Esc. Prov. | MAE Esc. Nac. | Brier Score | Cob. 80% (P/N) | Cob. 95% (P/N) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Stage 1B Espacio-Estado** | 1.02 pp | 0.352 | 3.32 | 0.0747 | 96.4% / 85.0% | 98.7% / 92.5% |
| **Ensemble Calibrado** | 1.03 pp | 0.353 | 3.39 | 0.0761 | 96.1% / 83.8% | 98.6% / 91.9% |
| **Stage 1A Kernel** | 1.08 pp | 0.353 | 3.61 | 0.0773 | 95.3% / 80.8% | 98.3% / 90.2% |

Tasa empírica de acierto por nivel de certidumbre fuera de muestra:
- **Seguro ($\ge 99\%$)** → **99.3%** (`2.501 / 2.518`)
- **Probable ($75\text{–}99\%$)** → **92.4%** (`1.711 / 1.851`)
- **En juego ($25\text{–}75\%$)** → **51.6%** (`860 / 1.668`)
- **Remoto ($5\text{–}25\%$)** → **9.7%** (`138 / 1.424`)

---

## Instalación y uso del CLI

```bash
# Instalar dependencias con uv
uv sync --frozen

# Ejecutar toda la cadena (ingesta -> ratings -> backtest -> forecast -> dashboard)
uv run elecciones run-all --offline

# Comandos individuales
uv run elecciones fetch --offline                  # Validar snapshots y metadata BOE
uv run elecciones ratings                          # Calcular MAE y sesgos de casa
uv run elecciones backtest                         # Calibración LOOCV 2015-2023
uv run elecciones forecast --n-sims 50000          # Simulación 50k (~1.1s) -> forecast_29n.json
uv run elecciones forecast --exclude-cis-headline  # Sensibilidad sin CIS en titular
uv run elecciones dashboard                        # Compilar docs/index.html autónomo
uv run elecciones allocate -i distrito.json --json # Reparto LOREG Art. 163 ad-hoc
```

### Añadir o corregir encuestas manualmente

Edita [`data/polls/manual_polls_2026.csv`](data/polls/manual_polls_2026.csv) con filas `action=add` o `action=override` para incorporar sondeos nuevos o corregir erratas antes de ejecutar `elecciones forecast` o `elecciones run-all --offline`.

---

## Cumplimiento legal: Veda Electoral (LOREG Art. 69.7)

En cumplimiento del **artículo 69.7 de la LOREG** (prohibición de publicación y difusión de sondeos electorales durante los cinco días anteriores a la votación), tanto el CLI (`fetch`, `run-all`) como el flujo automático diario [`.github/workflows/update_forecast.yml`](.github/workflows/update_forecast.yml) comprueban la zona horaria `Europe/Madrid` y bloquean automáticamente la ingesta y publicación entre:

- **Inicio**: `2026-11-24 00:00:00 CET`
- **Fin**: `2026-11-29 20:00:00 CET`

Puedes verificar el guardián localmente con:
```bash
uv run elecciones fetch --offline --respect-blackout --now 2026-11-25T12:00:00+01:00
```

---

## Calidad, pruebas y CI/CD

```bash
uv run ruff check .
uv run pytest
```

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): ejecuta `ruff`, toda la batería `pytest`, pruebas de humo del CLI y verificación del bloqueo Art. 69.7 en cada *push* y *pull request*.
- [`.github/workflows/update_forecast.yml`](.github/workflows/update_forecast.yml): cron diario (`07:00 UTC`) con corte legal LOREG Art. 69.7 y despliegue automático en GitHub Pages desde `docs/`.

---

This is not an official Google product.
