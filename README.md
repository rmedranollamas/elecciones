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
    - **Ensemble calibrado (`ensemble`)**: combinación lineal convexa calibrada mediante backtest LOOCV (`0.05 Kernel + 0.95 Espacio-Estado`).
  - **Etapa 2 — Proyección provincial IPF (`provincial.py`)**:
    - Prior compuesto provincial, matrices de transferencia de coaliciones (`UPN–PP`, `FA`/`Podemos`/`Compromís-FV`/`UPC`/`Ara Més`/`Adelante`, `SALF`, `AC`) estrictamente enmascaradas por $M_{c,p}$, desplazamiento *swing* logit multinomial + rastrillado IPF ponderado por participación, y encogimiento *Empirical Bayes* con encuestas autonómicas/provinciales y el CIS #3584.
  - **Etapa 3 — Simulador Monte Carlo jerárquico Student-$t$ (`covariance.py`, `simulator.py`)**:
    - `50.000` simulaciones en `< 1.5s` (`~1.42s` en CPU estándar) mediante vectorización comprimida sobre celdas activas ($M_{c,p}=1$).
    - Descomposición jerárquica de errores en log-razón (**Nacional + CCAA + Provincial**) con colas pesadas Student-$t$ ($\nu = 4.5$, $\sigma_{\text{nac}}=0.1146$, $\sigma_{\text{CCAA}}=0.0983$, $\sigma_{\text{prov}}=0.0641$), choques correlacionados de bloque izquierda/derecha, covariancia posterior de Etapa 1 y variación estocástica de participación y voto en blanco.
    - Clasificación de escaños en niveles de certidumbre: **Seguro** ($\ge 99\%$), **Probable** ($75\text{–}99\%$), **En juego** ($25\text{–}75\%$), **Remoto** ($5\text{–}25\%$).
  - **Etapa 4 — Investidura Art. 99 CE (`investiture.py`)**:
    - Probabilidades de **1.ª votación** ($\ge 176$ escaños, mayoría absoluta), **2.ª votación** (mayoría simple estricta $\text{Sí} > \text{No}$ con reglas explícitas de abstención) y riesgo de bloqueo institucional / repetición electoral (**Art. 99.5 CE**).
- **Tablero editorial cero-backend (`docs/index.html`)**:
  - Hemiciclo interactivo SVG de 350 escaños, cartograma de las 52 circunscripciones con inspector D'Hondt dinámico (en modo base y bajo escenarios contrafactuales / *"What-If"*), series temporales causales de 100 días, simulador en vivo *"What-If"* sobre las 52 circunscripciones en el navegador y constructor interactivo de pactos de investidura.

---

## Calibración histórica fuera de muestra (LOOCV 2015–2023)

Evaluado de manera estricta fuera de muestra dejando fuera cada elección completa (calibrando sesgos/pesos históricos de forma causal únicamente con convocatorias anteriores; `260` carreras provinciales · `1.750` escaños · horizontes `T-15d`, `T-6d`, `T-1d`):

| Motor | MAE Voto (P / N) | MAE Esc. (P / N) | Brier / LogLoss | Cob. Voto 80/95 (P) | Cob. Voto 80/95 (N) | Cob. Esc. 80/95 (N) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Stage 1B Espacio-Estado** | 2.33 / 1.05 pp | 0.358 / 3.43 | 0.0707 / 0.242 | 75.8% / 85.9% | 85.3% / 97.8% | 84.7% / 92.6% |
| **Ensemble Calibrado** | 2.35 / 1.06 pp | 0.362 / 3.50 | 0.0714 / 0.244 | 75.5% / 85.8% | 83.1% / 97.8% | 85.6% / 91.5% |
| **Stage 1A Kernel** | 2.42 / 1.13 pp | 0.363 / 3.69 | 0.0732 / 0.249 | 76.3% / 88.5% | 80.2% / 96.3% | 82.4% / 92.5% |

*(En el horizonte final `T-1d`, el ensemble alcanza un MAE de **0.91 pp** en voto nacional, **2.15 pp** en voto provincial y **2.84 escaños** nacionales).*

Tasa empírica de acierto por nivel de certidumbre fuera de muestra:
- **Seguro ($\ge 99\%$)** → **99.6%** (`2.123 / 2.131`)
- **Probable ($75\text{–}99\%$)** → **93.7%** (`1.885 / 2.011`)
- **En juego ($25\text{–}75\%$)** → **52.7%** (`1.080 / 2.049`)
- **Remoto ($5\text{–}25\%$)** → **7.2%** (`134 / 1.859`)

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
uv run elecciones forecast --n-sims 50000          # Simulación 50k (~1.4s) -> forecast_29n.json
uv run elecciones forecast --exclude-cis-headline  # Sensibilidad sin CIS en titular
uv run elecciones dashboard                        # Compilar docs/index.html autónomo
uv run elecciones allocate -i distrito.json --json # Reparto LOREG Art. 163 ad-hoc
```

### Añadir, corregir o excluir encuestas manualmente

Edita [`data/polls/manual_polls_2026.csv`](data/polls/manual_polls_2026.csv) (o pasa `--manual-polls /ruta/overlay.csv` a `elecciones forecast`) con filas `action=add`, `action=override` o `action=exclude` para incorporar sondeos nuevos, corregir erratas o descartar olas concretas antes de simular.

### Experimentación, sensibilidad y barridos de parámetros

Consulta la guía completa en **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)** para ejecutar experimentos aislados sin sobrescribir `data/outputs/` ni `docs/index.html`:
- Aislamiento de salidas (`-o /tmp/...`) y *sandboxes* completos con `ELECCIONES_REPO_ROOT`.
- Ablaciones de motor (`--engine ensemble|kernel|state_space`), exclusión de CIS (`--exclude-cis-headline`) y *replays* históricos en el tiempo (`--as-of YYYY-MM-DD`).
- Superposición de sondeos hipotéticos (`--manual-polls`) y horizontes personalizados en LOOCV (`elecciones backtest --horizons ...`).
- Barridos programáticos en Python sobre colas Student-$t$ (`HierarchicalCovarianceConfig`), hiperparámetros de Etapas 1–2 (`ModelHyperparameters`), matrices de transferencia de coaliciones (`coalition_transfers.json`) y pactos de investidura Art. 99 personalizados.

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
- [`.github/workflows/update_forecast.yml`](.github/workflows/update_forecast.yml): cron dos veces al día (`07:17` y `19:17 UTC`) con corte legal LOREG Art. 69.7 y despliegue automático en GitHub Pages desde `docs/`.

---

This is not an official Google product.
