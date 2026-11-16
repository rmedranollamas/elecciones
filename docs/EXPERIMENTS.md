# Running Experiments & Sensitivity Analyses

This guide covers how to run isolated CLI and Python experiments on the **29-N (2026) Congress Statistical Predictor** without overwriting the repository's tracked outputs (`data/outputs/*`, `docs/index.html`) or data snapshots (`data/polls/*`, `data/metadata/*`).

---

## 1. Golden Rule: Output & Workspace Isolation

By default, CLI subcommands write to `data/outputs/` and `docs/index.html`. Use explicit output flags (`-o` / `--output`) or `ELECCIONES_REPO_ROOT` to keep experiments non-destructive.

### Option A — Redirect individual CLI outputs to `/tmp`

```bash
# Run a forecast experiment into /tmp
uv run elecciones forecast \
  --as-of 2026-11-16 \
  --engine state_space \
  --n-sims 20000 \
  --output /tmp/exp_forecast.json

# Compile a standalone HTML dashboard from that experimental bundle
uv run elecciones dashboard \
  --input /tmp/exp_forecast.json \
  --output /tmp/exp_dashboard.html
```

### Option B — Full sandboxed workspace via `ELECCIONES_REPO_ROOT`

Every path helper in [`src/elecciones/config.py`](../src/elecciones/config.py) resolves relative to `ELECCIONES_REPO_ROOT` when set. You can clone `data/` (excluding `raw_cache/`) into a scratch directory to freely edit `parties_2026.json`, `coalition_transfers.json`, or `regional_2026.csv` and run `elecciones run-all --offline`:

```bash
EXP_DIR=/tmp/elecciones_sandbox
mkdir -p "$EXP_DIR"
ln -sf "$PWD/pyproject.toml" "$EXP_DIR/pyproject.toml"
rsync -a --exclude='raw_cache' data/ "$EXP_DIR/data/"

ELECCIONES_REPO_ROOT="$EXP_DIR" uv run elecciones run-all \
  --offline \
  --no-respect-blackout \
  --as-of 2026-11-16 \
  --backtest-sims 200 \
  --n-sims 10000
# Outputs land in $EXP_DIR/data/outputs/ and $EXP_DIR/docs/index.html
```

---

## 2. CLI Experiment Recipes

### 2.1 Stage 1 Engine & CIS Headline Ablations

Compare the three Stage 1 latent-support engines (`ensemble`, `kernel`, `state_space`) with and without CIS in the national headline aggregator:

```bash
for eng in ensemble kernel state_space; do
  for cis_flag in --no-exclude-cis-headline --exclude-cis-headline; do
    tag="${eng}_${cis_flag#--}"
    uv run elecciones forecast \
      --as-of 2026-11-16 \
      --engine "$eng" \
      "$cis_flag" \
      --n-sims 20000 \
      --output "/tmp/fc_${tag}.json"
  done
done
```

> **Note:** `--exclude-cis-headline` zeros CIS polls in Stage 1 national headline aggregation (`kernel` and `state_space`) while preserving the nationally de-biased CIS #3584 provincial cross-tab (`data/polls/cis_3584_provincial.csv`) for sub-national geography in Stage 2.

### 2.2 Historical Point-in-Time ("Time-Travel") Campaign Replays

Replay the 29-N forecast as it stood on any date using `--as-of YYYY-MM-DD`. Both Stage 1 and Stage 2 strictly filter polls to `fieldwork_end <= as_of` (and `pub_date <= as_of` when present) and propagate random-walk uncertainty across the remaining days to `2026-11-29`:

```bash
for cutoff in 2026-09-15 2026-10-01 2026-10-15 2026-11-01 2026-11-16; do
  uv run elecciones forecast \
    --as-of "$cutoff" \
    --n-sims 20000 \
    --output "/tmp/fc_${cutoff}.json"
done
```

### 2.3 Monte Carlo Precision & PRNG Seed Checks

- Rapid iteration (`~0.15s`): `--n-sims 5000`
- Default production (`~1.4s`): `--n-sims 50000 --seed 20261129`
- High-precision tail probabilities (`~5.5s`): `--n-sims 200000 --seed 42`

### 2.4 Out-of-Sample LOOCV Horizon & Calibration Experiments

Re-run leave-one-election-out calibration (`2015-12` through `2023-07`, `260` constituency races, strictly causal pollster ratings) across custom pre-election horizons:

```bash
uv run elecciones backtest \
  --horizons "30,15,6,1" \
  --n-sims 300 \
  --seed 20261129 \
  --output /tmp/bt_custom_horizons.json
```

### 2.5 Ad-Hoc LOREG Art. 163 Apportionment (`elecciones allocate`)

Test exact provincial D'Hondt thresholds (`3%` over `sum(party_votes) + blank_votes`, `0%` plurality in Ceuta `51` / Melilla `52`, and Art. 163.1.d total-vote tie-breaking) from JSON files or `stdin`:

```bash
uv run elecciones allocate --json -i - <<'EOF'
{
  "cpro": "28",
  "seats": 38,
  "blank_votes": 28500,
  "party_votes": {
    "PP": 1410000,
    "PSOE": 985000,
    "VOX": 620000,
    "FA": 365000,
    "PODEMOS": 112000,
    "SALF": 98000
  }
}
EOF
```

For a multi-district chamber payload, pass `{"constituencies": [ ... ]}`.

---

## 3. Hypothetical Poll Overlays (`--manual-polls`)

You can inject new polls, override existing rows, or exclude outlier waves without editing `data/polls/national_2026.csv` by passing `--manual-polls /path/to/overlay.csv` to `elecciones forecast`.

### Schema & Scale Rules

1. **26 header columns** (defined in [`src/elecciones/ingest/cross_check.py`](../src/elecciones/ingest/cross_check.py)):
   ```text
   action,election_id,pollster,media,fieldwork_start,fieldwork_end,pub_date,sample_size,turnout,is_cis,PP,PSOE,VOX,FA,COMPROMIS_FV,FA_FV,PODEMOS,SALF,ERC,JUNTS,EH_BILDU,PNV,BNG,CCA,ADELANTE,UPN_PP
   ```
2. **Percentage-point scale (`0–100`)**: CSV party cells are written in percentage points (e.g. `31.4` for `31.4%`), **never** unit fractions (`0.314`).
3. **Missing vs. zero**: Leave unreported regional/minor parties **blank** (`""`). Do **not** fill unreported parties with `0`, which would record an explicit `0.0%` observation.
4. **Supported `action` values**:
   - `add`: appends a new poll (or replaces an exact match on `(election_id, pollster, fieldwork_start, fieldwork_end)`).
   - `override`: replaces a matching poll in `national_2026.csv` (inserts if no match exists).
   - `exclude`: drops any matching poll from `national_2026.csv` (party columns may be left empty).

### Example: Simulating a Late Campaign Shock

```bash
cat <<'EOF' > /tmp/late_shock_polls.csv
action,election_id,pollster,media,fieldwork_start,fieldwork_end,pub_date,sample_size,turnout,is_cis,PP,PSOE,VOX,FA,COMPROMIS_FV,FA_FV,PODEMOS,SALF,ERC,JUNTS,EH_BILDU,PNV,BNG,CCA,ADELANTE,UPN_PP
add,2026-11,40dB.,El País,2026-11-17,2026-11-19,2026-11-20,2000,69.5,false,32.0,28.5,17.2,7.4,1.3,,2.1,1.6,2.4,1.4,1.4,0.9,1.0,0.4,1.5,
exclude,2026-11,CIS,,2026-10-07,2026-10-30,,,,false,,,,,,,,,,,,,,,,
EOF

uv run elecciones forecast \
  --as-of 2026-11-20 \
  --manual-polls /tmp/late_shock_polls.csv \
  --n-sims 50000 \
  --output /tmp/fc_late_shock.json
```

---

## 4. Programmatic Python Experiments (Stages 1–4)

For parameter sweeps, custom covariance structures, coalition-transfer counterfactuals, or custom Art. 99 investiture pacts, invoke the Python API directly with `uv run python`.

> **Important Calibration Override Gotcha:**
> [`run_forecast_simulation()`](../src/elecciones/models/simulator.py) calls [`load_calibrated_config()`](../src/elecciones/backtest/runner.py) on `data/outputs/backtest_summary.json` by default. That file overrides `ensemble_weight_kernel`, `ensemble_weight_state_space`, `student_t_df`, and the `HierarchicalCovarianceConfig` shock scales.
> - To test custom Stage 3 shock/tail parameters in `run_forecast_simulation()`, pass `cov_config_override=HierarchicalCovarianceConfig(...)`.
> - To test custom `ModelHyperparameters` without `backtest_summary.json` overriding ensemble weights, either call the stage functions (`run_stage1`, `project_2026_provinces`, `run_hierarchical_monte_carlo`) directly or point `backtest_summary_path` at a non-existent path (e.g. `Path("/tmp/none.json")`).

### 4.1 Sweeping Student-t Tail Heaviness & National/Regional Shock Scales

```python
from datetime import date
from elecciones.models.covariance import HierarchicalCovarianceConfig
from elecciones.models.simulator import run_forecast_simulation

for df, nat_sd, label in [
    (4.5, 0.1146, "calibrated_baseline"),
    (3.0, 0.1400, "heavy_tails_high_volatility"),
    (30.0, 0.0800, "near_gaussian_tight"),
]:
    cov = HierarchicalCovarianceConfig(
        student_t_df=df,
        nat_scale=nat_sd,
        ccaa_scale=0.0983,
        prov_scale=0.0641,
        bloc_right_scale=0.0213,
        bloc_left_scale=0.0213,
    )
    bundle = run_forecast_simulation(
        as_of=date(2026, 11, 16),
        n_sims=20_000,
        cov_config_override=cov,
        write_output=False,
    )
    pp = bundle.national["seats"]["PP"]
    pv = bundle.investiture.first_ballot["pp_vox"]
    print(
        f"{label:28s} | PP med={pp['median']} ({pp['p10']}-{pp['p90']}) "
        f"| P(PP+Vox>=176)={pv.prob:.1%}"
    )
```

### 4.2 Stage 1 & Stage 2 Hyperparameter Sensitivity (Recency, CIS Cap, Shrinkage)

All Stage 1–2 knobs live on [`ModelHyperparameters`](../src/elecciones/config.py):

| Parameter | Default | Role |
| :--- | ---: | :--- |
| `half_life_campaign_days` | `7.5` | Kernel & Kish `n_eff` exponential half-life within 60 days of election |
| `half_life_quiet_days` | `14.0` | Exponential half-life > 60 days before election |
| `sample_size_cap_general` | `4000` | Maximum effective `N` for non-CIS polls |
| `sample_size_cap_cis` | `2000` | Maximum effective `N` for CIS macro-surveys |
| `state_space_rw_var_per_day` | `0.0014` | Daily ALR random-walk innovation variance (governs blackout drift) |
| `state_space_obs_scale` | `0.045` | Non-sampling observation noise floor in Stage 1B |
| `stage2_prior_weight` | `900.0` | Stage 2 Dirichlet prior pseudocount `n0` per constituency |
| `stage2_cis_max_frac` | `0.30` | Cap on CIS #3584 provincial pseudocount relative to `n0` |
| `stage2_regional_divisor` | `1.5` | Divisor scaling effective `N` of 2026 CCAA/provincial polls |

Example comparing Stage 2 **with** vs. **without** CIS #3584 & regional poll Empirical-Bayes shrinkage:

```python
from dataclasses import replace
from datetime import date
from elecciones.config import DEFAULT_HYPERPARAMETERS, ELECTION_DATE_2026, get_polls_dir
from elecciones.ingest.wikipedia import load_national_polls
from elecciones.models.covariance import HierarchicalCovarianceConfig, run_hierarchical_monte_carlo
from elecciones.models.ensemble import run_stage1
from elecciones.models.pollster_ratings import compute_pollster_ratings
from elecciones.models.provincial import project_2026_provinces
from elecciones.schemas import load_constituencies_catalog, load_parties_catalog

as_of = date(2026, 11, 16)
hp = replace(
    DEFAULT_HYPERPARAMETERS,
    half_life_campaign_days=5.0,
    ensemble_weight_kernel=0.05,
    ensemble_weight_state_space=0.95,
)
polls = load_national_polls(get_polls_dir() / "national_2026.csv")
ratings = compute_pollster_ratings(polls_2026=polls, hyperparams=hp, as_of=as_of)
s1 = run_stage1(
    polls,
    as_of,
    engine="ensemble",
    target_election_date=ELECTION_DATE_2026,
    ratings=ratings,
    hyperparams=hp,
)
consts = load_constituencies_catalog()
parties = load_parties_catalog()

for shrink in (True, False):
    s2 = project_2026_provinces(
        s1.mean_shares,
        as_of,
        blank_nat=s1.blank_share,
        constituencies=consts,
        parties=parties,
        ratings=ratings,
        hyperparams=hp,
        apply_shrinkage=shrink,
    )
    sim = run_hierarchical_monte_carlo(
        s2,
        stage1_result=s1,
        parties_catalog=parties,
        config=HierarchicalCovarianceConfig(
            student_t_df=4.5, nat_scale=0.1146, ccaa_scale=0.0983, prov_scale=0.0641
        ),
        hyperparams=hp,
        n_sims=20_000,
    )
    means = dict(zip(sim.party_ids, sim.national_seats.mean(axis=0).round(2), strict=True))
    print(f"apply_shrinkage={str(shrink):5s} | PP={means['PP']} PSOE={means['PSOE']} ERC={means['ERC']} JUNTS={means['JUNTS']}")
```

### 4.3 Coalition Transfer & Ballot Mask Counterfactuals

[`project_2026_provinces()`](../src/elecciones/models/provincial.py) accepts an in-memory `transfers` dictionary (modeled on [`data/metadata/coalition_transfers.json`](../data/metadata/coalition_transfers.json)) and a `PartiesCatalog` (modeled on [`data/metadata/parties_2026.json`](../data/metadata/parties_2026.json)):

```python
import copy
from datetime import date
from elecciones.config import get_metadata_dir
from elecciones.ingest.cis import load_coalition_transfers
from elecciones.models.provincial import project_2026_provinces

base_transfers = load_coalition_transfers(get_metadata_dir() / "coalition_transfers.json")
custom_transfers = copy.deepcopy(base_transfers)

# Counterfactual: higher 2023 Sumar -> FA retention in Andalusia & general territory
custom_transfers["sumar_2023_split"]["andalusia"] = {
    "FA": 0.55, "ADELANTE": 0.15, "PODEMOS": 0.10, "PSOE": 0.15, "OTROS": 0.05
}
custom_transfers["navarra_upn_pp_merger"]["consolidation_retention"] = 0.82

s2_cf = project_2026_provinces(
    {"PP": 0.3096, "PSOE": 0.2685, "VOX": 0.1905, "FA": 0.0610, "PODEMOS": 0.0183},
    date(2026, 11, 16),
    transfers=custom_transfers,
)
print("Navarre UPN_PP share:", round(s2_cf.districts["31"].final_shares["UPN_PP"] * 100, 2), "%")
print("Seville FA share:", round(s2_cf.districts["41"].final_shares["FA"] * 100, 2), "%")
```

### 4.4 Custom Art. 99 Investiture Coalitions (`1st` & `2nd` Ballot)

Evaluate arbitrary 1st-ballot (`>= 176`) and 2nd-ballot (`Sí > No` with explicit abstentions) pacts directly on simulated seat draws via [`compute_investiture_summary()`](../src/elecciones/models/investiture.py):

```python
import json
import numpy as np
from elecciones.models.investiture import compute_investiture_summary

# Using the 200-draw sample embedded in forecast_29n.json (or sim.national_seats from 50k draws)
with open("data/outputs/forecast_29n.json", encoding="utf-8") as fh:
    fc = json.load(fh)
sample_rows = fc["sim_sample"]
party_ids = tuple(sample_rows[0]["seats"].keys())
seats_mat = np.asarray(
    [[row["seats"].get(p, 0) for p in party_ids] for row in sample_rows],
    dtype=np.int32,
)

inv = compute_investiture_summary(
    seats_mat,
    party_ids,
    first_ballot_specs=[
        ("pp_pnv_cca_junts", "PP + UPN + PNV + CCa + Junts (>=176)", ("PP", "UPN_PP", "PNV", "CCA", "JUNTS")),
    ],
    second_ballot_specs=[
        {
            "key": "psoe_fa_erc_bildu_pnv_bng_abst_vox_never",
            "label": "PSOE + socios (abstención Junts, Podemos y CCa)",
            "yes": ("PSOE", "FA", "COMPROMIS_FV", "UPC", "ARA_MES", "ADELANTE", "ERC", "EH_BILDU", "PNV", "BNG"),
            "abstain": ("JUNTS", "PODEMOS", "CCA"),
        }
    ],
)
print("1st ballot:", inv.first_ballot["pp_pnv_cca_junts"].to_dict())
print("2nd ballot:", inv.second_ballot["psoe_fa_erc_bildu_pnv_bng_abst_vox_never"].to_dict())
```

---

## 5. Browser "What-If" Simulator vs. Python Monte Carlo

The standalone dashboard (`docs/index.html`) includes live interactive controls (`Ensemble / Kernel / Espacio-Estado`, `Con CIS / Sin CIS`, national swing sliders, and the Art. 99 Pact Builder). Use the right tool depending on what quantity you need:

| Question / Output Needed | Browser `docs/index.html` (Instant) | Python `elecciones forecast` (`~1.4s`) |
| :--- | :---: | :---: |
| Exact 52-district LOREG Art. 163 D'Hondt seats under national vote shifts | Yes | Yes |
| Updated provincial D'Hondt quotient tables & last-seat tipping-point margins | Yes | Yes |
| 350-seat hemicycle & deterministic coalition seat totals | Yes | Yes |
| Full `50,000`-draw `P10–P90` seat intervals & `Seguro / Probable / En juego` probabilities | Approximated / base MC | **Exact** |
| Re-estimated Jackman house effects, state-space filter, or regional EB shrinkage | Precomputed endpoints | **Exact** |
| Monte Carlo Art. 99 1st/2nd-ballot & Art. 99.5 deadlock probabilities | Base MC (`sim_sample`) | **Exact** |
