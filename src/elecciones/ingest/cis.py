"""CIS Preelectoral Study #3584 (N=45,074) parser, IPF de-biasing, and coalition transfer priors.

Produces:
- ``data/polls/cis_3584_provincial.csv``: 52-district raw (BOE-masked) and nationally
  IPF de-biased CIS #3584 shares across all 27 canonical parties (unit shares summing to 1.0).
- ``data/metadata/coalition_transfers.json``: 2023 -> 2026 recall-of-vote transfer matrix
  (Sumar breakup into FA/Podemos/Compromís-FV/UPC/Ara Més/Adelante, Navarra PP+UPN -> UPN-PP,
  SALF 2024 EP provincial anchor, and unproclaimed-list notes).
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import openpyxl

from elecciones.config import (
    get_elections_dir,
    get_metadata_dir,
    get_polls_dir,
    get_raw_cache_dir,
    make_http_client,
)
from elecciones.ingest.elections import resolve_cpro_from_name, strip_accents
from elecciones.schemas import (
    ALL_CPROS,
    ConstituenciesCatalog,
    PartiesCatalog,
    PollRecord,
    SchemaValidationError,
    load_constituencies_catalog,
    load_parties_catalog,
)

CIS_STUDY_ID: Final[str] = "3584"
CIS_AVANCE_INDEX_URL: Final[str] = "https://www.cis.es/es/avance-de-resultados"
CIS_FALLBACK_XLSX_URL: Final[str] = (
    "https://www.cis.es/documents/d/cis/3584_multi_a-xlsx"
)

# Row labels in CIS #3584 Provincia / RV EG23 P5aR -> canonical 27-party IDs
CIS_LABEL_TO_PARTY: Final[dict[str, str]] = {
    "psoe": "PSOE",
    "pp": "PP",
    "vox": "VOX",
    "frente amplio (cha-fa en aragon y comuns-fa en cataluna)": "FA",
    "frente amplio": "FA",
    "erc": "ERC",
    "erc-podemos**": "ERC",
    "eh bildu": "EH_BILDU",
    "eh-bildu": "EH_BILDU",
    "se acabo la fiesta": "SALF",
    "salf": "SALF",
    "podemos": "PODEMOS",
    "compromis: front valencia": "COMPROMIS_FV",
    "adelante andalucia": "ADELANTE",
    "bng": "BNG",
    "junts": "JUNTS",
    "eaj-pnv": "PNV",
    "upc (unidad del pueblo canario)": "UPC",
    "upc": "UPC",
    "upn-pp": "UPN_PP",
    "pacma": "PACMA",
    "ara mes-front ampli": "ARA_MES",
    "coalicion canaria-municipalistas primero canarias": "CCA",
    "cc-municipalistas primero canarias": "CCA",
    "frente obrero": "FO",
    "falange espanola de las jons": "OTROS",
    "democracia 21": "OTROS",
    "soria ya": "SORIA_YA",
    "teruel existe": "TERUEL_EXISTE",
    "geroa bai": "GEROA_BAI",
    "escanos en blanco": "OTROS",
    "drago canarias verdes": "OTROS",
    "otro partido": "OTROS",
    "otros partidos": "OTROS",
    "en blanco": "BLANK",
}


def _norm_cis_label(raw: Any) -> str:
    txt = strip_accents(str(raw or "")).lower().strip()
    return re.sub(r"\s+", " ", txt)


def ensure_cis_3584_xlsx(
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
) -> Path:
    """Return local cached path for CIS #3584 XLSX, fetching from cis.es if needed."""
    cache_dir = raw_cache_dir or get_raw_cache_dir()
    target = cache_dir / "cis_3584_multi_A.xlsx"
    if target.is_file() and target.stat().st_size > 0:
        return target
    if offline:
        raise FileNotFoundError(
            f"Missing cached CIS #3584 workbook {target} in offline mode."
        )

    download_url: str | None = None
    with make_http_client(timeout=60.0) as client:
        try:
            idx_resp = client.get(CIS_AVANCE_INDEX_URL)
            idx_resp.raise_for_status()
            m = re.search(
                r'href="([^"]*3584[^"]*multi[^"]*\.xlsx[^"]*|[^"]*3584_multi[^"]*)"',
                idx_resp.text,
                flags=re.I,
            )
            if m:
                href = m.group(1)
                download_url = (
                    href
                    if href.startswith("http")
                    else f"https://www.cis.es{href}"
                )
        except Exception:
            download_url = None

        url_to_try = download_url or CIS_FALLBACK_XLSX_URL
        resp = client.get(url_to_try)
        resp.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)
    return target


def parse_cis_3584_weights(wb: openpyxl.Workbook) -> dict[str, dict[str, Any]]:
    """Parse provincial sample weights (Cuadro 1) from ``Tabla ponderaciones``."""
    ws = wb["Tabla ponderaciones"]
    weights: dict[str, dict[str, Any]] = {}
    in_cuadro1 = False
    for row in ws.iter_rows(min_row=1, max_row=75, values_only=True):
        c0 = str(row[0] or "").strip() if len(row) > 0 else ""
        c1 = str(row[1] or "").strip() if len(row) > 1 else ""
        if "Cuadro 2" in c0 or "Cuadro 2" in c1:
            break
        if c0 == "Código" and "Provincia" in c1:
            in_cuadro1 = True
            continue
        if in_cuadro1 and c0.isdigit() and 1 <= int(c0) <= 52:
            cpro = f"{int(c0):02d}"
            weights[cpro] = {
                "province": c1,
                "n_designed": int(round(float(row[2] or 0))),
                "n_realized": int(round(float(row[3] or 0))),
                "pond_weight": float(row[4] or 1.0),
                "error_pct": float(row[5] or 0.0),
            }
    if len(weights) != 52:
        raise SchemaValidationError(
            f"CIS #3584 Tabla ponderaciones parsed {len(weights)} districts != 52"
        )
    return weights


def parse_cis_3584_headline(wb: openpyxl.Workbook) -> dict[str, dict[str, float]]:
    """Parse national ``Estimación de Voto`` & direct intention from CIS #3584."""
    ws = wb["Estimación de Voto"]
    direct_census_pct: dict[str, float] = {}
    estimacion_valid_pct: dict[str, float] = {}
    for row in ws.iter_rows(min_row=8, max_row=40, values_only=True):
        label = str(row[0] or "").strip() if len(row) > 0 else ""
        if not label:
            continue
        norm = _norm_cis_label(label)
        pid = CIS_LABEL_TO_PARTY.get(norm)
        if pid is None:
            continue
        v_dir = row[1] if len(row) > 1 else None
        v_est = row[3] if len(row) > 3 else None
        if isinstance(v_dir, (int, float)):
            direct_census_pct[pid] = direct_census_pct.get(pid, 0.0) + float(v_dir)
        if isinstance(v_est, (int, float)):
            estimacion_valid_pct[pid] = estimacion_valid_pct.get(pid, 0.0) + float(v_est)

    if "PP" not in estimacion_valid_pct or "PSOE" not in estimacion_valid_pct:
        raise SchemaValidationError("CIS #3584 Estimación de Voto missing PP/PSOE")
    return {
        "direct_census_pct": direct_census_pct,
        "estimacion_valid_pct": estimacion_valid_pct,
    }


def _find_cis_question_block(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    header_hint: str,
) -> tuple[list[str], list[tuple[str, list[float]]], list[float]]:
    """Locate ``Pregunta 5aR`` table on ``Provincia`` or ``RV EG23`` sheet."""
    all_rows = list(ws.iter_rows(min_row=1, values_only=True))
    p5_start: int | None = None
    for idx, row in enumerate(all_rows):
        txt = " ".join(str(x or "") for x in row[:3])
        if "Pregunta 5aR" in txt:
            p5_start = idx
            break
    if p5_start is None:
        # Fallback: scan for row whose column 1 equals Araba/Álava or TOTAL
        for idx, row in enumerate(all_rows):
            c1 = str(row[1] or "").strip() if len(row) > 1 else ""
            if c1 in {"Araba/Álava", "TOTAL"} and idx > 300:
                p5_start = idx - 2
                break
    if p5_start is None:
        raise SchemaValidationError(f"Could not locate Pregunta 5aR block ({header_hint})")

    # Find header row containing columns (next non-empty row after p5_start with >=10 cells)
    hdr_idx: int | None = None
    for j in range(p5_start, min(len(all_rows), p5_start + 15)):
        non_empty = sum(1 for x in all_rows[j][1:] if x is not None and str(x).strip() != "")
        if non_empty >= 10:
            hdr_idx = j
            break
    if hdr_idx is None:
        raise SchemaValidationError(f"Could not locate column headers near row {p5_start}")

    col_names = [str(x).strip() for x in all_rows[hdr_idx][1:] if x is not None and str(x).strip()]
    n_cols = len(col_names)

    party_rows: list[tuple[str, list[float]]] = []
    n_row: list[float] = []
    for k in range(hdr_idx + 1, min(len(all_rows), hdr_idx + 50)):
        r = all_rows[k]
        lbl = str(r[0] or "").strip()
        if not lbl:
            if party_rows:
                break
            continue
        if lbl == "(N)":
            n_row = [float(r[1 + c] or 0.0) for c in range(n_cols)]
            break
        vals = [float(r[1 + c] or 0.0) for c in range(n_cols)]
        party_rows.append((lbl, vals))

    if not party_rows or not n_row:
        raise SchemaValidationError(
            f"Incomplete P5aR block ({header_hint}): rows={len(party_rows)}, n_row={len(n_row)}"
        )
    return col_names, party_rows, n_row


def parse_cis_3584_provincial(
    wb: openpyxl.Workbook,
    parties: PartiesCatalog,
) -> dict[str, dict[str, Any]]:
    """Extract raw BOE-masked unit shares and weighted sample counts per province."""
    ws = wb["Provincia"]
    col_names, party_rows, n_row = _find_cis_question_block(ws, "Araba")
    if len(col_names) != 52:
        raise SchemaValidationError(
            f"Expected 52 province columns in CIS Provincia P5aR, got {len(col_names)}"
        )

    col_cpros = [resolve_cpro_from_name(name) for name in col_names]
    if set(col_cpros) != set(ALL_CPROS):
        raise SchemaValidationError("CIS Provincia P5aR does not cover all 52 cpros")

    party_ids = parties.party_ids()
    mask_map = {p.id: p.boe_mask_2026 for p in parties.parties}

    result: dict[str, dict[str, Any]] = {}
    for j, cpro in enumerate(col_cpros):
        raw_pct: dict[str, float] = {pid: 0.0 for pid in party_ids}
        for label, vals in party_rows:
            norm = _norm_cis_label(label)
            pid = CIS_LABEL_TO_PARTY.get(norm, "OTROS")
            if pid == "BLANK" or pid not in raw_pct:
                continue
            raw_pct[pid] += float(vals[j])

        # Enforce strict BOE candidacy mask (e.g., PACMA/OTROS zeroed where not proclaimed)
        masked_shares = {
            pid: (raw_pct[pid] / 100.0 if mask_map[pid][cpro] == 1 else 0.0)
            for pid in party_ids
        }
        tot = sum(masked_shares.values())
        if tot <= 0.0:
            raise SchemaValidationError(f"Zero total masked CIS share in cpro={cpro}")
        norm_shares = {pid: v / tot for pid, v in masked_shares.items()}

        result[cpro] = {
            "weighted_n": float(n_row[j]),
            "raw_shares": norm_shares,
        }

    return result


def compute_non_cis_national_consensus(
    polls: Sequence[PollRecord],
    parties: PartiesCatalog,
    *,
    as_of: date = date(2026, 11, 23),
    window_days: int = 75,
    half_life_days: float = 12.0,
) -> dict[str, float]:
    """Compute a clean non-CIS consensus across polled parties on ``national_2026.csv``."""
    eligible = [
        p
        for p in polls
        if not p.is_cis
        and p.fieldwork_end <= as_of
        and (as_of - p.fieldwork_end).days <= window_days
    ]
    if not eligible:
        eligible = [p for p in polls if not p.is_cis and p.fieldwork_end <= as_of]

    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    for poll in eligible:
        age = max(0, (as_of - poll.fieldwork_end).days)
        recency = 0.5 ** (age / half_life_days)
        n_eff = float(min(poll.sample_size or 1000, 4000)) ** 0.5
        wt = recency * (n_eff / 31.6227766)
        for pid, val in poll.party_shares.items():
            if pid == "FA_FV":
                continue
            sums[pid] = sums.get(pid, 0.0) + wt * val
            weights[pid] = weights.get(pid, 0.0) + wt

    valid_ids = set(parties.party_ids())
    consensus: dict[str, float] = {}
    for pid, s_val in sums.items():
        if pid in valid_ids and weights.get(pid, 0.0) > 0.0:
            consensus[pid] = s_val / weights[pid]
    return consensus


def debias_cis_provincial_ipf(
    raw_by_cpro: Mapping[str, Mapping[str, Any]],
    constituencies: ConstituenciesCatalog,
    parties: PartiesCatalog,
    national_consensus: Mapping[str, float] | None = None,
    *,
    max_iters: int = 120,
    tol: float = 1e-11,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """De-bias CIS #3584 52-province shares via masked IPF to match non-CIS consensus.

    Strictly preserves ``M_{c,p} == 0 => 0.0`` and guarantees ``sum_p debias[c, p] == 1.0``
    in every constituency as well as electorate-weighted national alignment with
    ``national_consensus``.
    """
    cpros = ALL_CPROS
    party_ids = list(parties.party_ids())
    c_idx = {c: i for i, c in enumerate(cpros)}
    p_idx = {p: j for j, p in enumerate(party_ids)}

    consts_map = constituencies.by_cpro()
    validos = np.array(
        [float(consts_map[c].validos_2023 or 1) for c in cpros], dtype=np.float64
    )
    w_c = validos / validos.sum()

    mask = np.array(parties.mask_matrix(cpros), dtype=np.float64)
    X = np.zeros((len(cpros), len(party_ids)), dtype=np.float64)
    for c in cpros:
        i = c_idx[c]
        raw_s = raw_by_cpro[c]["raw_shares"]
        for p in party_ids:
            j = p_idx[p]
            if mask[i, j] == 1.0:
                X[i, j] = max(0.0, float(raw_s.get(p, 0.0)))

    # Normalize initial X rows
    row_sums = X.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise SchemaValidationError("Zero initial row sum in CIS IPF matrix")
    X /= row_sums

    # Raw national electorate-weighted average over active parties
    raw_nat = (w_c[:, None] * X).sum(axis=0)

    # Construct target national marginals over all 27 parties (with sum(target) == 1.0)
    cons = dict(national_consensus or {})
    target = raw_nat.copy()

    # Keep micro-parties unpolled nationally at their empirical CIS baseline
    # while anchoring national & main regional polled parties to consensus.
    pp_upn_raw = raw_nat[p_idx["PP"]] + raw_nat[p_idx["UPN_PP"]]
    if "PP" in cons and cons["PP"] > 0.0 and pp_upn_raw > 0.0:
        pp_share_of_duo = raw_nat[p_idx["PP"]] / pp_upn_raw
        target[p_idx["PP"]] = cons["PP"] * pp_share_of_duo
        target[p_idx["UPN_PP"]] = cons["PP"] * (1.0 - pp_share_of_duo)

    direct_override_parties = (
        "PSOE",
        "VOX",
        "FA",
        "COMPROMIS_FV",
        "PODEMOS",
        "ADELANTE",
        "SALF",
        "ERC",
        "JUNTS",
        "EH_BILDU",
        "PNV",
        "BNG",
        "CCA",
    )
    for pid in direct_override_parties:
        if pid in cons and cons[pid] > 0.0 and raw_nat[p_idx[pid]] > 0.0:
            target[p_idx[pid]] = float(cons[pid])

    # Zero any party that has 0 BOE support nationally (e.g. AC, CUP, UPL, ALHORA if absent in CIS)
    for j in range(len(party_ids)):
        if raw_nat[j] <= 0.0:
            target[j] = 0.0

    # Rescale un-overridden micro-parties slightly so sum(target) == 1.0 cleanly,
    # keeping overridden polled parties exact up to minor residual renormalization.
    overridden = {p_idx[p] for p in ("PP", "UPN_PP", *direct_override_parties) if p in p_idx}
    overridden = {j for j in overridden if target[j] > 0.0}
    other_idx = [j for j in range(len(party_ids)) if j not in overridden and target[j] > 0.0]
    over_sum = float(target[list(overridden)].sum()) if overridden else 0.0
    other_sum = float(target[other_idx].sum()) if other_idx else 0.0
    if other_idx and other_sum > 0.0 and over_sum < 0.99:
        scale_other = (1.0 - over_sum) / other_sum
        for j in other_idx:
            target[j] *= scale_other
    else:
        target /= target.sum()

    # Iterative Proportional Fitting (masked)
    active_cols = target > 0.0
    for _ in range(max_iters):
        cur_nat = (w_c[:, None] * X).sum(axis=0)
        col_scale = np.ones(len(party_ids), dtype=np.float64)
        valid = active_cols & (cur_nat > 0.0)
        col_scale[valid] = target[valid] / cur_nat[valid]
        X_next = X * col_scale[None, :]
        X_next *= mask
        X_next /= X_next.sum(axis=1, keepdims=True)
        diff = float(np.max(np.abs(X_next - X)))
        X = X_next
        if diff < tol:
            break

    # Final exact float64 row normalization and zero-mask enforcement
    X *= mask
    X /= X.sum(axis=1, keepdims=True)

    debias_map: dict[str, dict[str, float]] = {}
    for i, c in enumerate(cpros):
        debias_map[c] = {
            party_ids[j]: (0.0 if mask[i, j] == 0.0 else float(X[i, j]))
            for j in range(len(party_ids))
        }

    achieved_nat = (w_c[:, None] * X).sum(axis=0)
    achieved_map = {party_ids[j]: float(achieved_nat[j]) for j in range(len(party_ids))}
    return debias_map, achieved_map


def parse_cis_3584_rv_eg23(wb: openpyxl.Workbook) -> dict[str, Any]:
    """Parse ``RV EG23`` P5aR transfer matrix across 2023 Congress origin blocs."""
    ws = wb["RV EG23"]
    col_names, party_rows, n_row = _find_cis_question_block(ws, "TOTAL")

    origin_totals: dict[str, float] = {
        col: float(n_row[idx]) for idx, col in enumerate(col_names)
    }
    raw_matrix: dict[str, dict[str, float]] = {col: {} for col in col_names}

    for label, vals in party_rows:
        norm = _norm_cis_label(label)
        dest_pid = CIS_LABEL_TO_PARTY.get(norm, "OTROS")
        if dest_pid == "BLANK":
            continue
        for idx, origin in enumerate(col_names):
            raw_matrix[origin][dest_pid] = raw_matrix[origin].get(dest_pid, 0.0) + float(
                vals[idx]
            )

    # Normalize each origin's observed 2026 destinations onto simplex
    norm_matrix: dict[str, dict[str, float]] = {}
    for origin, dests in raw_matrix.items():
        tot = sum(dests.values())
        if tot > 0.0:
            norm_matrix[origin] = {
                k: round(v / tot, 6) for k, v in dests.items() if v > 0.0
            }
        else:
            norm_matrix[origin] = {}

    return {
        "origin_weighted_n": origin_totals,
        "raw_transition_pct": raw_matrix,
        "normalized_transitions": norm_matrix,
    }


def build_coalition_transfer_priors(
    rv_data: Mapping[str, Any],
    headline: Mapping[str, Any],
    europarl_csv: Path,
    parties: PartiesCatalog,
) -> dict[str, Any]:
    """Construct validated ``data/metadata/coalition_transfers.json``."""
    sumar_norm = rv_data["normalized_transitions"].get("Sumar", {})

    # Extract SALF 2024 EP provincial share baseline
    salf_ep_shares: dict[str, float] = {}
    salf_ep_nat = 0.0
    if europarl_csv.is_file():
        with europarl_csv.open("r", encoding="utf-8", newline="") as fh:
            tot_val = 0
            tot_salf = 0
            for row in csv.DictReader(fh):
                cpro = str(row["cpro"]).zfill(2)
                val = int(row["validos"])
                sv = int(row.get("votes_SALF", 0) or 0)
                salf_ep_shares[cpro] = sv / val if val > 0 else 0.0
                tot_val += val
                tot_salf += sv
            salf_ep_nat = tot_salf / tot_val if tot_val > 0 else 0.0

    upn_nav_norm = rv_data["normalized_transitions"].get("UPN", {})
    pp_norm = rv_data["normalized_transitions"].get("PP", {})

    # Geographic conditional allocation weights for Sumar 2023 voters across 29-N zones
    sumar_split_zones: dict[str, dict[str, float]] = {
        "national_cis_3584": dict(sumar_norm),
        "valencian_community": {
            "COMPROMIS_FV": 0.68,
            "PSOE": 0.25,
            "VOX": 0.02,
            "PP": 0.02,
            "OTROS": 0.03,
        },
        "canary_islands": {
            "UPC": 0.62,
            "PSOE": 0.26,
            "CCA": 0.05,
            "VOX": 0.02,
            "OTROS": 0.05,
        },
        "illes_balears": {
            "ARA_MES": 0.62,
            "PSOE": 0.28,
            "VOX": 0.02,
            "PP": 0.02,
            "OTROS": 0.06,
        },
        "catalonia": {
            "FA": 0.56,
            "ERC": 0.20,
            "PSOE": 0.20,
            "OTROS": 0.04,
        },
        "andalusia": {
            "FA": 0.34,
            "ADELANTE": 0.25,
            "PODEMOS": 0.14,
            "PSOE": 0.22,
            "OTROS": 0.05,
        },
        "general_territory": {
            "FA": 0.51,
            "PODEMOS": 0.17,
            "PSOE": 0.26,
            "VOX": 0.02,
            "PP": 0.01,
            "OTROS": 0.03,
        },
    }

    payload: dict[str, Any] = {
        "cis_study": {
            "study_id": CIS_STUDY_ID,
            "sample_size_designed": 45000,
            "sample_size_realized": 45074,
            "blank_estimate_share": float(
                headline.get("estimacion_valid_pct", {}).get("BLANK", 0.5)
            )
            / 100.0,
            "others_estimate_share": float(
                headline.get("estimacion_valid_pct", {}).get("OTROS", 3.4)
            )
            / 100.0,
            "estimacion_valid_pct": dict(headline.get("estimacion_valid_pct", {})),
            "direct_census_pct": dict(headline.get("direct_census_pct", {})),
        },
        "rv_eg23": {
            "origin_weighted_n": dict(rv_data["origin_weighted_n"]),
            "normalized_transitions": dict(rv_data["normalized_transitions"]),
        },
        "sumar_2023_split": sumar_split_zones,
        "navarra_upn_pp_merger": {
            "cpro": "31",
            "source_parties_2023": ["PP", "UPN"],
            "target_party_2026": "UPN_PP",
            "consolidation_retention": 0.94,
            "cis_upn_to_upn_pp_share": float(upn_nav_norm.get("UPN_PP", 0.835)),
            "cis_upn_to_pp_leakage_note": "PP absent on ballot in cpro='31'; UPN+PP unified.",
            "cis_pp_retention_national": float(pp_norm.get("PP", 0.722)),
        },
        "salf_2024_ep_anchor": {
            "national_2024_ep_share": round(salf_ep_nat, 6),
            "provincial_2024_ep_shares": {
                k: round(v, 6) for k, v in sorted(salf_ep_shares.items())
            },
            "absent_cpros_2026": [
                c for c, v in parties.get("SALF").boe_mask_2026.items() if v == 0
            ],
        },
        "unproclaimed_lists_2026": {
            p: {
                "proclaimed_districts": sum(parties.get(p).boe_mask_2026.values()),
                "stage2_prior_multiplier": 0.0,
            }
            for p in ("AC", "CUP", "UPL")
        },
    }
    return payload


def build_cis_3584_and_transfers(
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
    polls_dir: Path | None = None,
    metadata_dir: Path | None = None,
    elections_dir: Path | None = None,
    national_2026_polls: Sequence[PollRecord] | None = None,
    write_files: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Parse CIS #3584, run national IPF de-biasing, and write snapshots."""
    meta_root = metadata_dir or get_metadata_dir()
    polls_root = polls_dir or get_polls_dir()
    elec_root = elections_dir or get_elections_dir()

    consts = load_constituencies_catalog(meta_root / "constituencies_2026.json")
    parties = load_parties_catalog(meta_root / "parties_2026.json")

    if national_2026_polls is None:
        p26_csv = polls_root / "national_2026.csv"
        if p26_csv.is_file():
            from elecciones.ingest.wikipedia import load_national_polls

            national_2026_polls = load_national_polls(p26_csv)
        else:
            national_2026_polls = []

    xlsx_path = ensure_cis_3584_xlsx(offline=offline, raw_cache_dir=raw_cache_dir)
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    try:
        pond_weights = parse_cis_3584_weights(wb)
        headline = parse_cis_3584_headline(wb)
        prov_raw = parse_cis_3584_provincial(wb, parties)
        rv_data = parse_cis_3584_rv_eg23(wb)
    finally:
        wb.close()

    consensus = (
        compute_non_cis_national_consensus(national_2026_polls, parties)
        if national_2026_polls
        else {}
    )
    debias_map, achieved_nat = debias_cis_provincial_ipf(
        prov_raw, consts, parties, consensus
    )

    party_ids = list(parties.party_ids())
    consts_map = consts.by_cpro()
    rows_out: dict[str, dict[str, Any]] = {}
    for cpro in ALL_CPROS:
        w_info = pond_weights[cpro]
        p_info = prov_raw[cpro]
        rows_out[cpro] = {
            "cpro": cpro,
            "constituency": consts_map[cpro].name,
            "codauto": consts_map[cpro].codauto,
            "n_designed": int(w_info["n_designed"]),
            "n_realized": int(w_info["n_realized"]),
            "pond_weight": float(w_info["pond_weight"]),
            "weighted_n": float(p_info["weighted_n"]),
            "raw_shares": dict(p_info["raw_shares"]),
            "debias_shares": dict(debias_map[cpro]),
        }

    transfers = build_coalition_transfer_priors(
        rv_data,
        headline,
        elec_root / "europarl_2024_06.csv",
        parties,
    )
    transfers["cis_ipf_national_target_achieved"] = {
        k: round(v, 6) for k, v in achieved_nat.items() if v > 0.0
    }

    if write_files:
        cis_csv = polls_root / "cis_3584_provincial.csv"
        cis_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "cpro",
            "constituency",
            "codauto",
            "n_designed",
            "n_realized",
            "pond_weight",
            "weighted_n",
            *[f"raw_{p}" for p in party_ids],
            *[f"debias_{p}" for p in party_ids],
        ]
        with cis_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for cpro in ALL_CPROS:
                entry = rows_out[cpro]
                row_dict: dict[str, Any] = {
                    "cpro": cpro,
                    "constituency": entry["constituency"],
                    "codauto": entry["codauto"],
                    "n_designed": entry["n_designed"],
                    "n_realized": entry["n_realized"],
                    "pond_weight": f"{entry['pond_weight']:.6f}",
                    "weighted_n": f"{entry['weighted_n']:.4f}",
                }
                for p in party_ids:
                    row_dict[f"raw_{p}"] = f"{entry['raw_shares'][p]:.8f}"
                    row_dict[f"debias_{p}"] = f"{entry['debias_shares'][p]:.8f}"
                w.writerow(row_dict)

        tr_path = meta_root / "coalition_transfers.json"
        tr_path.parent.mkdir(parents=True, exist_ok=True)
        tr_path.write_text(
            json.dumps(transfers, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return rows_out, transfers


def load_cis_3584_provincial(
    csv_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load validated ``data/polls/cis_3584_provincial.csv`` unit shares."""
    target = csv_path or (get_polls_dir() / "cis_3584_provincial.csv")
    if not target.is_file():
        raise FileNotFoundError(f"Missing CIS provincial CSV: {target}")
    out: dict[str, dict[str, Any]] = {}
    with target.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cpro = str(row["cpro"]).zfill(2)
            raw_s: dict[str, float] = {}
            deb_s: dict[str, float] = {}
            for k, v in row.items():
                if k.startswith("raw_"):
                    raw_s[k[4:]] = float(v)
                elif k.startswith("debias_"):
                    deb_s[k[7:]] = float(v)
            if abs(sum(raw_s.values()) - 1.0) > 1e-6:
                raise SchemaValidationError(f"CIS raw_shares sum != 1 in cpro={cpro}")
            if abs(sum(deb_s.values()) - 1.0) > 1e-6:
                raise SchemaValidationError(f"CIS debias_shares sum != 1 in cpro={cpro}")
            out[cpro] = {
                "cpro": cpro,
                "constituency": row["constituency"],
                "codauto": str(row["codauto"]).zfill(2),
                "n_designed": int(row["n_designed"]),
                "n_realized": int(row["n_realized"]),
                "pond_weight": float(row["pond_weight"]),
                "weighted_n": float(row["weighted_n"]),
                "raw_shares": raw_s,
                "debias_shares": deb_s,
            }
    if len(out) != 52:
        raise SchemaValidationError(f"Expected 52 rows in {target}, got {len(out)}")
    return out


def load_coalition_transfers(json_path: Path | None = None) -> dict[str, Any]:
    """Load ``data/metadata/coalition_transfers.json``."""
    target = json_path or (get_metadata_dir() / "coalition_transfers.json")
    if not target.is_file():
        raise FileNotFoundError(f"Missing coalition transfers JSON: {target}")
    return dict(json.loads(target.read_text(encoding="utf-8")))
