"""Stage 2: 52-Province Composite Prior, Logit Swing, Masked IPF & Regional Shrinkage.

Projects national Stage 1 vote-share distributions down to all 52 Congress
constituencies while strictly respecting:
1. 3 Nov 2026 BOE candidacy proclamation mask ``M_{c,p} == 0 => 0.0``.
2. Exact district valid-vote simplex ``sum_p shares[c, p] + blank[c] == 1.0 ± 1e-9``
   both before (``pre_shrinkage_ipf_shares``) and after (``final_shares``)
   Empirical-Bayes shrinkage onto CIS #3584 & 2026 CCAA/provincial polls.
3. Explicit coalition transition matrices from ``coalition_transfers.json``
   (Navarre ``PP + UPN -> UPN_PP``, Sumar breakup across zones, ``SALF`` EP 2024
   provincial anchor, ``VOX`` SC Tenerife seed, unproclaimed ``AC/CUP/UPL``).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np

from elecciones.config import (
    DEFAULT_HYPERPARAMETERS,
    ModelHyperparameters,
    get_elections_dir,
    get_metadata_dir,
    get_polls_dir,
)
from elecciones.ingest.cis import load_cis_3584_provincial, load_coalition_transfers
from elecciones.ingest.elections import load_congress_results
from elecciones.ingest.wikipedia import load_regional_polls
from elecciones.models.kernel_aggregator import compute_wave_overlap_penalties
from elecciones.models.pollster_ratings import (
    _PARTISAN_INTERNAL_POLLSTERS,
    is_benchmark_or_excluded_pollster,
)
from elecciones.schemas import (
    ALL_CPROS,
    AUTONOMOUS_CITY_CPROS,
    SIMPLEX_TOL,
    ConstituenciesCatalog,
    ConstituencyElectionResult,
    PartiesCatalog,
    PollsterRatingRecord,
    RegionalPollRecord,
    SchemaValidationError,
    district_simplex_sum,
    load_constituencies_catalog,
    load_parties_catalog,
)

# Regional/historical fallback baseline weights for earliest election 2015-12
# (Derived purely from Spanish territorial political geography known prior to Dec 2015,
# zero leakage from Dec 2015 actual counts.)
_NAT_GEO_PRIOR_2015: Final[dict[str, dict[str, float]]] = {
    "PP": {
        "01": 0.90,
        "02": 1.05,
        "03": 0.85,
        "04": 1.05,
        "05": 0.82,
        "06": 1.05,
        "07": 1.32,
        "08": 1.30,
        "09": 0.42,
        "10": 1.10,
        "11": 1.25,
        "12": 1.32,
        "13": 1.22,
        "14": 1.42,
        "15": 1.02,
        "16": 0.42,
        "17": 1.32,
        "18": 1.55,
        "19": 1.55,
    },
    "PSOE": {
        "01": 1.40,
        "02": 1.05,
        "03": 1.05,
        "04": 0.85,
        "05": 0.95,
        "06": 0.92,
        "07": 1.02,
        "08": 1.25,
        "09": 0.72,
        "10": 0.92,
        "11": 1.52,
        "12": 0.95,
        "13": 0.82,
        "14": 0.92,
        "15": 0.72,
        "16": 0.62,
        "17": 1.05,
        "18": 1.05,
        "19": 1.05,
    },
    "PODEMOS": {
        "09": 1.15,
        "10": 1.18,
        "12": 1.20,
        "13": 1.05,
        "15": 1.10,
        "16": 1.22,
        "05": 1.12,
        "04": 1.08,
    },
    "CS": {
        "09": 1.05,
        "13": 1.32,
        "10": 1.12,
        "14": 1.12,
        "01": 0.98,
    },
}

_SUMAR_DEST_PARTIES: Final[frozenset[str]] = frozenset(
    {"FA", "PODEMOS", "COMPROMIS_FV", "UPC", "ARA_MES", "ADELANTE"}
)


@dataclass(frozen=True, slots=True)
class Stage2DistrictResult:
    """Point-in-time Stage 2 vote-share projection for one constituency."""

    cpro: str
    name: str
    codauto: str
    seats: int
    censo: int
    validos_prior: int
    turnout_mean: float
    turnout_weight: float
    prior_shares: dict[str, float]
    pre_shrinkage_ipf_shares: dict[str, float]
    pre_shrinkage_blank_share: float
    final_shares: dict[str, float]
    blank_share: float
    cis_weight: float
    regional_poll_weight: float


@dataclass(frozen=True, slots=True)
class Stage2Projection:
    """Full 52-district Stage 2 projection with vectorized matrices for Stage 3."""

    as_of_date: date
    election_id: str
    party_ids: tuple[str, ...]
    cpros: tuple[str, ...]
    codautos: tuple[str, ...]
    districts: dict[str, Stage2DistrictResult]
    stage1_targets: dict[str, float]
    target_national: dict[str, float]
    pre_shrinkage_national: dict[str, float]
    post_shrinkage_national: dict[str, float]
    ipf_max_marginal_err: float
    active_mask: np.ndarray
    turnout_weights: np.ndarray
    prior_matrix: np.ndarray
    pre_shrinkage_matrix: np.ndarray
    pre_shrinkage_blank: np.ndarray
    final_matrix: np.ndarray
    final_blank: np.ndarray
    censo_vec: np.ndarray
    turnout_mean_vec: np.ndarray
    seats_vec: np.ndarray
    city_vec: np.ndarray


def _exact_normalize_row(
    values: np.ndarray,
    active_mask_row: np.ndarray,
    blank_share: float,
) -> np.ndarray:
    """Enforce zero mask and IEEE-754 exact valid-vote simplex ``1.0 - blank_share``."""
    out = np.where(active_mask_row, np.maximum(0.0, values), 0.0).astype(np.float64)
    target = 1.0 - float(blank_share)
    if target <= 0.0:
        raise SchemaValidationError(f"blank_share must be < 1.0, got {blank_share}")
    s = float(out.sum())
    if s <= 0.0:
        active_idx = np.flatnonzero(active_mask_row)
        if active_idx.size == 0:
            raise SchemaValidationError("Zero active parties in constituency row")
        out[active_idx] = target / float(active_idx.size)
    else:
        out *= target / s

    # Compensate 1-2 IEEE-754 ULP residual on the largest entry so float dict sums to exact 1.0
    total_valid = float(math.fsum([*[float(x) for x in out], float(blank_share)]))
    residual = 1.0 - total_valid
    if abs(residual) > 1e-15:
        top_idx = int(np.argmax(out))
        out[top_idx] = max(0.0, float(out[top_idx]) + residual)
    return out


def _exact_normalize_matrix(
    mat: np.ndarray,
    active_mask: np.ndarray,
    blank_vec: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(mat, dtype=np.float64)
    for i in range(mat.shape[0]):
        out[i] = _exact_normalize_row(mat[i], active_mask[i], float(blank_vec[i]))
    return out


def _sumar_zone_for_codauto(codauto: str) -> str:
    if codauto == "10":
        return "valencian_community"
    if codauto == "05":
        return "canary_islands"
    if codauto == "04":
        return "illes_balears"
    if codauto == "09":
        return "catalonia"
    if codauto == "01":
        return "andalusia"
    return "general_territory"


def split_sumar_2023_share(
    cpro: str,
    codauto: str,
    sumar_share: float,
    active_mask_map: Mapping[str, int],
    transfers: Mapping[str, Any],
    *,
    ep24_fa_ratio: float | None = None,
) -> dict[str, float]:
    """Redistribute 2023 Sumar share cleanly onto 29-N active parties conserving exact mass.

    ``sum(returned.values())`` matches ``sumar_share`` within machine precision and every
    entry with ``active_mask_map[p] == 0`` is strictly excluded.
    """
    if sumar_share <= 0.0:
        return {}
    zone_map: Mapping[str, Mapping[str, float]] = transfers.get("sumar_2023_split", {})
    zone = _sumar_zone_for_codauto(codauto)
    raw_weights = dict(zone_map.get(zone) or zone_map.get("general_territory") or {})

    # In Navarra ("31"), 2023 PP/UPN shifted into UPN_PP rather than PP
    if cpro == "31" and "PP" in raw_weights and active_mask_map.get("PP", 0) == 0:
        pp_w = raw_weights.pop("PP")
        if active_mask_map.get("UPN_PP", 0) == 1:
            raw_weights["UPN_PP"] = raw_weights.get("UPN_PP", 0.0) + pp_w

    # Modulate local FA vs PODEMOS balance inside general_territory / andalusia via 2024 EP ratio
    if (
        ep24_fa_ratio is not None
        and "FA" in raw_weights
        and "PODEMOS" in raw_weights
        and active_mask_map.get("FA", 0) == 1
        and active_mask_map.get("PODEMOS", 0) == 1
    ):
        left_pair = raw_weights["FA"] + raw_weights["PODEMOS"]
        base_ratio = raw_weights["FA"] / max(1e-9, left_pair)
        # Blend 70% zone prior + 30% local EP 2024 SUMAR/(SUMAR+PODEMOS) signal
        adj_ratio = max(0.45, min(0.92, 0.70 * base_ratio + 0.30 * ep24_fa_ratio))
        raw_weights["FA"] = left_pair * adj_ratio
        raw_weights["PODEMOS"] = left_pair * (1.0 - adj_ratio)

    # Filter strictly onto proclaimed active lists M_{c,p} == 1
    filtered: dict[str, float] = {}
    missing_mass = 0.0
    for pid, w in raw_weights.items():
        if active_mask_map.get(pid, 0) == 1 and w > 0.0:
            filtered[pid] = float(w)
        elif w > 0.0:
            missing_mass += float(w)

    # Redirect unproclaimed destination shares within same family:
    # e.g., Teruel (FA=0 -> PODEMOS/TE/PSOE), Ceuta (FA=0 -> PODEMOS), Melilla (PSOE)
    if missing_mass > 0.0:
        fallback_order = ("PODEMOS", "TERUEL_EXISTE", "COMPROMIS_FV", "UPC", "ARA_MES", "PSOE")
        recips = [p for p in fallback_order if active_mask_map.get(p, 0) == 1]
        if not recips:
            recips = [p for p in filtered if filtered[p] > 0.0]
        if not recips:
            recips = [p for p, m in active_mask_map.items() if m == 1]
        if "PODEMOS" in recips and cpro in {"44", "51"}:
            # In Teruel ("44") & Ceuta ("51"), Podemos + PSOE absorb unproclaimed FA
            pod_w = 0.65
            sec = (
                "TERUEL_EXISTE"
                if cpro == "44" and active_mask_map.get("TERUEL_EXISTE", 0) == 1
                else "PSOE"
            )
            filtered["PODEMOS"] = filtered.get("PODEMOS", 0.0) + missing_mass * pod_w
            filtered[sec] = filtered.get(sec, 0.0) + missing_mass * (1.0 - pod_w)
        elif recips:
            split_each = missing_mass / len(recips)
            for p in recips:
                filtered[p] = filtered.get(p, 0.0) + split_each

    tot_w = sum(filtered.values())
    if tot_w <= 0.0:
        return {}
    return {pid: sumar_share * (w / tot_w) for pid, w in filtered.items()}


def _load_europarl_2024(elections_dir: Path) -> dict[str, dict[str, float]]:
    path = elections_dir / "europarl_2024_06.csv"
    out: dict[str, dict[str, float]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cpro = str(row["cpro"]).zfill(2)
            val = max(1.0, float(row["validos"]))
            shares: dict[str, float] = {"validos": val}
            for k, v in row.items():
                if k.startswith("votes_"):
                    shares[k[6:]] = float(v or 0) / val
            out[cpro] = shares
    return out


def _load_recent_regional_elections(
    elections_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Load CCAA and provincial shares (on 0..1) from ``regional_2024_2026.csv``."""
    path = elections_dir / "regional_2024_2026.csv"
    by_ccaa: dict[str, dict[str, float]] = {}
    by_prov: dict[str, dict[str, float]] = {}
    if not path.is_file():
        return by_ccaa, by_prov
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            shares: dict[str, float] = {}
            for k, v in row.items():
                if k.startswith("pct_") and v not in ("", None):
                    shares[k[4:]] = float(v) / 100.0
            st = str(row["scope_type"]).strip()
            sc = str(row["scope_code"]).zfill(2)
            if st == "ccaa":
                by_ccaa[sc] = shares
            elif st == "province":
                by_prov[sc] = shares
    return by_ccaa, by_prov


def build_2026_composite_prior(
    constituencies: ConstituenciesCatalog,
    parties: PartiesCatalog,
    *,
    c23_by_cpro: Mapping[str, ConstituencyElectionResult],
    ep24_by_cpro: Mapping[str, Mapping[str, float]],
    cis_prov: Mapping[str, Mapping[str, Any]],
    transfers: Mapping[str, Any],
    reg_elec_ccaa: Mapping[str, Mapping[str, float]] | None = None,
    reg_elec_prov: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build 52 x P composite prior matrix, blank shares, turnout means, and censo vector."""
    cpros = ALL_CPROS
    party_ids = parties.party_ids()
    n_c = len(cpros)
    n_p = len(party_ids)

    mask = np.array(parties.mask_matrix(cpros), dtype=bool)
    prior = np.zeros((n_c, n_p), dtype=np.float64)
    blank_vec = np.zeros(n_c, dtype=np.float64)
    turnout_mean = np.zeros(n_c, dtype=np.float64)
    censo_vec = np.zeros(n_c, dtype=np.float64)

    cis_blank_nat = float(transfers.get("cis_study", {}).get("blank_estimate_share", 0.005))
    nav_ret = float(transfers.get("navarra_upn_pp_merger", {}).get("consolidation_retention", 0.94))
    salf_ep_anchors: Mapping[str, float] = transfers.get("salf_2024_ep_anchor", {}).get(
        "provincial_2024_ep_shares", {}
    )
    ep24_sumar_pod_ratios: dict[str, float] = {}
    for cpro, ep_row in ep24_by_cpro.items():
        su = ep_row.get("SUMAR", 0.0)
        po = ep_row.get("PODEMOS", 0.0)
        if su + po > 0.0:
            ep24_sumar_pod_ratios[cpro] = su / (su + po)

    # Pre-compute national EP 2024 VOX share vs 2023 VOX share to scale SC Tenerife VOX cleanly
    ep_vox_num = sum(
        ep24_by_cpro[c].get("VOX", 0.0) * ep24_by_cpro[c].get("validos", 1.0)
        for c in ep24_by_cpro
    )
    ep_vox_den = sum(ep24_by_cpro[c].get("validos", 1.0) for c in ep24_by_cpro)
    ep_vox_nat = ep_vox_num / max(1.0, ep_vox_den)
    c23_vox_num = sum(r.party_votes.get("VOX", 0) for r in c23_by_cpro.values())
    c23_vox_den = sum(r.validos for r in c23_by_cpro.values())
    c23_vox_nat = c23_vox_num / max(1, c23_vox_den)

    consts_map = constituencies.by_cpro()
    for i, cpro in enumerate(cpros):
        con = consts_map[cpro]
        r23 = c23_by_cpro[cpro]
        ep24 = ep24_by_cpro.get(cpro, {})
        cis_deb = cis_prov.get(cpro, {}).get("debias_shares", {})
        mask_map = {p: int(mask[i, j]) for j, p in enumerate(party_ids)}

        censo_c = float(con.censo_2023 or r23.censo)
        censo_vec[i] = censo_c
        turnout_mean[i] = min(0.95, max(0.35, float(r23.validos) / max(1.0, float(r23.censo))))
        b23 = float(r23.blancos) / max(1.0, float(r23.validos))
        blank_vec[i] = min(0.025, max(0.004, 0.70 * b23 + 0.30 * cis_blank_nat))

        val23 = float(r23.validos)
        row_shares: dict[str, float] = {p: 0.0 for p in party_ids}

        # Direct 2023 continuing parties (excluding SUMAR, PP+UPN in 31, unproclaimed 2026 parties)
        unproclaimed_redirect = {"CUP", "PDECAT", "JM", "XAV", "UPL"}
        handled_23 = {"SUMAR", "NC", "ADELANTE", *unproclaimed_redirect}
        for p23, votes in r23.party_votes.items():
            if votes <= 0 or p23 in handled_23:
                continue
            s23 = votes / val23
            if cpro == "31" and p23 in {"PP", "UPN"}:
                continue
            if p23 in row_shares and mask_map.get(p23, 0) == 1:
                row_shares[p23] += s23
            else:
                if mask_map.get("OTROS", 0) == 1:
                    row_shares["OTROS"] += s23
                else:
                    row_shares["PSOE"] += 0.5 * s23
                    if mask_map.get("PP", 0) == 1:
                        row_shares["PP"] += 0.5 * s23

        # Unproclaimed 2023 lists -> OTROS or regional cognate in masked districts
        for up23 in unproclaimed_redirect:
            s_up = r23.party_votes.get(up23, 0) / val23
            if s_up <= 0.0:
                continue
            if up23 in {"CUP", "PDECAT"} and cpro in {"08", "17", "25", "43"}:
                # CUP / PDeCAT split cleanly between ERC, JUNTS, ALHORA, OTROS
                row_shares["ERC"] += 0.35 * s_up
                row_shares["JUNTS"] += 0.35 * s_up
                row_shares["ALHORA"] += 0.10 * s_up
                row_shares["OTROS"] += 0.20 * s_up
            elif mask_map.get("OTROS", 0) == 1:
                row_shares["OTROS"] += s_up
            else:
                row_shares["PP"] += 0.5 * s_up
                row_shares["PSOE"] += 0.5 * s_up

        # Navarre PP + UPN -> UPN_PP with consolidation retention & leakage
        if cpro == "31":
            comb_nav = (r23.party_votes.get("PP", 0) + r23.party_votes.get("UPN", 0)) / val23
            row_shares["UPN_PP"] += comb_nav * nav_ret
            leak = comb_nav * (1.0 - nav_ret)
            row_shares["VOX"] += 0.65 * leak
            row_shares["OTROS"] += 0.35 * leak

        # 2023 Sumar breakup across zones using explicit conservation helper
        sumar_23 = r23.party_votes.get("SUMAR", 0) / val23
        if sumar_23 > 0.0:
            splits = split_sumar_2023_share(
                cpro,
                con.codauto,
                sumar_23,
                mask_map,
                transfers,
                ep24_fa_ratio=ep24_sumar_pod_ratios.get(cpro),
            )
            for dest_p, d_share in splits.items():
                row_shares[dest_p] += d_share

        # 2023 Nueva Canarias (Canarias 35, 38) -> UPC + PSOE/CCA
        nc_23 = r23.party_votes.get("NC", 0) / val23
        if nc_23 > 0.0:
            if mask_map.get("UPC", 0) == 1:
                row_shares["UPC"] += 0.85 * nc_23
                row_shares["PSOE"] += 0.10 * nc_23
                row_shares["CCA"] += 0.05 * nc_23
            elif mask_map.get("OTROS", 0) == 1:
                row_shares["OTROS"] += nc_23

        # Explicit 2023 Adelante Andalucía votes (Cádiz "11" & Sevilla "41")
        ad_23 = r23.party_votes.get("ADELANTE", 0) / val23
        if ad_23 > 0.0 and mask_map.get("ADELANTE", 0) == 1:
            row_shares["ADELANTE"] += 0.85 * ad_23
            row_shares["PODEMOS"] += 0.15 * ad_23

        # Special active list seeds where 2023 Congress votes == 0 (SC Tenerife VOX, ALHORA, FO)
        if cpro == "38" and mask_map.get("VOX", 0) == 1 and r23.party_votes.get("VOX", 0) == 0:
            ep_vox = ep24.get("VOX", 0.11)
            scaled_ep_vox = min(0.26, ep_vox * (c23_vox_nat / max(1e-4, ep_vox_nat)))
            cis_vox = float(cis_deb.get("VOX", 0.18))
            row_shares["VOX"] = max(0.08, 0.55 * cis_vox + 0.45 * scaled_ep_vox)

        if mask_map.get("ALHORA", 0) == 1 and row_shares["ALHORA"] <= 1e-6:
            row_shares["ALHORA"] = max(0.0025, float(cis_deb.get("ALHORA", 0.0025)))

        if mask_map.get("FO", 0) == 1 and row_shares["FO"] <= 1e-6:
            row_shares["FO"] = max(
                0.0005,
                0.5 * float(cis_deb.get("FO", 0.0005)) + 0.5 * float(ep24.get("FO", 0.0005)),
            )

        # SALF 2024 EP provincial anchor where proclaimed
        if mask_map.get("SALF", 0) == 1:
            ep_salf = float(salf_ep_anchors.get(cpro, ep24.get("SALF", 0.04)))
            salf_seed = max(0.004, ep_salf * 0.42)
            row_shares["SALF"] = salf_seed

        # Mild structural adjustment from 2024-2026 regional elections (Galicia/Euskadi/Cat)
        reg_source = (
            (reg_elec_prov or {}).get(cpro)
            or (reg_elec_ccaa or {}).get(con.codauto)
        )
        if reg_source:
            for p_reg, reg_val in reg_source.items():
                if (
                    p_reg in {"BNG", "EH_BILDU", "PNV", "ERC", "JUNTS", "ADELANTE"}
                    and mask_map.get(p_reg, 0) == 1
                    and row_shares.get(p_reg, 0.0) > 0.0
                    and reg_val > 0.0
                ):
                    # Gentle 12% geometric blend with recent autonomous election structural trend
                    cur = row_shares[p_reg]
                    row_shares[p_reg] = (cur ** 0.88) * (reg_val ** 0.12)

        # Ensure every M_{c,p} == 1 has positive seed > 0 so log-odds IPF never freezes a list at 0
        for p in party_ids:
            if mask_map[p] == 1 and row_shares[p] <= 1e-7:
                fallback = max(0.0004, float(cis_deb.get(p, 0.0004)))
                row_shares[p] = fallback

        vec = np.array([row_shares[p] for p in party_ids], dtype=np.float64)
        prior[i] = _exact_normalize_row(vec, mask[i], blank_vec[i])

    return prior, blank_vec, turnout_mean, censo_vec


def build_ipf_national_targets(
    party_ids: Sequence[str],
    prior_matrix: np.ndarray,
    turnout_weights: np.ndarray,
    stage1_mean: Mapping[str, float],
    blank_nat: float,
    *,
    election_id: str = "2026-11",
) -> tuple[dict[str, float], list[int]]:
    """Construct complete valid-vote national marginal targets matching ``1.0 - blank_nat``.

    For 2026, splits ``stage1_mean["PP"]`` proportionally between ``PP`` (51 districts) and
    ``UPN_PP`` (Navarre ``31``) using the empirical electorate prior ratio, matching CIS IPF.
    """
    p_idx = {p: j for j, p in enumerate(party_ids)}
    nat_prior = turnout_weights @ prior_matrix
    target = nat_prior.copy()

    tracked_set: set[int] = set()
    if (
        election_id == "2026-11"
        and "PP" in stage1_mean
        and "PP" in p_idx
        and "UPN_PP" in p_idx
        and nat_prior[p_idx["UPN_PP"]] > 0.0
    ):
        j_pp = p_idx["PP"]
        j_upn = p_idx["UPN_PP"]
        duo = nat_prior[j_pp] + nat_prior[j_upn]
        share_pp = nat_prior[j_pp] / max(1e-12, duo)
        target[j_pp] = float(stage1_mean["PP"]) * share_pp
        target[j_upn] = float(stage1_mean["PP"]) * (1.0 - share_pp)
        tracked_set.update((j_pp, j_upn))

    for party, val in stage1_mean.items():
        if party == "OTROS":
            continue
        if party == "PP" and "PP" in p_idx and p_idx["PP"] in tracked_set:
            continue
        j = p_idx.get(party)
        if j is not None and nat_prior[j] > 0.0 and val > 0.0:
            target[j] = float(val)
            tracked_set.add(j)

    # Zero unproclaimed parties (nat_prior == 0)
    for j in range(len(party_ids)):
        if nat_prior[j] <= 0.0:
            target[j] = 0.0

    # Keep sum(target) == 1.0 - blank_nat by scaling non-tracked micro-parties + OTROS
    cand_target = 1.0 - blank_nat
    tracked_sum = float(target[list(tracked_set)].sum()) if tracked_set else 0.0
    untracked_idx = [
        j for j in range(len(party_ids)) if j not in tracked_set and target[j] > 0.0
    ]
    untracked_sum = float(target[untracked_idx].sum()) if untracked_idx else 0.0

    if untracked_idx and untracked_sum > 0.0 and tracked_sum < cand_target - 1e-4:
        scale_un = (cand_target - tracked_sum) / untracked_sum
        for j in untracked_idx:
            target[j] *= scale_un
    elif float(target.sum()) > 0.0:
        target *= cand_target / float(target.sum())

    target_dict = {party_ids[j]: float(target[j]) for j in range(len(party_ids))}
    return target_dict, sorted(tracked_set)


def rake_provincial_ipf(
    prior_matrix: np.ndarray,
    active_mask: np.ndarray,
    blank_vec: np.ndarray,
    turnout_weights: np.ndarray,
    target_vec: np.ndarray,
    *,
    tracked_indices: Sequence[int] | None = None,
    n_iters: int = 80,
    tol: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Run masked multinomial-logit swing + turnout-weighted IPF across 52 districts.

    Guarantees:
    - ``M_{c,p} == 0 => out[c, p] == 0.0``
    - ``abs(out[c].sum() + blank_vec[c] - 1.0) <= 1e-12`` for every district ``c``
    - ``max_j |sum_c w_c out[c, j] - target_vec[j]|`` converges below ``1e-7``.
    """
    n_c, n_p = prior_matrix.shape
    nat_prior = turnout_weights @ prior_matrix

    # Step 1: Initialize in log-odds space (uniform national log-odds shift alpha_p per party)
    X = np.zeros_like(prior_matrix, dtype=np.float64)
    for j in range(n_p):
        if nat_prior[j] > 0.0 and target_vec[j] > 0.0:
            alpha_j = math.log(target_vec[j] / nat_prior[j])
            log_ratio = np.where(
                active_mask[:, j] & (prior_matrix[:, j] > 0.0),
                np.log(np.maximum(1e-12, prior_matrix[:, j])) + alpha_j,
                -100.0,
            )
            X[:, j] = np.where(active_mask[:, j], np.exp(log_ratio), 0.0)
    X = _exact_normalize_matrix(X, active_mask, blank_vec)

    # Active marginal columns to enforce via IPF
    if tracked_indices is not None:
        t_set = set(tracked_indices)
        all_target_cols = [
            j
            for j in range(n_p)
            if j in t_set and target_vec[j] > 0.0 and nat_prior[j] > 0.0
        ]
    else:
        all_target_cols = [j for j in range(n_p) if target_vec[j] > 0.0 and nat_prior[j] > 0.0]

    # Step 2: Masked IPF alternating national margins & row simplices
    max_err = 1.0
    for it in range(max(10, n_iters)):
        cur_nat = turnout_weights @ X
        col_scale = np.ones(n_p, dtype=np.float64)
        for j in all_target_cols:
            if cur_nat[j] > 1e-12:
                col_scale[j] = target_vec[j] / cur_nat[j]
        X_next = X * col_scale[None, :]
        X_next = _exact_normalize_matrix(X_next, active_mask, blank_vec)

        nat_after = turnout_weights @ X_next
        if all_target_cols:
            diffs = np.abs(nat_after[all_target_cols] - target_vec[all_target_cols])
            max_err = float(np.max(diffs))
        X = X_next
        if it >= 6 and max_err < tol:
            break

    return X, max_err


def _canon_regional_poll_party(
    party: str,
    codauto: str,
    mask_map: Mapping[str, int],
    *,
    has_explicit_fa: bool,
    has_explicit_fv: bool,
    has_explicit_adelante: bool,
    has_explicit_upc: bool,
) -> dict[str, float]:
    """Map regional/provincial Wikipedia poll column names onto canonical masked 2026 parties."""
    if codauto == "15" and party in {"PP", "UPN", "UPN_PP"}:
        return {"UPN_PP": 1.0} if mask_map.get("UPN_PP", 0) == 1 else {}
    if party in {"COMPROMIS", "COMPROMIS_FV"} and mask_map.get("COMPROMIS_FV", 0) == 1:
        return {"COMPROMIS_FV": 1.0}
    if party in {"NC", "UPC"} and codauto == "05" and mask_map.get("UPC", 0) == 1:
        return {"UPC": 1.0}
    if codauto == "02" and party == "CHA" and not has_explicit_fa and mask_map.get("FA", 0) == 1:
        return {"FA": 1.0}
    if party == "SUMAR":
        if codauto == "10" and not has_explicit_fv and mask_map.get("COMPROMIS_FV", 0) == 1:
            return {"COMPROMIS_FV": 0.90}
        if codauto == "05" and not has_explicit_upc and mask_map.get("UPC", 0) == 1:
            return {"UPC": 0.85}
        if codauto == "04" and mask_map.get("ARA_MES", 0) == 1:
            return {"ARA_MES": 0.85}
        if codauto == "01" and not has_explicit_fa and mask_map.get("FA", 0) == 1:
            out_a = {"FA": 0.62}
            if not has_explicit_adelante and mask_map.get("ADELANTE", 0) == 1:
                out_a["ADELANTE"] = 0.25
            return out_a
        if codauto == "09" and not has_explicit_fa and mask_map.get("FA", 0) == 1:
            return {"FA": 0.95}
        if not has_explicit_fa and mask_map.get("FA", 0) == 1:
            return {"FA": 0.82}
        return {}
    if mask_map.get(party, 0) == 1:
        return {party: 1.0}
    return {}


def shrink_provinces_with_cis_and_regional_polls(
    pre_matrix: np.ndarray,
    active_mask: np.ndarray,
    blank_vec: np.ndarray,
    turnout_weights: np.ndarray,
    party_ids: Sequence[str],
    constituencies: ConstituenciesCatalog,
    as_of: date,
    *,
    cis_prov: Mapping[str, Mapping[str, Any]] | None = None,
    regional_polls: Sequence[RegionalPollRecord] | None = None,
    ratings_map: Mapping[str, PollsterRatingRecord] | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical-Bayes shrinkage toward CIS #3584 provincial cross-tabs and 2026 local polls."""
    cpros = ALL_CPROS
    n_c, n_p = pre_matrix.shape
    p_idx = {p: j for j, p in enumerate(party_ids)}
    c_idx = {c: i for i, c in enumerate(cpros)}
    consts_map = constituencies.by_cpro()
    by_codauto = constituencies.by_codauto()

    n0 = float(hyperparams.stage2_prior_weight)
    cis_w_num = np.zeros(n_c, dtype=np.float64)
    reg_w_num = np.zeros(n_c, dtype=np.float64)
    cur = pre_matrix.copy()

    # Track marginal precision N_eff per district after CIS/regional blending
    eff_n_c = np.full(n_c, n0, dtype=np.float64)
    did_shrink = False

    # Step 3A: Precision-weighted blend with nationally de-biased CIS #3584 provincial shares
    if cis_prov:
        for i, cpro in enumerate(cpros):
            c_row = cis_prov.get(cpro)
            if not c_row:
                continue
            deb_s = c_row.get("debias_shares", {})
            n_realized = float(c_row.get("n_realized", 0))
            if n_realized <= 0.0 or not deb_s:
                continue
            w_cis = min(
                n_realized / max(0.5, hyperparams.stage2_cis_divisor),
                hyperparams.stage2_cis_max_frac * n0,
            )
            cis_vec = np.array([float(deb_s.get(p, 0.0)) for p in party_ids], dtype=np.float64)
            cis_vec = _exact_normalize_row(cis_vec, active_mask[i], float(blank_vec[i]))
            cur[i] = (n0 * cur[i] + w_cis * cis_vec) / (n0 + w_cis)
            cur[i] = _exact_normalize_row(cur[i], active_mask[i], float(blank_vec[i]))
            cis_w_num[i] = w_cis
            eff_n_c[i] += w_cis
            did_shrink = True

    # Step 3B: Contemporaneous 2026 CCAA & Provincial polls <= as_of
    if regional_polls:
        eligible = [
            rp
            for rp in regional_polls
            if not is_benchmark_or_excluded_pollster(rp.pollster)
            and rp.fieldwork_end <= as_of
            and (rp.pub_date is None or rp.pub_date <= as_of)
            and (as_of - rp.fieldwork_end).days <= 210
        ]
        # Track CCAA-level waves to discount duplicate provincial breakdowns from the same survey
        ccaa_poll_keys = {
            (rp.pollster, rp.fieldwork_end, rp.scope_code)
            for rp in eligible
            if rp.scope_type == "ccaa"
        }

        # CCAA-level consensus adjustments first, then finer province-level blends
        scope_groups: dict[tuple[str, str], list[RegionalPollRecord]] = defaultdict(list)
        for rp in eligible:
            scope_groups[(rp.scope_type, rp.scope_code)].append(rp)

        def _poll_precision(rp: RegionalPollRecord) -> float:
            age_d = max(0, (as_of - rp.fieldwork_end).days)
            rec = math.exp(-age_d / 32.0)
            n_cap = float(min(rp.sample_size or 800, 3000))
            rating = ratings_map.get(rp.pollster) if ratings_map else None
            qual = max(0.25, rating.grade_weight) if rating else 1.0
            if rp.pollster in _PARTISAN_INTERNAL_POLLSTERS or rp.pollster == "ERC":
                qual *= 0.15
            return rec * qual * (n_cap / max(0.5, hyperparams.stage2_regional_divisor))

        def _accumulate_scope_polls(
            plist: list[RegionalPollRecord],
            codauto_c: str,
            mask_map: Mapping[str, int],
            *,
            apply_ccaa_dup_discount: bool,
        ) -> tuple[dict[str, float], dict[str, float]]:
            overlap_u = compute_wave_overlap_penalties(plist)
            party_num: dict[str, float] = defaultdict(float)
            party_den: dict[str, float] = defaultdict(float)
            for idx_rp, rp in enumerate(plist):
                w = _poll_precision(rp) * float(overlap_u[idx_rp])
                if (
                    apply_ccaa_dup_discount
                    and (rp.pollster, rp.fieldwork_end, codauto_c) in ccaa_poll_keys
                ):
                    w *= 0.55
                if w <= 1e-12:
                    continue
                has_fa = "FA" in rp.party_shares
                has_fv = "COMPROMIS_FV" in rp.party_shares
                has_ad = "ADELANTE" in rp.party_shares
                has_upc = "UPC" in rp.party_shares
                # Aggregate mapped columns within the same poll before adding weight to
                # denominator so split lists (e.g. Navarre PP+UPN -> UPN_PP) sum cleanly;
                # skip polls with large unproclaimed push-poll parties (e.g. Madrid IE > 6%).
                max_unmapped_non_party = 0.0
                rp_mapped: dict[str, float] = defaultdict(float)
                for raw_p, val in rp.party_shares.items():
                    mapped = _canon_regional_poll_party(
                        raw_p,
                        codauto_c,
                        mask_map,
                        has_explicit_fa=has_fa,
                        has_explicit_fv=has_fv,
                        has_explicit_adelante=has_ad,
                        has_explicit_upc=has_upc,
                    )
                    if not mapped and raw_p not in p_idx and val > max_unmapped_non_party:
                        max_unmapped_non_party = val
                    for canon_p, factor in mapped.items():
                        rp_mapped[canon_p] += val * factor
                if max_unmapped_non_party > 0.06:
                    continue
                for canon_p, mapped_val in rp_mapped.items():
                    party_num[canon_p] += w * mapped_val
                    party_den[canon_p] += w
            return party_num, party_den

        # CCAA polls (ratio-adjust CCAA mean while preserving intra-CCAA geography)
        for (st, sc), plist in sorted(scope_groups.items()):
            if st != "ccaa":
                continue
            members = by_codauto.get(sc, [])
            if not members:
                continue
            member_idx = [c_idx[m.cpro] for m in members]
            codauto_c = sc
            union_mask = {
                party_ids[j]: int(np.any(active_mask[member_idx, j]))
                for j in range(n_p)
            }
            party_num, party_den = _accumulate_scope_polls(
                plist, codauto_c, union_mask, apply_ccaa_dup_discount=False
            )
            if not party_den:
                continue
            did_shrink = True
            mw = turnout_weights[member_idx]
            mw_norm = mw / max(1e-12, float(mw.sum()))
            ccaa_cur_mean = mw_norm @ cur[member_idx]  # (P,)
            n_cur_ccaa = float(np.mean(eff_n_c[member_idx]))

            max_w_ccaa = max(party_den.values())
            w_eff_ccaa = min(max_w_ccaa, 0.45 * n_cur_ccaa)
            for canon_p, w_tot in party_den.items():
                j = p_idx.get(canon_p)
                if j is None or ccaa_cur_mean[j] <= 1e-6:
                    continue
                y_poll = party_num[canon_p] / w_tot
                w_eff = min(w_tot, 0.45 * n_cur_ccaa)
                target_ccaa_j = (n_cur_ccaa * float(ccaa_cur_mean[j]) + w_eff * y_poll) / (
                    n_cur_ccaa + w_eff
                )
                ratio = max(0.55, min(1.80, target_ccaa_j / max(1e-6, float(ccaa_cur_mean[j]))))
                for ci in member_idx:
                    if active_mask[ci, j]:
                        cur[ci, j] *= ratio
            for ci in member_idx:
                cur[ci] = _exact_normalize_row(cur[ci], active_mask[ci], float(blank_vec[ci]))
                reg_w_num[ci] += w_eff_ccaa
                eff_n_c[ci] += w_eff_ccaa

        # Direct provincial polls
        for (st, sc), plist in sorted(scope_groups.items()):
            if st != "province" or sc not in c_idx:
                continue
            ci = c_idx[sc]
            codauto_c = consts_map[sc].codauto
            mask_map = {party_ids[j]: int(active_mask[ci, j]) for j in range(n_p)}
            party_num, party_den = _accumulate_scope_polls(
                plist, codauto_c, mask_map, apply_ccaa_dup_discount=True
            )
            if not party_den:
                continue
            did_shrink = True
            n_cur_p = eff_n_c[ci]
            max_w_prov = max(party_den.values())
            w_eff_prov = min(max_w_prov, 0.40 * n_cur_p)
            for canon_p, w_tot in party_den.items():
                j = p_idx.get(canon_p)
                if j is None or not active_mask[ci, j]:
                    continue
                y_poll = party_num[canon_p] / w_tot
                w_eff = min(w_tot, 0.40 * n_cur_p)
                cur[ci, j] = (n_cur_p * cur[ci, j] + w_eff * y_poll) / (n_cur_p + w_eff)
            cur[ci] = _exact_normalize_row(cur[ci], active_mask[ci], float(blank_vec[ci]))
            reg_w_num[ci] += w_eff_prov
            eff_n_c[ci] += w_eff_prov

    denom = n0 + cis_w_num + reg_w_num
    cis_weights = cis_w_num / denom
    reg_weights = reg_w_num / denom

    # Step 3C: Re-anchor nationwide parties onto Stage 1 national marginals via masked IPF so
    # local empirical-Bayes updates shape intra-party geography without introducing national drift
    # from unbalanced regional poll coverage or inflated CIS "OTROS".
    if did_shrink:
        pre_nat = turnout_weights @ pre_matrix
        nationwide_cols = [
            p_idx[p]
            for p in ("PP", "PSOE", "VOX", "FA", "PODEMOS", "SALF", "OTROS")
            if p in p_idx and pre_nat[p_idx[p]] > 0.0
        ]
        j_pp = p_idx.get("PP")
        j_upn = p_idx.get("UPN_PP")
        pp_duo_target = (
            float(pre_nat[j_pp] + pre_nat[j_upn])
            if j_pp is not None and j_upn is not None
            else None
        )
        if nationwide_cols:
            for _it in range(40):
                cur_nat = turnout_weights @ cur
                col_scale = np.ones(n_p, dtype=np.float64)
                for j in nationwide_cols:
                    if cur_nat[j] <= 1e-12:
                        continue
                    targ_j = float(pre_nat[j])
                    if pp_duo_target is not None and j == j_pp and j_upn is not None:
                        targ_j = max(0.01, pp_duo_target - float(cur_nat[j_upn]))
                    col_scale[j] = targ_j / float(cur_nat[j])
                cur = _exact_normalize_matrix(cur * col_scale[None, :], active_mask, blank_vec)
                cur_nat_after = turnout_weights @ cur
                errs = [
                    abs(
                        float(cur_nat_after[j])
                        - (
                            max(0.01, pp_duo_target - float(cur_nat_after[j_upn]))
                            if pp_duo_target is not None and j == j_pp and j_upn is not None
                            else float(pre_nat[j])
                        )
                    )
                    for j in nationwide_cols
                ]
                if max(errs, default=0.0) < 1e-8:
                    break

    final_mat = _exact_normalize_matrix(cur, active_mask, blank_vec)
    return final_mat, cis_weights, reg_weights


def project_2026_provinces(
    stage1_mean: Mapping[str, float],
    as_of: date,
    *,
    blank_nat: float = 0.0081,
    constituencies: ConstituenciesCatalog | None = None,
    parties: PartiesCatalog | None = None,
    c23_results: Sequence[ConstituencyElectionResult] | None = None,
    cis_prov: Mapping[str, Mapping[str, Any]] | None = None,
    regional_polls: Sequence[RegionalPollRecord] | None = None,
    ratings: Sequence[PollsterRatingRecord] | None = None,
    transfers: Mapping[str, Any] | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    elections_dir: Path | None = None,
    polls_dir: Path | None = None,
    metadata_dir: Path | None = None,
    apply_shrinkage: bool = True,
) -> Stage2Projection:
    """Run full Stage 2 52-province projection for the 29-N 2026 election."""
    meta_root = metadata_dir or get_metadata_dir()
    elec_root = elections_dir or get_elections_dir()
    polls_root = polls_dir or get_polls_dir()

    consts = constituencies or load_constituencies_catalog(meta_root / "constituencies_2026.json")
    party_cat = parties or load_parties_catalog(meta_root / "parties_2026.json")
    party_ids = party_cat.party_ids()
    cpros = ALL_CPROS

    c23_list = c23_results or load_congress_results("2023-07", elections_dir=elec_root)
    c23_by_cpro = {r.cpro: r for r in c23_list}
    ep24_by_cpro = _load_europarl_2024(elec_root)
    reg_elec_ccaa, reg_elec_prov = _load_recent_regional_elections(elec_root)

    if transfers is None:
        transfers = load_coalition_transfers(meta_root / "coalition_transfers.json")
    if cis_prov is None and apply_shrinkage:
        cis_csv = polls_root / "cis_3584_provincial.csv"
        cis_prov = load_cis_3584_provincial(cis_csv) if cis_csv.is_file() else {}
    elif cis_prov is None:
        cis_prov = {}

    if regional_polls is None and apply_shrinkage:
        reg_csv = polls_root / "regional_2026.csv"
        regional_polls = load_regional_polls(reg_csv) if reg_csv.is_file() else []

    prior_mat, blank_vec, turnout_mean, censo_vec = build_2026_composite_prior(
        consts,
        party_cat,
        c23_by_cpro=c23_by_cpro,
        ep24_by_cpro=ep24_by_cpro,
        cis_prov=cis_prov,
        transfers=transfers,
        reg_elec_ccaa=reg_elec_ccaa,
        reg_elec_prov=reg_elec_prov,
    )

    exp_valid = censo_vec * turnout_mean
    turnout_weights = exp_valid / float(exp_valid.sum())
    active_mask = np.array(party_cat.mask_matrix(cpros), dtype=bool)

    cur_blank_nat = float(turnout_weights @ blank_vec)
    if cur_blank_nat > 0.0:
        blank_vec = np.clip(blank_vec * (blank_nat / cur_blank_nat), 0.002, 0.04)
    eff_blank_nat = float(turnout_weights @ blank_vec)
    prior_mat = _exact_normalize_matrix(prior_mat, active_mask, blank_vec)

    target_nat_dict, _ = build_ipf_national_targets(
        party_ids,
        prior_mat,
        turnout_weights,
        stage1_mean,
        eff_blank_nat,
        election_id="2026-11",
    )
    target_vec = np.array([target_nat_dict[p] for p in party_ids], dtype=np.float64)

    pre_mat, max_err = rake_provincial_ipf(
        prior_mat,
        active_mask,
        blank_vec,
        turnout_weights,
        target_vec,
        tracked_indices=None,
        n_iters=hyperparams.stage2_ipf_iters,
    )

    if apply_shrinkage:
        ratings_map = {r.pollster: r for r in ratings} if ratings else {}
        final_mat, cis_ws, reg_ws = shrink_provinces_with_cis_and_regional_polls(
            pre_mat,
            active_mask,
            blank_vec,
            turnout_weights,
            party_ids,
            consts,
            as_of,
            cis_prov=cis_prov,
            regional_polls=regional_polls,
            ratings_map=ratings_map,
            hyperparams=hyperparams,
        )
    else:
        final_mat = pre_mat.copy()
        cis_ws = np.zeros(len(cpros), dtype=np.float64)
        reg_ws = np.zeros(len(cpros), dtype=np.float64)

    pre_nat = turnout_weights @ pre_mat
    post_nat = turnout_weights @ final_mat

    consts_map = consts.by_cpro()
    districts: dict[str, Stage2DistrictResult] = {}
    codautos = tuple(consts_map[c].codauto for c in cpros)
    seats_vec = np.array([consts_map[c].seats_2026 for c in cpros], dtype=np.int32)
    city_vec = np.array([c in AUTONOMOUS_CITY_CPROS for c in cpros], dtype=bool)

    for i, cpro in enumerate(cpros):
        con = consts_map[cpro]
        b_i = float(blank_vec[i])
        prior_d = {party_ids[j]: float(prior_mat[i, j]) for j in range(len(party_ids))}
        pre_d = {party_ids[j]: float(pre_mat[i, j]) for j in range(len(party_ids))}
        fin_d = {party_ids[j]: float(final_mat[i, j]) for j in range(len(party_ids))}
        if abs(district_simplex_sum(pre_d, b_i) - 1.0) > SIMPLEX_TOL:
            raise SchemaValidationError(f"cpro={cpro} pre_shrinkage simplex violation")
        if abs(district_simplex_sum(fin_d, b_i) - 1.0) > SIMPLEX_TOL:
            raise SchemaValidationError(f"cpro={cpro} final simplex violation")

        districts[cpro] = Stage2DistrictResult(
            cpro=cpro,
            name=con.name,
            codauto=con.codauto,
            seats=con.seats_2026,
            censo=int(round(float(censo_vec[i]))),
            validos_prior=int(round(float(exp_valid[i]))),
            turnout_mean=float(turnout_mean[i]),
            turnout_weight=float(turnout_weights[i]),
            prior_shares=prior_d,
            pre_shrinkage_ipf_shares=pre_d,
            pre_shrinkage_blank_share=b_i,
            final_shares=fin_d,
            blank_share=b_i,
            cis_weight=float(cis_ws[i]),
            regional_poll_weight=float(reg_ws[i]),
        )

    return Stage2Projection(
        as_of_date=as_of,
        election_id="2026-11",
        party_ids=party_ids,
        cpros=cpros,
        codautos=codautos,
        districts=districts,
        stage1_targets={k: float(v) for k, v in stage1_mean.items()},
        target_national=target_nat_dict,
        pre_shrinkage_national={party_ids[j]: float(pre_nat[j]) for j in range(len(party_ids))},
        post_shrinkage_national={party_ids[j]: float(post_nat[j]) for j in range(len(party_ids))},
        ipf_max_marginal_err=max_err,
        active_mask=active_mask,
        turnout_weights=turnout_weights,
        prior_matrix=prior_mat,
        pre_shrinkage_matrix=pre_mat,
        pre_shrinkage_blank=blank_vec,
        final_matrix=final_mat,
        final_blank=blank_vec.copy(),
        censo_vec=censo_vec,
        turnout_mean_vec=turnout_mean,
        seats_vec=seats_vec,
        city_vec=city_vec,
    )


def project_historical_provinces(
    election_id: str,
    stage1_mean: Mapping[str, float],
    as_of: date,
    *,
    constituencies: ConstituenciesCatalog | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    elections_dir: Path | None = None,
    metadata_dir: Path | None = None,
) -> Stage2Projection:
    """Leak-free historical Stage 2 projection for out-of-sample LOOCV backtesting.

    Uses only prior general election baselines (or pre-2015 regional geography for
    ``2015-12``) plus the proclaimed ballot party list presence for ``election_id``,
    raked via identical multinomial-logit swing + masked IPF onto ``stage1_mean``.
    """
    meta_root = metadata_dir or get_metadata_dir()
    elec_root = elections_dir or get_elections_dir()
    consts = constituencies or load_constituencies_catalog(meta_root / "constituencies_2026.json")
    consts_map = consts.by_cpro()

    cur_rows = {r.cpro: r for r in load_congress_results(election_id, elections_dir=elec_root)}
    prev_map = {
        "2016-06": "2015-12",
        "2019-04": "2016-06",
        "2019-11": "2019-04",
        "2023-07": "2019-11",
    }
    prev_eid = prev_map.get(election_id)
    prev_rows = (
        {r.cpro: r for r in load_congress_results(prev_eid, elections_dir=elec_root)}
        if prev_eid
        else None
    )

    cpros = ALL_CPROS
    # Build canonical party IDs appearing either on ballots in this election or Stage 1
    ballot_parties: set[str] = set()
    for r in cur_rows.values():
        for p, v in r.party_votes.items():
            if v > 0:
                ballot_parties.add(p)
    for p in stage1_mean:
        if p != "OTROS":
            ballot_parties.add(p)
    ballot_parties.add("OTROS")
    party_ids = tuple(
        sorted(ballot_parties, key=lambda p: (p not in stage1_mean, p == "OTROS", p))
    )
    p_idx = {p: j for j, p in enumerate(party_ids)}
    n_c, n_p = len(cpros), len(party_ids)

    # Known proclaimed candidacy mask on election_id ballot
    active_mask = np.zeros((n_c, n_p), dtype=bool)
    for i, cpro in enumerate(cpros):
        pv = cur_rows[cpro].party_votes
        for p, j in p_idx.items():
            if pv.get(p, 0) > 0:
                active_mask[i, j] = True
        if "OTROS" in p_idx and not np.any(active_mask[i]):
            active_mask[i, p_idx["OTROS"]] = True

    prior_mat = np.zeros((n_c, n_p), dtype=np.float64)
    blank_vec = np.zeros(n_c, dtype=np.float64)
    turnout_mean = np.zeros(n_c, dtype=np.float64)
    censo_vec = np.zeros(n_c, dtype=np.float64)
    seats_vec = np.array([cur_rows[c].seats for c in cpros], dtype=np.int32)
    city_vec = np.array([c in AUTONOMOUS_CITY_CPROS for c in cpros], dtype=bool)

    for i, cpro in enumerate(cpros):
        con = consts_map[cpro]
        cod = con.codauto
        # Strict non-leaky censo/turnout baseline from previous election if available
        if prev_rows is not None:
            pr = prev_rows[cpro]
            censo_vec[i] = float(pr.censo)
            turnout_mean[i] = min(0.95, max(0.35, float(pr.validos) / max(1.0, float(pr.censo))))
            blank_vec[i] = min(0.025, max(0.004, float(pr.blancos) / max(1.0, float(pr.validos))))
            p_val = float(pr.validos)
            pv = pr.party_votes

            row = np.zeros(n_p, dtype=np.float64)
            for p, j in p_idx.items():
                if not active_mask[i, j]:
                    continue
                if p == "UP" and election_id == "2016-06":
                    base = (pv.get("PODEMOS", 0) + pv.get("IU", 0)) / p_val
                elif p == "NA_SUMA" and election_id == "2019-04" and cpro == "31":
                    base = pv.get("PP", 0) / p_val
                elif p == "MP" and election_id == "2019-11":
                    base = 0.25 * (pv.get("UP", 0) / p_val) + 0.04 * (pv.get("PSOE", 0) / p_val)
                elif p == "SUMAR" and election_id == "2023-07":
                    base = (
                        pv.get("UP", 0)
                        + pv.get("MP", 0)
                        + pv.get("COMPROMIS", 0)
                        + pv.get("CHA", 0)
                        + pv.get("MES", 0)
                    ) / p_val
                elif p == "PP" and election_id == "2023-07" and cpro == "31":
                    base = 0.55 * (pv.get("NA_SUMA", 0) / p_val)
                elif p == "UPN" and election_id == "2023-07" and cpro == "31":
                    base = 0.45 * (pv.get("NA_SUMA", 0) / p_val)
                elif p == "TERUEL_EXISTE" and pv.get(p, 0) == 0:
                    base = 0.02 if cpro != "44" else 0.15
                else:
                    base = pv.get(p, 0) / p_val
                row[j] = max(0.004 if p in stage1_mean else 0.001, base)
            prior_mat[i] = _exact_normalize_row(row, active_mask[i], float(blank_vec[i]))
        else:
            # 2015-12: purely structural pre-2015 geography + Stage 1 national level
            con_censo = float(con.censo_2023 or 500_000)
            censo_vec[i] = con_censo
            turnout_mean[i] = 0.72
            blank_vec[i] = 0.008
            row = np.zeros(n_p, dtype=np.float64)
            for p, j in p_idx.items():
                if not active_mask[i, j]:
                    continue
                nat_s = max(0.003, float(stage1_mean.get(p, 0.005)))
                geo_mult = _NAT_GEO_PRIOR_2015.get(p, {}).get(cod, 1.0)
                if p not in stage1_mean:
                    # Regional party not tracked nationally in 2015 polls (ERC/JUNTS/PNV/BILDU/CCA)
                    geo_mult = 12.0 if p in {"ERC", "JUNTS", "PNV", "EH_BILDU"} else 4.0
                row[j] = nat_s * geo_mult
            prior_mat[i] = _exact_normalize_row(row, active_mask[i], float(blank_vec[i]))

    exp_valid = censo_vec * turnout_mean
    turnout_weights = exp_valid / float(exp_valid.sum())
    blank_nat = float(turnout_weights @ blank_vec)

    target_dict, tracked_idx = build_ipf_national_targets(
        party_ids,
        prior_mat,
        turnout_weights,
        stage1_mean,
        blank_nat,
        election_id=election_id,
    )
    target_vec = np.array([target_dict[p] for p in party_ids], dtype=np.float64)
    pre_mat, max_err = rake_provincial_ipf(
        prior_mat,
        active_mask,
        blank_vec,
        turnout_weights,
        target_vec,
        tracked_indices=None,
        n_iters=hyperparams.stage2_ipf_iters,
    )
    final_mat = pre_mat.copy()
    pre_nat = turnout_weights @ pre_mat
    post_nat = turnout_weights @ final_mat

    codautos = tuple(consts_map[c].codauto for c in cpros)
    districts: dict[str, Stage2DistrictResult] = {}
    for i, cpro in enumerate(cpros):
        con = consts_map[cpro]
        b_i = float(blank_vec[i])
        districts[cpro] = Stage2DistrictResult(
            cpro=cpro,
            name=con.name,
            codauto=con.codauto,
            seats=int(seats_vec[i]),
            censo=int(round(float(censo_vec[i]))),
            validos_prior=int(round(float(exp_valid[i]))),
            turnout_mean=float(turnout_mean[i]),
            turnout_weight=float(turnout_weights[i]),
            prior_shares={party_ids[j]: float(prior_mat[i, j]) for j in range(n_p)},
            pre_shrinkage_ipf_shares={party_ids[j]: float(pre_mat[i, j]) for j in range(n_p)},
            pre_shrinkage_blank_share=b_i,
            final_shares={party_ids[j]: float(final_mat[i, j]) for j in range(n_p)},
            blank_share=b_i,
            cis_weight=0.0,
            regional_poll_weight=0.0,
        )

    return Stage2Projection(
        as_of_date=as_of,
        election_id=election_id,
        party_ids=party_ids,
        cpros=cpros,
        codautos=codautos,
        districts=districts,
        stage1_targets={k: float(v) for k, v in stage1_mean.items()},
        target_national=target_dict,
        pre_shrinkage_national={party_ids[j]: float(pre_nat[j]) for j in range(n_p)},
        post_shrinkage_national={party_ids[j]: float(post_nat[j]) for j in range(n_p)},
        ipf_max_marginal_err=max_err,
        active_mask=active_mask,
        turnout_weights=turnout_weights,
        prior_matrix=prior_mat,
        pre_shrinkage_matrix=pre_mat,
        pre_shrinkage_blank=blank_vec,
        final_matrix=final_mat,
        final_blank=blank_vec.copy(),
        censo_vec=censo_vec,
        turnout_mean_vec=turnout_mean,
        seats_vec=seats_vec,
        city_vec=city_vec,
    )
