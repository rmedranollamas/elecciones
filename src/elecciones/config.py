"""Centralized configuration, paths, LOREG legal parameters, and model hyperparameters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import httpx

MADRID_TZ: Final[ZoneInfo] = ZoneInfo("Europe/Madrid")

CONGRESS_TOTAL_SEATS: Final[int] = 350
SENATE_DIRECT_SEATS: Final[int] = 208
CONSTITUENCY_COUNT: Final[int] = 52
PROVINCE_COUNT: Final[int] = 50
DEFAULT_PROVINCE_THRESHOLD: Final[float] = 0.03
AUTONOMOUS_CITY_CPROS: Final[frozenset[str]] = frozenset({"51", "52"})

ELECTION_DATE_2026: Final[date] = date(2026, 11, 29)
ELECTION_DATE: Final[date] = ELECTION_DATE_2026
BLACKOUT_CUTOFF_DATE_2026: Final[date] = date(2026, 11, 23)
BLACKOUT_START_CET: Final[datetime] = datetime(2026, 11, 24, 0, 0, 0, tzinfo=MADRID_TZ)
BLACKOUT_END_CET: Final[datetime] = datetime(2026, 11, 29, 20, 0, 0, tzinfo=MADRID_TZ)
BLACKOUT_BANNER: Final[str] = (
    "LOREG Art. 69.7 blackout window active "
    "(2026-11-24 00:00 CET to 2026-11-29 20:00 CET); "
    "skipping poll fetching and automated forecast publication."
)

SIMULATION_SEED: Final[int] = 20261129
DEFAULT_N_SIMS: Final[int] = 50_000
BACKTEST_DEFAULT_N_SIMS: Final[int] = 2_000

HISTORICAL_ELECTION_DATES: Final[dict[str, date]] = {
    "2015-12": date(2015, 12, 20),
    "2016-06": date(2016, 6, 26),
    "2019-04": date(2019, 4, 28),
    "2019-11": date(2019, 11, 10),
    "2023-07": date(2023, 7, 23),
}

SEAT_TIER_THRESHOLDS: Final[dict[str, float]] = {
    "seguro": 0.99,
    "probable": 0.75,
    "en_juego": 0.25,
    "remoto": 0.05,
}


@dataclass(frozen=True, slots=True)
class ModelHyperparameters:
    """Default Stage 1-4 hyperparameters (updated by backtest calibration)."""

    half_life_campaign_days: float = 7.5
    half_life_quiet_days: float = 14.0
    sample_size_cap_general: int = 4000
    sample_size_cap_cis: int = 2000
    prior_mae_anchor: float = 1.80
    eb_rating_tau: float = 3.0
    eb_house_tau: float = 3.0
    herding_ratio_threshold: float = 0.55
    state_space_rw_var_per_day: float = 0.0014
    state_space_obs_scale: float = 0.045
    ensemble_weight_kernel: float = 0.45
    ensemble_weight_state_space: float = 0.55
    student_t_df: float = 5.0
    stage2_ipf_iters: int = 80
    stage2_prior_weight: float = 900.0
    stage2_cis_max_frac: float = 0.30
    stage2_cis_divisor: float = 2.5
    stage2_regional_divisor: float = 1.5
    sim_sample_size: int = 200


DEFAULT_HYPERPARAMETERS: Final[ModelHyperparameters] = ModelHyperparameters()

HTTP_USER_AGENT: Final[str] = (
    "elecciones-29n-predictor/0.2.0 (https://github.com/rmedranollamas/elecciones)"
)


def get_repo_root() -> Path:
    """Resolve repository root, honoring ``ELECCIONES_REPO_ROOT`` if set."""
    override = os.environ.get("ELECCIONES_REPO_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


def get_data_dir() -> Path:
    return get_repo_root() / "data"


def get_metadata_dir() -> Path:
    return get_data_dir() / "metadata"


def get_elections_dir() -> Path:
    return get_data_dir() / "elections"


def get_polls_dir() -> Path:
    return get_data_dir() / "polls"


def get_outputs_dir() -> Path:
    return get_data_dir() / "outputs"


def get_raw_cache_dir() -> Path:
    return get_data_dir() / "raw_cache"


def get_docs_dir() -> Path:
    return get_repo_root() / "docs"


def is_in_blackout_window(now: str | datetime | None = None) -> bool:
    """Return True if ``now`` falls inside LOREG Art. 69.7 blackout window (Europe/Madrid).

    The prohibition runs from ``2026-11-24 00:00 CET`` until polls close at
    ``2026-11-29 20:00 CET``.
    """
    if now is None:
        dt = datetime.now(MADRID_TZ)
    elif isinstance(now, str):
        raw = now.strip()
        if not raw:
            dt = datetime.now(MADRID_TZ)
        elif len(raw) == 10:
            d = date.fromisoformat(raw)
            dt = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=MADRID_TZ)
        else:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            dt = (
                parsed.replace(tzinfo=MADRID_TZ)
                if parsed.tzinfo is None
                else parsed.astimezone(MADRID_TZ)
            )
    elif isinstance(now, datetime):
        dt = now.replace(tzinfo=MADRID_TZ) if now.tzinfo is None else now.astimezone(MADRID_TZ)
    else:
        raise TypeError(f"Unsupported type for 'now': {type(now).__name__}")

    return BLACKOUT_START_CET <= dt < BLACKOUT_END_CET


def make_http_client(timeout: float = 30.0) -> httpx.Client:
    """Create a configured ``httpx.Client`` honoring proxy and CA env settings."""
    import ssl

    import httpx

    ca_bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    ssl_ctx: ssl.SSLContext | bool = (
        ssl.create_default_context(cafile=ca_bundle)
        if ca_bundle and Path(ca_bundle).is_file()
        else True
    )
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    mounts: dict[str, httpx.BaseTransport] = {}
    if http_proxy:
        mounts["http://"] = httpx.HTTPTransport(proxy=http_proxy, verify=ssl_ctx)
    if https_proxy:
        mounts["https://"] = httpx.HTTPTransport(proxy=https_proxy, verify=ssl_ctx)
    # Bypass proxy for loopback addresses cleanly (avoids httpx NO_PROXY='[::1]' URL parse crash).
    for host in ("localhost", "127.0.0.1"):
        mounts[f"http://{host}"] = httpx.HTTPTransport()
        mounts[f"https://{host}"] = httpx.HTTPTransport(verify=ssl_ctx)

    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        verify=ssl_ctx,
        mounts=mounts if mounts else None,
        trust_env=False,
        headers={
            "User-Agent": HTTP_USER_AGENT,
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        },
    )
