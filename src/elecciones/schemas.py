"""Core typed schemas, validators, and serialization helpers for ``elecciones``.

Conventions enforced across the pipeline:
1. **INE Codes**: ``cpro`` is always a zero-padded 2-digit string in ``"01".."52"``;
   ``codauto`` is always a zero-padded 2-digit string in ``"01".."19"`` (official INE
   coding: ``"07"`` = Castilla y León, ``"08"`` = Castilla-La Mancha).
2. **Poll CSV Scale vs. Model Simplex Scale**:
   - Poll CSV snapshots (``national_*.csv``, ``regional_2026.csv``,
     ``manual_polls_2026.csv``) store party figures strictly in **percentage points**
     ``[0.0, 100.0]``, leaving unreported minor parties blank/``NaN``.
   - In-memory ``PollRecord`` and ``RegionalPollRecord`` store ``party_shares`` on the
     **unit simplex scale** ``[0.0, 1.0]`` for reported parties only (never coercing
     missing/``NaN`` cells to ``0.0``).
3. **Full Float64 Serialization**: Provincial share dictionaries
   (``pre_shrinkage_ipf_shares``, ``final_shares``) preserve full IEEE-754 ``float64``
   precision on JSON serialization so ``1.0 ± 1e-9`` simplex invariants hold after
   round-tripping.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

DistrictType: TypeAlias = Literal["province", "autonomous_city"]
NationalGroup: TypeAlias = Literal["PP", "PSOE", "VOX", "IZQUIERDAS", "OTROS"]
PollScopeType: TypeAlias = Literal["ccaa", "province"]
ManualPollAction: TypeAlias = Literal["add", "override", "exclude"]
SeatTier: TypeAlias = Literal["seguro", "probable", "en_juego", "remoto"]
ForecastEngine: TypeAlias = Literal["ensemble", "kernel", "state_space"]

POLL_CSV_SCALE: Final[str] = "percentage_points_0_100"
MODEL_SHARE_SCALE: Final[str] = "unit_simplex_0_1"
SIMPLEX_TOL: Final[float] = 1e-9
CONGRESS_TOTAL_SEATS: Final[int] = 350
SENATE_DIRECT_SEATS: Final[int] = 208

ALL_CPROS: Final[tuple[str, ...]] = tuple(f"{i:02d}" for i in range(1, 53))
ALL_CODAUTOS: Final[tuple[str, ...]] = tuple(f"{i:02d}" for i in range(1, 20))
AUTONOMOUS_CITY_CPROS: Final[frozenset[str]] = frozenset({"51", "52"})
VALID_CPROS_SET: Final[frozenset[str]] = frozenset(ALL_CPROS)
VALID_CODAUTOS_SET: Final[frozenset[str]] = frozenset(ALL_CODAUTOS)
NATIONAL_GROUPS: Final[frozenset[str]] = frozenset(
    {"PP", "PSOE", "VOX", "IZQUIERDAS", "OTROS"}
)
SEAT_TIERS: Final[frozenset[str]] = frozenset(
    {"seguro", "probable", "en_juego", "remoto"}
)
FORECAST_ENGINES: Final[frozenset[str]] = frozenset(
    {"ensemble", "kernel", "state_space"}
)
MANUAL_POLL_ACTIONS: Final[frozenset[str]] = frozenset({"add", "override", "exclude"})
MISSING_CELL_TOKENS: Final[frozenset[str]] = frozenset(
    {"", "nan", "na", "n/a", "null", "none", "?", "-", "\u2013", "\u2014"}
)

_HEX_COLOR_RE: Final[re.Pattern[str]] = re.compile(r"^#[0-9A-Fa-f]{6}$")

POLL_METADATA_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "election_id",
        "pollster",
        "media",
        "fieldwork_start",
        "fieldwork_end",
        "pub_date",
        "sample_size",
        "turnout",
        "is_cis",
        "action",
        "scope_type",
        "scope_code",
        "lead",
        "notes",
        "url",
    }
)


class SchemaValidationError(ValueError):
    """Raised when an input record, JSON payload, or CSV row violates a contract."""


class ThresholdExhaustionError(ValueError):
    """Raised in strict mode when positive valid votes yield zero eligible lists."""


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(
            f"{context} must be a JSON object/mapping, got {type(value).__name__}"
        )
    return value


def _require_sequence_not_str(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SchemaValidationError(
            f"{context} must be a list/sequence, got {type(value).__name__}"
        )
    return value


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(
            f"{field_name} must be a boolean, got {type(value).__name__}"
        )
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(
            f"{field_name} must be an integer, got {type(value).__name__}"
        )
    return value


def _require_number(value: Any, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise SchemaValidationError(f"{field_name} must be finite, got {value!r}")
    return value


def _require_float(value: Any, field_name: str) -> float:
    return float(_require_number(value, field_name))


def normalize_cpro(value: str | int) -> str:
    """Normalize an INE constituency code to a 2-digit string ``"01".."52"``."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SchemaValidationError(f"Invalid cpro type: {type(value).__name__}")
    raw = str(value).strip()
    if not raw.isdigit():
        raise SchemaValidationError(f"Invalid cpro value: {value!r}")
    norm = raw.zfill(2)
    if norm not in VALID_CPROS_SET:
        raise SchemaValidationError(f"cpro out of range '01'..'52': {value!r}")
    return norm


def normalize_codauto(value: str | int) -> str:
    """Normalize an INE autonomous-community code to ``"01".."19"``."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SchemaValidationError(f"Invalid codauto type: {type(value).__name__}")
    raw = str(value).strip()
    if not raw.isdigit():
        raise SchemaValidationError(f"Invalid codauto value: {value!r}")
    norm = raw.zfill(2)
    if norm not in VALID_CODAUTOS_SET:
        raise SchemaValidationError(f"codauto out of range '01'..'19': {value!r}")
    return norm


def parse_iso_date(value: str | date) -> date:
    """Parse an ISO ``YYYY-MM-DD`` string or return an existing ``date`` object."""
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise SchemaValidationError(
            f"Expected ISO date string, got {type(value).__name__}"
        )
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise SchemaValidationError(f"Invalid ISO date {value!r}") from exc


def parse_bool_token(value: Any, field_name: str = "bool") -> bool:
    """Parse boolean values from JSON or CSV strings (``true/false/1/0/yes/no``)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        tok = value.strip().lower()
        if tok in {"true", "1", "yes", "t", "y"}:
            return True
        if tok in {"false", "0", "no", "f", "n", ""}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise SchemaValidationError(f"Invalid boolean token for {field_name}: {value!r}")


def pct_to_share(pct: float) -> float:
    """Convert percentage points ``[0.0, 100.0]`` to unit share ``[0.0, 1.0]``."""
    val = _require_float(pct, "pct")
    if val < 0.0 or val > 100.0:
        raise SchemaValidationError(f"Percentage point value out of [0, 100]: {val}")
    return val / 100.0


def share_to_pct(share: float) -> float:
    """Convert unit share ``[0.0, 1.0]`` to percentage points ``[0.0, 100.0]``."""
    val = _require_float(share, "share")
    if val < 0.0 or val > 1.0:
        raise SchemaValidationError(f"Unit share value out of [0, 1]: {val}")
    return val * 100.0


def _is_missing_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in MISSING_CELL_TOKENS


def classify_seat_tier(prob: float) -> SeatTier | None:
    """Classify marginal seat probability ``P(S_{c,p} >= k)`` per El País tiers."""
    p = _require_float(prob, "prob")
    if p < 0.0 or p > 1.0:
        raise SchemaValidationError(f"Seat probability must be in [0, 1], got {p}")
    if p >= 0.99:
        return "seguro"
    if p >= 0.75:
        return "probable"
    if p >= 0.25:
        return "en_juego"
    if p >= 0.05:
        return "remoto"
    return None


def district_simplex_sum(
    shares: Mapping[str, float], blank_share: float = 0.0
) -> float:
    """Compute valid-vote simplex sum with ``BLANK`` inside ``shares`` or separate."""
    total = sum(_require_float(v, f"shares[{k}]") for k, v in shares.items())
    has_blank_key = any(k.upper() == "BLANK" for k in shares)
    return (
        total if has_blank_key else total + _require_float(blank_share, "blank_share")
    )


@dataclass(frozen=True, slots=True)
class SubdistrictSenateSpec:
    """Insular Senate constituency specification under LOREG Art. 165.1.b."""

    id: str
    name: str
    direct_seats: int
    max_votes_per_ballot: int

    def __post_init__(self) -> None:
        _require_non_empty_str(self.id, "subdistrict.id")
        _require_non_empty_str(self.name, "subdistrict.name")
        seats = _require_int(self.direct_seats, "subdistrict.direct_seats")
        max_v = _require_int(
            self.max_votes_per_ballot, "subdistrict.max_votes_per_ballot"
        )
        if seats < 1:
            raise SchemaValidationError(
                f"subdistrict.direct_seats must be >= 1, got {seats}"
            )
        if max_v < 1 or max_v > seats:
            raise SchemaValidationError(
                f"subdistrict.max_votes_per_ballot ({max_v}) must be in 1..{seats}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SubdistrictSenateSpec:
        d = _require_mapping(data, "SubdistrictSenateSpec")
        return cls(
            id=_require_non_empty_str(d.get("id"), "subdistrict.id"),
            name=_require_non_empty_str(d.get("name"), "subdistrict.name"),
            direct_seats=_require_int(
                d.get("direct_seats"), "subdistrict.direct_seats"
            ),
            max_votes_per_ballot=_require_int(
                d.get("max_votes_per_ballot"), "subdistrict.max_votes_per_ballot"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "direct_seats": self.direct_seats,
            "max_votes_per_ballot": self.max_votes_per_ballot,
        }


@dataclass(frozen=True, slots=True)
class SenateSpec:
    """Extensible Senate rules per constituency under LOREG Arts. 165-166."""

    direct_seats: int
    max_votes_per_ballot: int
    subdistricts: tuple[SubdistrictSenateSpec, ...] = ()

    def __post_init__(self) -> None:
        seats = _require_int(self.direct_seats, "senate.direct_seats")
        max_v = _require_int(self.max_votes_per_ballot, "senate.max_votes_per_ballot")
        if seats < 1:
            raise SchemaValidationError(
                f"senate.direct_seats must be >= 1, got {seats}"
            )
        if max_v < 1 or max_v > seats:
            raise SchemaValidationError(
                f"senate.max_votes_per_ballot ({max_v}) must be in 1..{seats}"
            )
        if self.subdistricts:
            sub_ids = [s.id for s in self.subdistricts]
            if len(sub_ids) != len(set(sub_ids)):
                raise SchemaValidationError(
                    "Duplicate subdistrict IDs in senate.subdistricts"
                )
            sub_sum = sum(s.direct_seats for s in self.subdistricts)
            if sub_sum != seats:
                raise SchemaValidationError(
                    f"Subdistrict Senate seats ({sub_sum}) != direct_seats ({seats})"
                )
            max_sub_v = max(s.max_votes_per_ballot for s in self.subdistricts)
            if max_sub_v != max_v:
                raise SchemaValidationError(
                    f"max_votes_per_ballot ({max_v}) != max subdistrict ({max_sub_v})"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SenateSpec:
        d = _require_mapping(data, "SenateSpec")
        raw_subs = _require_sequence_not_str(
            d.get("subdistricts", []), "senate.subdistricts"
        )
        return cls(
            direct_seats=_require_int(d.get("direct_seats"), "senate.direct_seats"),
            max_votes_per_ballot=_require_int(
                d.get("max_votes_per_ballot"), "senate.max_votes_per_ballot"
            ),
            subdistricts=tuple(SubdistrictSenateSpec.from_dict(s) for s in raw_subs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_seats": self.direct_seats,
            "max_votes_per_ballot": self.max_votes_per_ballot,
            "subdistricts": [s.to_dict() for s in self.subdistricts],
        }


@dataclass(frozen=True, slots=True)
class Constituency:
    """Metadata for one of Spain's 52 Congress constituencies."""

    cpro: str
    name: str
    short_name: str
    codauto: str
    ccaa_name: str
    district_type: DistrictType
    threshold: float
    seats_2019: int
    seats_2023: int
    seats_2026: int
    cartogram_row: int
    cartogram_col: int
    senate: SenateSpec
    censo_2023: int | None = None
    validos_2023: int | None = None

    def __post_init__(self) -> None:
        norm_cpro = normalize_cpro(self.cpro)
        if norm_cpro != self.cpro:
            object.__setattr__(self, "cpro", norm_cpro)
        norm_cod = normalize_codauto(self.codauto)
        if norm_cod != self.codauto:
            object.__setattr__(self, "codauto", norm_cod)

        _require_non_empty_str(self.name, "constituency.name")
        _require_non_empty_str(self.short_name, "constituency.short_name")
        _require_non_empty_str(self.ccaa_name, "constituency.ccaa_name")

        if self.district_type not in ("province", "autonomous_city"):
            raise SchemaValidationError(
                f"Invalid district_type: {self.district_type!r}"
            )

        is_city = self.cpro in AUTONOMOUS_CITY_CPROS
        if is_city and self.district_type != "autonomous_city":
            raise SchemaValidationError(
                f"cpro {self.cpro} must have district_type='autonomous_city'"
            )
        if not is_city and self.district_type != "province":
            raise SchemaValidationError(
                f"cpro {self.cpro} must have district_type='province'"
            )

        thresh = _require_float(self.threshold, "constituency.threshold")
        if is_city and abs(thresh) > 1e-12:
            raise SchemaValidationError(
                f"Autonomous city {self.cpro} must have threshold=0.0"
            )
        if not is_city and abs(thresh - 0.03) > 1e-12:
            raise SchemaValidationError(
                f"Province {self.cpro} must have threshold=0.03, got {thresh}"
            )

        for attr in ("seats_2019", "seats_2023", "seats_2026"):
            val = _require_int(getattr(self, attr), attr)
            if is_city and val != 1:
                raise SchemaValidationError(
                    f"Autonomous city {self.cpro} must have {attr}=1, got {val}"
                )
            if not is_city and val < 2:
                raise SchemaValidationError(
                    f"Province {self.cpro} must have {attr} >= 2, got {val}"
                )

        if _require_int(self.cartogram_row, "cartogram_row") < 0:
            raise SchemaValidationError("cartogram_row must be >= 0")
        if _require_int(self.cartogram_col, "cartogram_col") < 0:
            raise SchemaValidationError("cartogram_col must be >= 0")

        if (
            self.censo_2023 is not None
            and _require_int(self.censo_2023, "censo_2023") <= 0
        ):
            raise SchemaValidationError("censo_2023 must be > 0")
        if (
            self.validos_2023 is not None
            and _require_int(self.validos_2023, "validos_2023") <= 0
        ):
            raise SchemaValidationError("validos_2023 must be > 0")

    @property
    def is_autonomous_city(self) -> bool:
        return self.district_type == "autonomous_city"

    def seats_for_election(self, election_id: str) -> int:
        if election_id.startswith("2019"):
            return self.seats_2019
        if election_id.startswith("2023"):
            return self.seats_2023
        if election_id.startswith("2026"):
            return self.seats_2026
        raise SchemaValidationError(
            f"No static seat vector on Constituency for {election_id!r}"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Constituency:
        d = _require_mapping(data, "Constituency")
        dtype = _require_non_empty_str(d.get("district_type"), "district_type")
        if dtype not in ("province", "autonomous_city"):
            raise SchemaValidationError(f"Invalid district_type: {dtype!r}")
        dtype_lit: DistrictType = (
            "autonomous_city" if dtype == "autonomous_city" else "province"
        )
        censo = d.get("censo_2023")
        validos = d.get("validos_2023")
        return cls(
            cpro=normalize_cpro(d.get("cpro", "")),
            name=_require_non_empty_str(d.get("name"), "name"),
            short_name=_require_non_empty_str(d.get("short_name"), "short_name"),
            codauto=normalize_codauto(d.get("codauto", "")),
            ccaa_name=_require_non_empty_str(d.get("ccaa_name"), "ccaa_name"),
            district_type=dtype_lit,
            threshold=_require_float(d.get("threshold"), "threshold"),
            seats_2019=_require_int(d.get("seats_2019"), "seats_2019"),
            seats_2023=_require_int(d.get("seats_2023"), "seats_2023"),
            seats_2026=_require_int(d.get("seats_2026"), "seats_2026"),
            cartogram_row=_require_int(d.get("cartogram_row"), "cartogram_row"),
            cartogram_col=_require_int(d.get("cartogram_col"), "cartogram_col"),
            senate=SenateSpec.from_dict(_require_mapping(d.get("senate"), "senate")),
            censo_2023=_require_int(censo, "censo_2023") if censo is not None else None,
            validos_2023=_require_int(validos, "validos_2023")
            if validos is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cpro": self.cpro,
            "name": self.name,
            "short_name": self.short_name,
            "codauto": self.codauto,
            "ccaa_name": self.ccaa_name,
            "district_type": self.district_type,
            "threshold": self.threshold,
            "seats_2019": self.seats_2019,
            "seats_2023": self.seats_2023,
            "seats_2026": self.seats_2026,
            "cartogram_row": self.cartogram_row,
            "cartogram_col": self.cartogram_col,
            "senate": self.senate.to_dict(),
        }
        if self.censo_2023 is not None:
            out["censo_2023"] = self.censo_2023
        if self.validos_2023 is not None:
            out["validos_2023"] = self.validos_2023
        return out


@dataclass(frozen=True, slots=True)
class ConstituenciesCatalog:
    """Validated 52-district catalog for ``data/metadata/constituencies_2026.json``."""

    constituencies: tuple[Constituency, ...]
    chamber: str = "congress"
    election_date: str = "2026-11-29"
    total_seats: int = 350
    boe_source: str = "BOE-A-2026-20742"
    total_senate_direct_seats: int | None = 208

    def __post_init__(self) -> None:
        self.validate(expected_senate_seats=self.total_senate_direct_seats)

    def validate(self, expected_senate_seats: int | None = 208) -> None:
        _require_non_empty_str(self.chamber, "chamber")
        parse_iso_date(self.election_date)
        total = _require_int(self.total_seats, "total_seats")
        if len(self.constituencies) != 52:
            raise SchemaValidationError(
                f"Expected 52 constituencies, got {len(self.constituencies)}"
            )
        cpros = [c.cpro for c in self.constituencies]
        if set(cpros) != VALID_CPROS_SET:
            missing = sorted(VALID_CPROS_SET - set(cpros))
            raise SchemaValidationError(
                f"Missing/duplicate cpro codes; missing={missing}"
            )

        codauto_names: dict[str, str] = {}
        for c in self.constituencies:
            prev = codauto_names.setdefault(c.codauto, c.ccaa_name)
            if prev != c.ccaa_name:
                raise SchemaValidationError(
                    f"ccaa_name mismatch on {c.codauto}: {prev!r} vs {c.ccaa_name!r}"
                )
        if set(codauto_names.keys()) != VALID_CODAUTOS_SET:
            missing_cod = sorted(VALID_CODAUTOS_SET - set(codauto_names.keys()))
            raise SchemaValidationError(f"Missing INE codauto codes: {missing_cod}")

        for attr, expected in (
            ("seats_2019", 350),
            ("seats_2023", 350),
            ("seats_2026", total),
        ):
            s = sum(getattr(c, attr) for c in self.constituencies)
            if s != expected:
                raise SchemaValidationError(
                    f"Sum of {attr} is {s}, expected {expected}"
                )

        coords = [(c.cartogram_row, c.cartogram_col) for c in self.constituencies]
        if len(coords) != len(set(coords)):
            raise SchemaValidationError(
                "Duplicate (cartogram_row, cartogram_col) coordinates detected"
            )

        if expected_senate_seats is not None:
            sen_sum = sum(c.senate.direct_seats for c in self.constituencies)
            if sen_sum != expected_senate_seats:
                raise SchemaValidationError(
                    f"senate.direct_seats sum {sen_sum} != {expected_senate_seats}"
                )

        if self.election_date == "2026-11-29":
            by_id = {c.cpro: c for c in self.constituencies}
            if by_id["28"].seats_2026 != 38:
                raise SchemaValidationError(
                    f"Madrid ('28') seats_2026={by_id['28'].seats_2026}, expected 38"
                )
            if by_id["11"].seats_2026 != 8:
                raise SchemaValidationError(
                    f"Cádiz ('11') requires seats_2026=8, got {by_id['11'].seats_2026}"
                )

    def by_cpro(self) -> dict[str, Constituency]:
        return {c.cpro: c for c in self.constituencies}

    def get(self, cpro: str | int) -> Constituency:
        norm = normalize_cpro(cpro)
        for c in self.constituencies:
            if c.cpro == norm:
                return c
        raise KeyError(norm)

    def by_codauto(self) -> dict[str, list[Constituency]]:
        grouped: dict[str, list[Constituency]] = {}
        for c in self.constituencies:
            grouped.setdefault(c.codauto, []).append(c)
        return grouped

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstituenciesCatalog:
        d = _require_mapping(data, "ConstituenciesCatalog")
        raw_items = _require_sequence_not_str(d.get("constituencies"), "constituencies")
        sen_total = d.get("total_senate_direct_seats", 208)
        return cls(
            chamber=_require_non_empty_str(d.get("chamber", "congress"), "chamber"),
            election_date=_require_non_empty_str(
                d.get("election_date", "2026-11-29"), "election_date"
            ),
            total_seats=_require_int(d.get("total_seats", 350), "total_seats"),
            boe_source=_require_non_empty_str(
                d.get("boe_source", "BOE-A-2026-20742"), "boe_source"
            ),
            total_senate_direct_seats=(
                _require_int(sen_total, "total_senate_direct_seats")
                if sen_total is not None
                else None
            ),
            constituencies=tuple(Constituency.from_dict(item) for item in raw_items),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "chamber": self.chamber,
            "election_date": self.election_date,
            "boe_source": self.boe_source,
            "total_seats": self.total_seats,
        }
        if self.total_senate_direct_seats is not None:
            out["total_senate_direct_seats"] = self.total_senate_direct_seats
        out["constituencies"] = [c.to_dict() for c in self.constituencies]
        return out

    @classmethod
    def load_json(cls, path: Path) -> ConstituenciesCatalog:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class PartySpec:
    """Canonical party/coalition metadata and 52-district BOE ballot mask."""

    id: str
    name: str
    short_name: str
    color: str
    national_group: NationalGroup
    bloc: str
    boe_mask_2026: dict[str, int]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_str(self.id, "party.id")
        _require_non_empty_str(self.name, "party.name")
        _require_non_empty_str(self.short_name, "party.short_name")
        _require_non_empty_str(self.bloc, "party.bloc")
        if not isinstance(self.color, str) or not _HEX_COLOR_RE.match(self.color):
            raise SchemaValidationError(
                f"party.color must match #RRGGBB, got {self.color!r}"
            )
        if self.national_group not in NATIONAL_GROUPS:
            raise SchemaValidationError(
                f"Invalid national_group {self.national_group!r}"
            )

        normalized_mask: dict[str, int] = {}
        for k, v in _require_mapping(self.boe_mask_2026, "boe_mask_2026").items():
            norm_k = normalize_cpro(k)
            iv = _require_int(v, f"boe_mask_2026[{k}]")
            if iv not in (0, 1):
                raise SchemaValidationError(
                    f"boe_mask_2026[{norm_k}] must be 0 or 1, got {iv}"
                )
            normalized_mask[norm_k] = iv
        if set(normalized_mask.keys()) != VALID_CPROS_SET:
            missing = sorted(VALID_CPROS_SET - set(normalized_mask.keys()))
            raise SchemaValidationError(
                f"Party {self.id} boe_mask_2026 missing cpros: {missing}"
            )
        object.__setattr__(
            self, "boe_mask_2026", {c: normalized_mask[c] for c in ALL_CPROS}
        )

    def contests(self, cpro: str | int) -> bool:
        return self.boe_mask_2026[normalize_cpro(cpro)] == 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PartySpec:
        d = _require_mapping(data, "PartySpec")
        ng = _require_non_empty_str(d.get("national_group"), "national_group")
        if ng == "PP":
            ng_lit: NationalGroup = "PP"
        elif ng == "PSOE":
            ng_lit = "PSOE"
        elif ng == "VOX":
            ng_lit = "VOX"
        elif ng == "IZQUIERDAS":
            ng_lit = "IZQUIERDAS"
        elif ng == "OTROS":
            ng_lit = "OTROS"
        else:
            raise SchemaValidationError(f"Invalid national_group: {ng!r}")
        raw_aliases = _require_sequence_not_str(d.get("aliases", ()), "party.aliases")
        raw_mask = _require_mapping(d.get("boe_mask_2026"), "boe_mask_2026")
        return cls(
            id=_require_non_empty_str(d.get("id"), "id"),
            name=_require_non_empty_str(d.get("name"), "name"),
            short_name=_require_non_empty_str(d.get("short_name"), "short_name"),
            color=_require_non_empty_str(d.get("color"), "color"),
            national_group=ng_lit,
            bloc=_require_non_empty_str(d.get("bloc"), "bloc"),
            boe_mask_2026={
                str(k): _require_int(v, f"boe_mask_2026[{k}]")
                for k, v in raw_mask.items()
            },
            aliases=tuple(_require_non_empty_str(a, "alias") for a in raw_aliases),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "short_name": self.short_name,
            "color": self.color,
            "national_group": self.national_group,
            "bloc": self.bloc,
            "boe_mask_2026": dict(self.boe_mask_2026),
        }
        if self.aliases:
            out["aliases"] = list(self.aliases)
        return out


@dataclass(frozen=True, slots=True)
class PartiesCatalog:
    """Validated party registry backing ``data/metadata/parties_2026.json``."""

    parties: tuple[PartySpec, ...]
    election_date: str = "2026-11-29"
    boe_source: str = "BOE-A-2026-23103"

    def __post_init__(self) -> None:
        parse_iso_date(self.election_date)
        _require_non_empty_str(self.boe_source, "boe_source")
        if not self.parties:
            raise SchemaValidationError("PartiesCatalog.parties must not be empty")
        ids = [p.id for p in self.parties]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("Duplicate party IDs in PartiesCatalog")
        for cpro in ALL_CPROS:
            if sum(p.boe_mask_2026[cpro] for p in self.parties) < 1:
                raise SchemaValidationError(
                    f"District {cpro} has 0 proclaimed candidacies across all parties"
                )

    def party_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.parties)

    def get(self, party_id: str) -> PartySpec:
        for p in self.parties:
            if p.id == party_id:
                return p
        raise KeyError(party_id)

    def mask_matrix(self, cpros: Sequence[str] = ALL_CPROS) -> list[list[int]]:
        """Return ``len(cpros) x len(parties)`` binary matrix M_{c,p} in {0, 1}."""
        return [
            [p.boe_mask_2026[normalize_cpro(c)] for p in self.parties] for c in cpros
        ]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PartiesCatalog:
        d = _require_mapping(data, "PartiesCatalog")
        raw_parties = _require_sequence_not_str(d.get("parties"), "parties")
        return cls(
            election_date=_require_non_empty_str(
                d.get("election_date", "2026-11-29"), "election_date"
            ),
            boe_source=_require_non_empty_str(
                d.get("boe_source", "BOE-A-2026-23103"), "boe_source"
            ),
            parties=tuple(PartySpec.from_dict(p) for p in raw_parties),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "election_date": self.election_date,
            "boe_source": self.boe_source,
            "parties": [p.to_dict() for p in self.parties],
        }

    @classmethod
    def load_json(cls, path: Path) -> PartiesCatalog:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _parse_csv_party_shares(
    row: Mapping[str, Any], party_columns: Sequence[str] | None = None
) -> dict[str, float]:
    """Parse percentage-point CSV columns ``[0, 100]`` into unit shares ``[0, 1]``."""
    cols = (
        list(party_columns)
        if party_columns is not None
        else [k for k in row if k not in POLL_METADATA_COLUMNS]
    )
    shares: dict[str, float] = {}
    for col in cols:
        raw = row.get(col)
        if _is_missing_cell(raw):
            continue
        pct = (
            float(str(raw).strip())
            if isinstance(raw, str)
            else _require_float(raw, col)
        )
        shares[col] = pct_to_share(pct)
    return shares


@dataclass(frozen=True, slots=True)
class PollRecord:
    """Normalized national opinion poll with reported shares on ``[0.0, 1.0]``."""

    election_id: str
    pollster: str
    fieldwork_start: date
    fieldwork_end: date
    party_shares: dict[str, float]
    media: str = ""
    pub_date: date | None = None
    sample_size: int | None = None
    turnout: float | None = None
    is_cis: bool = False
    allow_empty_shares: bool = False

    def __post_init__(self) -> None:
        _require_non_empty_str(self.election_id, "election_id")
        _require_non_empty_str(self.pollster, "pollster")
        start = parse_iso_date(self.fieldwork_start)
        end = parse_iso_date(self.fieldwork_end)
        object.__setattr__(self, "fieldwork_start", start)
        object.__setattr__(self, "fieldwork_end", end)
        if end < start:
            raise SchemaValidationError(
                f"fieldwork_end ({end}) precedes fieldwork_start ({start})"
            )
        if self.pub_date is not None:
            object.__setattr__(self, "pub_date", parse_iso_date(self.pub_date))
        if (
            self.sample_size is not None
            and _require_int(self.sample_size, "sample_size") <= 0
        ):
            raise SchemaValidationError(
                f"sample_size must be > 0, got {self.sample_size}"
            )
        if self.turnout is not None:
            t = _require_float(self.turnout, "turnout")
            if t < 0.0 or t > 1.0:
                raise SchemaValidationError(f"turnout share must be in [0, 1], got {t}")
        _require_bool(self.is_cis, "is_cis")

        cleaned: dict[str, float] = {}
        for p, v in _require_mapping(self.party_shares, "party_shares").items():
            pid = _require_non_empty_str(p, "party_id")
            sv = _require_float(v, f"party_shares[{pid}]")
            if sv < 0.0 or sv > 1.0:
                raise SchemaValidationError(
                    f"party_shares[{pid}] must be unit share in [0, 1], got {sv}"
                )
            cleaned[pid] = sv
        if not cleaned and not self.allow_empty_shares:
            raise SchemaValidationError(
                "party_shares must contain at least one observed party"
            )
        if sum(cleaned.values()) > 1.05:
            raise SchemaValidationError(
                f"Observed party_shares sum ({sum(cleaned.values()):.4f}) > 1.05"
            )
        object.__setattr__(self, "party_shares", cleaned)

    @property
    def fieldwork_days(self) -> int:
        return (self.fieldwork_end - self.fieldwork_start).days + 1

    @property
    def midpoint_date(self) -> date:
        return self.fieldwork_start + timedelta(
            days=(self.fieldwork_end - self.fieldwork_start).days // 2
        )

    @property
    def effective_pub_date(self) -> date:
        return self.pub_date if self.pub_date is not None else self.fieldwork_end

    def party_percentages(self) -> dict[str, float]:
        return {p: share_to_pct(v) for p, v in self.party_shares.items()}

    @classmethod
    def from_csv_row(
        cls,
        row: Mapping[str, Any],
        party_columns: Sequence[str] | None = None,
        *,
        allow_empty_shares: bool = False,
    ) -> PollRecord:
        r = _require_mapping(row, "PollRecord CSV row")
        raw_n = r.get("sample_size")
        sample_n = None if _is_missing_cell(raw_n) else int(float(str(raw_n).strip()))
        raw_turnout = r.get("turnout")
        turnout_share: float | None = None
        if not _is_missing_cell(raw_turnout):
            tv = float(str(raw_turnout).strip())
            turnout_share = tv / 100.0 if tv > 1.0 else tv
        raw_pub = r.get("pub_date")
        pub = None if _is_missing_cell(raw_pub) else parse_iso_date(str(raw_pub))
        return cls(
            election_id=_require_non_empty_str(r.get("election_id"), "election_id"),
            pollster=_require_non_empty_str(r.get("pollster"), "pollster"),
            media=""
            if _is_missing_cell(r.get("media"))
            else str(r.get("media")).strip(),
            fieldwork_start=parse_iso_date(str(r.get("fieldwork_start", ""))),
            fieldwork_end=parse_iso_date(str(r.get("fieldwork_end", ""))),
            pub_date=pub,
            sample_size=sample_n,
            turnout=turnout_share,
            is_cis=parse_bool_token(r.get("is_cis", False), "is_cis"),
            party_shares=_parse_csv_party_shares(r, party_columns),
            allow_empty_shares=allow_empty_shares,
        )

    def to_csv_row(self, party_columns: Sequence[str]) -> dict[str, str]:
        out: dict[str, str] = {
            "election_id": self.election_id,
            "pollster": self.pollster,
            "media": self.media,
            "fieldwork_start": self.fieldwork_start.isoformat(),
            "fieldwork_end": self.fieldwork_end.isoformat(),
            "pub_date": self.pub_date.isoformat() if self.pub_date is not None else "",
            "sample_size": str(self.sample_size)
            if self.sample_size is not None
            else "",
            "turnout": f"{self.turnout * 100.0:.2f}"
            if self.turnout is not None
            else "",
            "is_cis": "true" if self.is_cis else "false",
        }
        for col in party_columns:
            val = self.party_shares.get(col)
            out[col] = f"{val * 100.0:.2f}" if val is not None else ""
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PollRecord:
        d = _require_mapping(data, "PollRecord")
        pub = d.get("pub_date")
        n = d.get("sample_size")
        t = d.get("turnout")
        return cls(
            election_id=_require_non_empty_str(d.get("election_id"), "election_id"),
            pollster=_require_non_empty_str(d.get("pollster"), "pollster"),
            media=str(d.get("media", "")).strip(),
            fieldwork_start=parse_iso_date(d.get("fieldwork_start", "")),
            fieldwork_end=parse_iso_date(d.get("fieldwork_end", "")),
            pub_date=parse_iso_date(pub) if pub is not None else None,
            sample_size=_require_int(n, "sample_size") if n is not None else None,
            turnout=_require_float(t, "turnout") if t is not None else None,
            is_cis=_require_bool(d.get("is_cis", False), "is_cis"),
            party_shares={
                str(k): _require_float(v, f"party_shares[{k}]")
                for k, v in _require_mapping(
                    d.get("party_shares"), "party_shares"
                ).items()
            },
            allow_empty_shares=_require_bool(
                d.get("allow_empty_shares", False), "allow_empty_shares"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "election_id": self.election_id,
            "pollster": self.pollster,
            "media": self.media,
            "fieldwork_start": self.fieldwork_start.isoformat(),
            "fieldwork_end": self.fieldwork_end.isoformat(),
            "pub_date": self.pub_date.isoformat()
            if self.pub_date is not None
            else None,
            "sample_size": self.sample_size,
            "turnout": self.turnout,
            "is_cis": self.is_cis,
            "party_shares": dict(self.party_shares),
        }


@dataclass(frozen=True, slots=True)
class RegionalPollRecord:
    """Sub-national CCAA or provincial opinion poll with shares on ``[0.0, 1.0]``."""

    scope_type: PollScopeType
    scope_code: str
    pollster: str
    fieldwork_start: date
    fieldwork_end: date
    party_shares: dict[str, float]
    pub_date: date | None = None
    sample_size: int | None = None
    media: str = ""
    is_cis: bool = False

    def __post_init__(self) -> None:
        if self.scope_type == "ccaa":
            object.__setattr__(self, "scope_code", normalize_codauto(self.scope_code))
        elif self.scope_type == "province":
            object.__setattr__(self, "scope_code", normalize_cpro(self.scope_code))
        else:
            raise SchemaValidationError(f"Invalid scope_type: {self.scope_type!r}")

        _require_non_empty_str(self.pollster, "pollster")
        start = parse_iso_date(self.fieldwork_start)
        end = parse_iso_date(self.fieldwork_end)
        object.__setattr__(self, "fieldwork_start", start)
        object.__setattr__(self, "fieldwork_end", end)
        if end < start:
            raise SchemaValidationError(
                f"fieldwork_end ({end}) precedes fieldwork_start ({start})"
            )
        if self.pub_date is not None:
            object.__setattr__(self, "pub_date", parse_iso_date(self.pub_date))
        if (
            self.sample_size is not None
            and _require_int(self.sample_size, "sample_size") <= 0
        ):
            raise SchemaValidationError(
                f"sample_size must be > 0, got {self.sample_size}"
            )
        _require_bool(self.is_cis, "is_cis")

        cleaned: dict[str, float] = {}
        for p, v in _require_mapping(self.party_shares, "party_shares").items():
            pid = _require_non_empty_str(p, "party_id")
            sv = _require_float(v, f"party_shares[{pid}]")
            if sv < 0.0 or sv > 1.0:
                raise SchemaValidationError(f"party_shares[{pid}] out of [0, 1]: {sv}")
            cleaned[pid] = sv
        if not cleaned:
            raise SchemaValidationError(
                "RegionalPollRecord.party_shares must not be empty"
            )
        if sum(cleaned.values()) > 1.05:
            raise SchemaValidationError(
                "Sum of regional poll party_shares exceeds 1.05"
            )
        object.__setattr__(self, "party_shares", cleaned)

    @property
    def fieldwork_days(self) -> int:
        return (self.fieldwork_end - self.fieldwork_start).days + 1

    @property
    def midpoint_date(self) -> date:
        return self.fieldwork_start + timedelta(
            days=(self.fieldwork_end - self.fieldwork_start).days // 2
        )

    @property
    def effective_pub_date(self) -> date:
        return self.pub_date if self.pub_date is not None else self.fieldwork_end

    @classmethod
    def from_csv_row(
        cls, row: Mapping[str, Any], party_columns: Sequence[str] | None = None
    ) -> RegionalPollRecord:
        r = _require_mapping(row, "RegionalPollRecord CSV row")
        st = _require_non_empty_str(r.get("scope_type"), "scope_type")
        if st == "ccaa":
            st_lit: PollScopeType = "ccaa"
        elif st == "province":
            st_lit = "province"
        else:
            raise SchemaValidationError(f"Invalid scope_type: {st!r}")
        raw_n = r.get("sample_size")
        sample_n = None if _is_missing_cell(raw_n) else int(float(str(raw_n).strip()))
        raw_pub = r.get("pub_date")
        pub = None if _is_missing_cell(raw_pub) else parse_iso_date(str(raw_pub))
        return cls(
            scope_type=st_lit,
            scope_code=str(r.get("scope_code", "")),
            pollster=_require_non_empty_str(r.get("pollster"), "pollster"),
            fieldwork_start=parse_iso_date(str(r.get("fieldwork_start", ""))),
            fieldwork_end=parse_iso_date(str(r.get("fieldwork_end", ""))),
            pub_date=pub,
            sample_size=sample_n,
            media=""
            if _is_missing_cell(r.get("media"))
            else str(r.get("media")).strip(),
            is_cis=parse_bool_token(r.get("is_cis", False), "is_cis"),
            party_shares=_parse_csv_party_shares(r, party_columns),
        )

    def to_csv_row(self, party_columns: Sequence[str]) -> dict[str, str]:
        out: dict[str, str] = {
            "scope_type": self.scope_type,
            "scope_code": self.scope_code,
            "pollster": self.pollster,
            "fieldwork_start": self.fieldwork_start.isoformat(),
            "fieldwork_end": self.fieldwork_end.isoformat(),
            "pub_date": self.pub_date.isoformat() if self.pub_date is not None else "",
            "sample_size": str(self.sample_size)
            if self.sample_size is not None
            else "",
        }
        for col in party_columns:
            val = self.party_shares.get(col)
            out[col] = f"{val * 100.0:.2f}" if val is not None else ""
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RegionalPollRecord:
        d = _require_mapping(data, "RegionalPollRecord")
        st = _require_non_empty_str(d.get("scope_type"), "scope_type")
        if st == "ccaa":
            st_lit: PollScopeType = "ccaa"
        elif st == "province":
            st_lit = "province"
        else:
            raise SchemaValidationError(f"Invalid scope_type: {st!r}")
        pub = d.get("pub_date")
        n = d.get("sample_size")
        return cls(
            scope_type=st_lit,
            scope_code=str(d.get("scope_code", "")),
            pollster=_require_non_empty_str(d.get("pollster"), "pollster"),
            fieldwork_start=parse_iso_date(d.get("fieldwork_start", "")),
            fieldwork_end=parse_iso_date(d.get("fieldwork_end", "")),
            pub_date=parse_iso_date(pub) if pub is not None else None,
            sample_size=_require_int(n, "sample_size") if n is not None else None,
            media=str(d.get("media", "")).strip(),
            is_cis=_require_bool(d.get("is_cis", False), "is_cis"),
            party_shares={
                str(k): _require_float(v, f"party_shares[{k}]")
                for k, v in _require_mapping(
                    d.get("party_shares"), "party_shares"
                ).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_type": self.scope_type,
            "scope_code": self.scope_code,
            "pollster": self.pollster,
            "fieldwork_start": self.fieldwork_start.isoformat(),
            "fieldwork_end": self.fieldwork_end.isoformat(),
            "pub_date": self.pub_date.isoformat()
            if self.pub_date is not None
            else None,
            "sample_size": self.sample_size,
            "media": self.media,
            "is_cis": self.is_cis,
            "party_shares": dict(self.party_shares),
        }


def _parse_manual_action(raw: Any) -> ManualPollAction:
    act = _require_non_empty_str(raw, "action").lower()
    if act == "add":
        return "add"
    if act == "override":
        return "override"
    if act == "exclude":
        return "exclude"
    raise SchemaValidationError(f"Invalid manual poll action: {act!r}")


@dataclass(frozen=True, slots=True)
class ManualPollOverlay:
    """User-editable poll overlay row from ``data/polls/manual_polls_2026.csv``."""

    action: ManualPollAction
    poll: PollRecord

    def __post_init__(self) -> None:
        if self.action not in MANUAL_POLL_ACTIONS:
            raise SchemaValidationError(
                f"Invalid manual action {self.action!r}; expected add|override|exclude"
            )
        if self.action in {"add", "override"} and not self.poll.party_shares:
            raise SchemaValidationError(
                f"Manual poll action {self.action!r} requires at least one party share"
            )

    @classmethod
    def from_csv_row(
        cls, row: Mapping[str, Any], party_columns: Sequence[str] | None = None
    ) -> ManualPollOverlay:
        r = _require_mapping(row, "ManualPollOverlay CSV row")
        act_lit = _parse_manual_action(r.get("action"))
        poll = PollRecord.from_csv_row(
            r, party_columns, allow_empty_shares=(act_lit == "exclude")
        )
        return cls(action=act_lit, poll=poll)

    def to_csv_row(self, party_columns: Sequence[str]) -> dict[str, str]:
        row = self.poll.to_csv_row(party_columns)
        row["action"] = self.action
        return row

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManualPollOverlay:
        d = _require_mapping(data, "ManualPollOverlay")
        act_lit = _parse_manual_action(d.get("action"))
        poll_dict = dict(_require_mapping(d.get("poll"), "poll"))
        if act_lit == "exclude":
            poll_dict.setdefault("allow_empty_shares", True)
        return cls(action=act_lit, poll=PollRecord.from_dict(poll_dict))

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "poll": self.poll.to_dict()}


@dataclass(frozen=True, slots=True)
class AllocationInput:
    """Validated single-constituency input to the LOREG Art. 163 seat allocator."""

    seats: int
    party_votes: dict[str, int | float]
    blank_votes: int | float = 0
    is_autonomous_city: bool = False
    cpro: str | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        s = _require_int(self.seats, "seats")
        if s < 1:
            raise SchemaValidationError(f"seats must be >= 1, got {s}")
        _require_bool(self.is_autonomous_city, "is_autonomous_city")

        if self.cpro is not None:
            object.__setattr__(self, "cpro", normalize_cpro(self.cpro))

        bv = _require_number(self.blank_votes, "blank_votes")
        if bv < 0:
            raise SchemaValidationError(f"blank_votes must be >= 0, got {bv}")

        pv_map = _require_mapping(self.party_votes, "party_votes")
        if not pv_map:
            raise SchemaValidationError("party_votes must be a non-empty mapping")
        cleaned_votes: dict[str, int | float] = {}
        for k, v in pv_map.items():
            pid = _require_non_empty_str(k, "party_id")
            nv = _require_number(v, f"party_votes[{pid}]")
            if nv < 0:
                raise SchemaValidationError(
                    f"Negative votes not allowed for {pid}: {nv}"
                )
            cleaned_votes[pid] = nv
        object.__setattr__(self, "party_votes", cleaned_votes)

        party_sum = sum(cleaned_votes.values())
        valid = party_sum + bv
        if valid <= 0:
            raise SchemaValidationError(
                "valid_votes (sum(party_votes) + blank_votes) must be > 0"
            )
        if self.is_autonomous_city and party_sum <= 0:
            raise SchemaValidationError(
                "Autonomous city plurality requires sum(party_votes) > 0"
            )

        eff_thresh = (
            (0.0 if self.is_autonomous_city else 0.03)
            if self.threshold is None
            else _require_float(self.threshold, "threshold")
        )
        if eff_thresh < 0.0 or eff_thresh >= 1.0:
            raise SchemaValidationError(
                f"threshold must be in [0.0, 1.0), got {eff_thresh}"
            )
        object.__setattr__(self, "threshold", eff_thresh)

    @property
    def is_integer_mode(self) -> bool:
        """True when all party votes and blank votes are exact Python integers."""
        return (
            isinstance(self.blank_votes, int)
            and not isinstance(self.blank_votes, bool)
            and all(
                isinstance(v, int) and not isinstance(v, bool)
                for v in self.party_votes.values()
            )
            and (
                self.is_autonomous_city
                or abs((self.threshold or 0.0) - 0.03) < 1e-12
                or abs(self.threshold or 0.0) < 1e-12
            )
        )

    @property
    def valid_votes(self) -> int | float:
        if self.is_integer_mode:
            return int(self.blank_votes) + sum(
                int(v) for v in self.party_votes.values()
            )
        return float(self.blank_votes) + sum(
            float(v) for v in self.party_votes.values()
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AllocationInput:
        d = _require_mapping(data, "AllocationInput")
        if "seats" not in d:
            raise SchemaValidationError(
                "AllocationInput missing required field 'seats'"
            )
        if "party_votes" not in d:
            raise SchemaValidationError(
                "AllocationInput missing required field 'party_votes'"
            )
        raw_cpro = d.get("cpro")
        cpro = normalize_cpro(raw_cpro) if raw_cpro is not None else None
        if "is_autonomous_city" in d and d["is_autonomous_city"] is not None:
            is_city = _require_bool(d["is_autonomous_city"], "is_autonomous_city")
        elif cpro is not None:
            is_city = cpro in AUTONOMOUS_CITY_CPROS
        else:
            is_city = False
        raw_pv = _require_mapping(d.get("party_votes"), "party_votes")
        pv = {
            str(k): _require_number(v, f"party_votes[{k}]") for k, v in raw_pv.items()
        }
        default_blank: int | float = (
            0
            if all(isinstance(v, int) and not isinstance(v, bool) for v in pv.values())
            else 0.0
        )
        raw_blank = d.get("blank_votes", default_blank)
        raw_thresh = d.get("threshold")
        return cls(
            seats=_require_int(d.get("seats"), "seats"),
            party_votes=pv,
            blank_votes=_require_number(raw_blank, "blank_votes"),
            is_autonomous_city=is_city,
            cpro=cpro,
            threshold=(
                _require_float(raw_thresh, "threshold")
                if raw_thresh is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seats": self.seats,
            "is_autonomous_city": self.is_autonomous_city,
            "blank_votes": self.blank_votes,
            "party_votes": dict(self.party_votes),
        }
        if self.cpro is not None:
            out["cpro"] = self.cpro
        if self.threshold is not None:
            out["threshold"] = self.threshold
        return out


@dataclass(frozen=True, slots=True)
class QuotientEntry:
    """Single D'Hondt quotient table row for provincial seat inspection."""

    party: str
    divisor: int
    quotient: float
    won: bool
    global_rank: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.party, "party")
        if _require_int(self.divisor, "divisor") < 1:
            raise SchemaValidationError("divisor must be >= 1")
        if _require_float(self.quotient, "quotient") < 0.0:
            raise SchemaValidationError("quotient must be >= 0.0")
        _require_bool(self.won, "won")
        if self.global_rank is not None and _require_int(self.global_rank, "global_rank") < 1:
            raise SchemaValidationError("global_rank must be >= 1")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuotientEntry:
        d = _require_mapping(data, "QuotientEntry")
        gr = d.get("global_rank")
        return cls(
            party=_require_non_empty_str(d.get("party"), "party"),
            divisor=_require_int(d.get("divisor"), "divisor"),
            quotient=_require_float(d.get("quotient"), "quotient"),
            won=_require_bool(d.get("won"), "won"),
            global_rank=_require_int(gr, "global_rank") if gr is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "party": self.party,
            "divisor": self.divisor,
            "quotient": self.quotient,
            "won": self.won,
        }
        if self.global_rank is not None:
            out["global_rank"] = self.global_rank
        return out


@dataclass(frozen=True, slots=True)
class RunnerUpGap:
    """Vote-share and vote-count gap for a contender chasing the final seat.

    The gap measures the additional votes (with fixed ``valid_votes`` in the
    constituency) needed by the challenger to beat the ``M_c``-th cutoff quotient
    or cross the LOREG 3% ``valid_votes`` threshold if currently below threshold:
    ``vote_gap_pp = 100.0 * vote_gap_votes / valid_votes``.
    """

    party: str
    vote_gap_pp: float
    vote_gap_abs: int

    def __post_init__(self) -> None:
        _require_non_empty_str(self.party, "party")
        if _require_float(self.vote_gap_pp, "vote_gap_pp") < 0.0:
            raise SchemaValidationError("vote_gap_pp must be >= 0.0")
        if _require_int(self.vote_gap_abs, "vote_gap_abs") < 0:
            raise SchemaValidationError("vote_gap_abs must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunnerUpGap:
        d = _require_mapping(data, "RunnerUpGap")
        return cls(
            party=_require_non_empty_str(d.get("party"), "party"),
            vote_gap_pp=_require_float(d.get("vote_gap_pp"), "vote_gap_pp"),
            vote_gap_abs=_require_int(d.get("vote_gap_abs"), "vote_gap_abs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "party": self.party,
            "vote_gap_pp": self.vote_gap_pp,
            "vote_gap_abs": self.vote_gap_abs,
        }


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Output of deterministic LOREG Art. 163 allocation for one constituency."""

    valid_votes: int | float
    eligible_parties: tuple[str, ...]
    seats: dict[str, int]
    cutoff_quotient: float | None
    ties_broken: int = 0
    cpro: str | None = None
    quotients: dict[str, list[float]] | None = None
    quotient_table: tuple[QuotientEntry, ...] | None = None
    last_seat_battle: LastSeatBattle | None = None

    def __post_init__(self) -> None:
        if _require_number(self.valid_votes, "valid_votes") <= 0:
            raise SchemaValidationError("valid_votes must be > 0")
        if _require_int(self.ties_broken, "ties_broken") < 0:
            raise SchemaValidationError("ties_broken must be >= 0")
        if (
            self.cutoff_quotient is not None
            and _require_float(self.cutoff_quotient, "cutoff_quotient") < 0.0
        ):
            raise SchemaValidationError("cutoff_quotient must be >= 0.0")
        if self.cpro is not None:
            object.__setattr__(self, "cpro", normalize_cpro(self.cpro))
        cleaned_seats: dict[str, int] = {}
        for k, v in _require_mapping(self.seats, "seats").items():
            pid = _require_non_empty_str(k, "party_id")
            sv = _require_int(v, f"seats[{pid}]")
            if sv < 0:
                raise SchemaValidationError(f"seats[{pid}] must be >= 0, got {sv}")
            cleaned_seats[pid] = sv
        object.__setattr__(self, "seats", cleaned_seats)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AllocationResult:
        d = _require_mapping(data, "AllocationResult")
        raw_elig = _require_sequence_not_str(
            d.get("eligible_parties", []), "eligible_parties"
        )
        cq = d.get("cutoff_quotient")
        raw_cpro = d.get("cpro")
        raw_q = d.get("quotients")
        q_map: dict[str, list[float]] | None = None
        if raw_q is not None:
            q_map = {
                str(k): [
                    _require_float(x, f"quotients[{k}]")
                    for x in _require_sequence_not_str(v, f"quotients[{k}]")
                ]
                for k, v in _require_mapping(raw_q, "quotients").items()
            }
        raw_qt = d.get("quotient_table")
        q_table = (
            tuple(
                QuotientEntry.from_dict(x)
                for x in _require_sequence_not_str(raw_qt, "quotient_table")
            )
            if raw_qt is not None
            else None
        )
        raw_lsb = d.get("last_seat_battle")
        lsb = LastSeatBattle.from_dict(raw_lsb) if raw_lsb is not None else None
        return cls(
            valid_votes=_require_number(d.get("valid_votes"), "valid_votes"),
            eligible_parties=tuple(
                _require_non_empty_str(p, "party") for p in raw_elig
            ),
            seats={
                str(k): _require_int(v, f"seats[{k}]")
                for k, v in _require_mapping(d.get("seats"), "seats").items()
            },
            cutoff_quotient=_require_float(cq, "cutoff_quotient")
            if cq is not None
            else None,
            ties_broken=_require_int(d.get("ties_broken", 0), "ties_broken"),
            cpro=normalize_cpro(raw_cpro) if raw_cpro is not None else None,
            quotients=q_map,
            quotient_table=q_table,
            last_seat_battle=lsb,
        )

    def to_dict(self, *, include_quotients: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "valid_votes": self.valid_votes,
            "eligible_parties": list(self.eligible_parties),
            "seats": dict(self.seats),
            "cutoff_quotient": self.cutoff_quotient,
            "ties_broken": self.ties_broken,
        }
        if self.cpro is not None:
            out["cpro"] = self.cpro
        if include_quotients and self.quotients is not None:
            out["quotients"] = {k: list(v) for k, v in self.quotients.items()}
        if include_quotients and self.quotient_table is not None:
            out["quotient_table"] = [q.to_dict() for q in self.quotient_table]
        if self.last_seat_battle is not None:
            out["last_seat_battle"] = self.last_seat_battle.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class BatchAllocationResult:
    """Batch multi-constituency output for ``elecciones allocate``."""

    constituencies: tuple[AllocationResult, ...]
    total_seats: int
    national_seats: dict[str, int]
    total_ties_broken: int = 0

    @classmethod
    def from_results(cls, results: Sequence[AllocationResult]) -> BatchAllocationResult:
        if not results:
            raise SchemaValidationError(
                "BatchAllocationResult requires at least 1 constituency"
            )
        nat: dict[str, int] = {}
        for r in results:
            for p, s in r.seats.items():
                nat[p] = nat.get(p, 0) + s
        return cls(
            constituencies=tuple(results),
            total_seats=sum(nat.values()),
            national_seats=nat,
            total_ties_broken=sum(r.ties_broken for r in results),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BatchAllocationResult:
        d = _require_mapping(data, "BatchAllocationResult")
        raw_items = _require_sequence_not_str(d.get("constituencies"), "constituencies")
        return cls.from_results([AllocationResult.from_dict(x) for x in raw_items])

    def to_dict(self) -> dict[str, Any]:
        return {
            "constituencies": [r.to_dict() for r in self.constituencies],
            "total_seats": self.total_seats,
            "national_seats": dict(self.national_seats),
            "total_ties_broken": self.total_ties_broken,
        }


def parse_allocation_payload(
    data: Mapping[str, Any],
) -> AllocationInput | list[AllocationInput]:
    """Parse a single-district or batch ``{"constituencies": ...}`` JSON payload."""
    d = _require_mapping(data, "Allocation CLI payload")
    if "constituencies" in d:
        raw_c = d["constituencies"]
        if isinstance(raw_c, Mapping):
            if not raw_c:
                raise SchemaValidationError("constituencies mapping must not be empty")
            out_map: list[AllocationInput] = []
            for k, v in raw_c.items():
                item = dict(_require_mapping(v, f"constituencies[{k}]"))
                item.setdefault("cpro", k)
                out_map.append(AllocationInput.from_dict(item))
            return out_map
        seq = _require_sequence_not_str(raw_c, "constituencies")
        if not seq:
            raise SchemaValidationError("constituencies list must not be empty")
        return [AllocationInput.from_dict(x) for x in seq]
    return AllocationInput.from_dict(d)


@dataclass(frozen=True, slots=True)
class ConstituencyElectionResult:
    """Official JEC/BOE scrutiny row (``data/elections/congress_*.csv``)."""

    election_id: str
    cpro: str
    constituency: str
    seats: int
    censo: int
    votantes: int
    nulos: int
    validos: int
    blancos: int
    party_votes: dict[str, int]
    party_seats: dict[str, int]

    def __post_init__(self) -> None:
        _require_non_empty_str(self.election_id, "election_id")
        object.__setattr__(self, "cpro", normalize_cpro(self.cpro))
        _require_non_empty_str(self.constituency, "constituency")
        s = _require_int(self.seats, "seats")
        if s < 1:
            raise SchemaValidationError(f"seats must be >= 1, got {s}")
        for attr in ("censo", "votantes", "nulos", "blancos"):
            if _require_int(getattr(self, attr), attr) < 0:
                raise SchemaValidationError(f"{attr} must be >= 0")
        if _require_int(self.validos, "validos") <= 0:
            raise SchemaValidationError("validos must be > 0")

        pv = {
            _require_non_empty_str(k, "party"): _require_int(v, f"votes_{k}")
            for k, v in _require_mapping(self.party_votes, "party_votes").items()
        }
        ps = {
            _require_non_empty_str(k, "party"): _require_int(v, f"seats_{k}")
            for k, v in _require_mapping(self.party_seats, "party_seats").items()
        }
        if any(v < 0 for v in pv.values()) or any(v < 0 for v in ps.values()):
            raise SchemaValidationError(
                "Negative votes or seats in ConstituencyElectionResult"
            )
        object.__setattr__(self, "party_votes", pv)
        object.__setattr__(self, "party_seats", ps)

        expected_valid = self.blancos + sum(pv.values())
        if self.validos != expected_valid:
            raise SchemaValidationError(
                f"{self.election_id} cpro={self.cpro}: validos ({self.validos}) != "
                f"blancos ({self.blancos}) + sum(party_votes) ({sum(pv.values())})"
            )
        if self.votantes > 0 and self.votantes != self.validos + self.nulos:
            raise SchemaValidationError(
                f"{self.election_id} cpro={self.cpro}: votantes ({self.votantes}) != "
                f"validos ({self.validos}) + nulos ({self.nulos})"
            )
        if sum(ps.values()) != s:
            raise SchemaValidationError(
                f"{self.election_id} cpro={self.cpro}: party_seats sum != seats ({s})"
            )

    @property
    def is_autonomous_city(self) -> bool:
        return self.cpro in AUTONOMOUS_CITY_CPROS

    def to_allocation_input(self) -> AllocationInput:
        return AllocationInput(
            seats=self.seats,
            party_votes=dict(self.party_votes),
            blank_votes=self.blancos,
            is_autonomous_city=self.is_autonomous_city,
            cpro=self.cpro,
        )

    @classmethod
    def from_csv_row(cls, row: Mapping[str, Any]) -> ConstituencyElectionResult:
        r = _require_mapping(row, "ConstituencyElectionResult CSV row")
        pv: dict[str, int] = {}
        ps: dict[str, int] = {}
        for col, raw_val in r.items():
            if col.startswith("votes_"):
                party = col[len("votes_") :]
                pv[party] = (
                    0 if _is_missing_cell(raw_val) else int(str(raw_val).strip())
                )
            elif col.startswith("seats_"):
                party = col[len("seats_") :]
                ps[party] = (
                    0 if _is_missing_cell(raw_val) else int(str(raw_val).strip())
                )
        return cls(
            election_id=_require_non_empty_str(r.get("election_id"), "election_id"),
            cpro=normalize_cpro(str(r.get("cpro", ""))),
            constituency=_require_non_empty_str(r.get("constituency"), "constituency"),
            seats=int(str(r.get("seats", "0")).strip()),
            censo=int(str(r.get("censo", "0")).strip()),
            votantes=int(str(r.get("votantes", "0")).strip()),
            nulos=int(str(r.get("nulos", "0")).strip()),
            validos=int(str(r.get("validos", "0")).strip()),
            blancos=int(str(r.get("blancos", "0")).strip()),
            party_votes=pv,
            party_seats=ps,
        )

    def to_csv_row(
        self, party_order: Sequence[str] | None = None
    ) -> dict[str, str | int]:
        parties = (
            list(party_order)
            if party_order is not None
            else sorted(self.party_votes.keys())
        )
        out: dict[str, str | int] = {
            "election_id": self.election_id,
            "cpro": self.cpro,
            "constituency": self.constituency,
            "seats": self.seats,
            "censo": self.censo,
            "votantes": self.votantes,
            "nulos": self.nulos,
            "validos": self.validos,
            "blancos": self.blancos,
        }
        for p in parties:
            out[f"votes_{p}"] = self.party_votes.get(p, 0)
        for p in parties:
            out[f"seats_{p}"] = self.party_seats.get(p, 0)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstituencyElectionResult:
        d = _require_mapping(data, "ConstituencyElectionResult")
        return cls(
            election_id=_require_non_empty_str(d.get("election_id"), "election_id"),
            cpro=normalize_cpro(d.get("cpro", "")),
            constituency=_require_non_empty_str(d.get("constituency"), "constituency"),
            seats=_require_int(d.get("seats"), "seats"),
            censo=_require_int(d.get("censo"), "censo"),
            votantes=_require_int(d.get("votantes"), "votantes"),
            nulos=_require_int(d.get("nulos"), "nulos"),
            validos=_require_int(d.get("validos"), "validos"),
            blancos=_require_int(d.get("blancos"), "blancos"),
            party_votes={
                str(k): _require_int(v, f"party_votes[{k}]")
                for k, v in _require_mapping(
                    d.get("party_votes"), "party_votes"
                ).items()
            },
            party_seats={
                str(k): _require_int(v, f"party_seats[{k}]")
                for k, v in _require_mapping(
                    d.get("party_seats"), "party_seats"
                ).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "election_id": self.election_id,
            "cpro": self.cpro,
            "constituency": self.constituency,
            "seats": self.seats,
            "censo": self.censo,
            "votantes": self.votantes,
            "nulos": self.nulos,
            "validos": self.validos,
            "blancos": self.blancos,
            "party_votes": dict(self.party_votes),
            "party_seats": dict(self.party_seats),
        }


def _parse_forecast_engine(raw: Any) -> ForecastEngine:
    eng = _require_non_empty_str(raw, "engine")
    if eng == "ensemble":
        return "ensemble"
    if eng == "kernel":
        return "kernel"
    if eng == "state_space":
        return "state_space"
    raise SchemaValidationError(f"Invalid engine: {eng!r}")


def _parse_seat_tier(raw: Any) -> SeatTier:
    t = _require_non_empty_str(raw, "tier")
    if t == "seguro":
        return "seguro"
    if t == "probable":
        return "probable"
    if t == "en_juego":
        return "en_juego"
    if t == "remoto":
        return "remoto"
    raise SchemaValidationError(f"Invalid SeatSlot.tier: {t!r}")


@dataclass(frozen=True, slots=True)
class ForecastMetadata:
    """Top-level ``metadata`` block inside ``data/outputs/forecast_29n.json``."""

    generated_at: str
    as_of_date: str
    election_date: str
    n_sims: int
    seed: int
    engine: ForecastEngine
    exclude_cis_headline: bool
    blackout_active: bool
    simulation_runtime_ms: float

    def __post_init__(self) -> None:
        _require_non_empty_str(self.generated_at, "generated_at")
        parse_iso_date(self.as_of_date)
        parse_iso_date(self.election_date)
        if _require_int(self.n_sims, "n_sims") < 1:
            raise SchemaValidationError("n_sims must be >= 1")
        _require_int(self.seed, "seed")
        if self.engine not in FORECAST_ENGINES:
            raise SchemaValidationError(f"Invalid engine: {self.engine!r}")
        _require_bool(self.exclude_cis_headline, "exclude_cis_headline")
        _require_bool(self.blackout_active, "blackout_active")
        if _require_float(self.simulation_runtime_ms, "simulation_runtime_ms") < 0.0:
            raise SchemaValidationError("simulation_runtime_ms must be >= 0.0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForecastMetadata:
        d = _require_mapping(data, "ForecastMetadata")
        return cls(
            generated_at=_require_non_empty_str(d.get("generated_at"), "generated_at"),
            as_of_date=_require_non_empty_str(d.get("as_of_date"), "as_of_date"),
            election_date=_require_non_empty_str(
                d.get("election_date"), "election_date"
            ),
            n_sims=_require_int(d.get("n_sims"), "n_sims"),
            seed=_require_int(d.get("seed"), "seed"),
            engine=_parse_forecast_engine(d.get("engine")),
            exclude_cis_headline=_require_bool(
                d.get("exclude_cis_headline"), "exclude_cis_headline"
            ),
            blackout_active=_require_bool(d.get("blackout_active"), "blackout_active"),
            simulation_runtime_ms=_require_float(
                d.get("simulation_runtime_ms"), "simulation_runtime_ms"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "as_of_date": self.as_of_date,
            "election_date": self.election_date,
            "n_sims": self.n_sims,
            "seed": self.seed,
            "engine": self.engine,
            "exclude_cis_headline": self.exclude_cis_headline,
            "blackout_active": self.blackout_active,
            "simulation_runtime_ms": self.simulation_runtime_ms,
        }


@dataclass(frozen=True, slots=True)
class VoteShareInterval:
    """National vote-share summary interval ``[0.0, 1.0]``."""

    mean: float
    median: float
    p5: float
    p10: float
    p90: float
    p95: float

    def __post_init__(self) -> None:
        for attr in ("mean", "median", "p5", "p10", "p90", "p95"):
            val = _require_float(getattr(self, attr), attr)
            if val < 0.0 or val > 1.0:
                raise SchemaValidationError(
                    f"VoteShareInterval.{attr} out of [0, 1]: {val}"
                )
        if not (self.p5 <= self.p10 + 1e-12 and self.p10 <= self.median + 1e-12):
            raise SchemaValidationError(
                "VoteShareInterval quantile order violated (p5 <= p10 <= median)"
            )
        if not (self.median <= self.p90 + 1e-12 and self.p90 <= self.p95 + 1e-12):
            raise SchemaValidationError(
                "VoteShareInterval quantile order violated (median <= p90 <= p95)"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VoteShareInterval:
        d = _require_mapping(data, "VoteShareInterval")
        return cls(
            mean=_require_float(d.get("mean"), "mean"),
            median=_require_float(d.get("median"), "median"),
            p5=_require_float(d.get("p5"), "p5"),
            p10=_require_float(d.get("p10"), "p10"),
            p90=_require_float(d.get("p90"), "p90"),
            p95=_require_float(d.get("p95"), "p95"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "p5": self.p5,
            "p10": self.p10,
            "p90": self.p90,
            "p95": self.p95,
        }


@dataclass(frozen=True, slots=True)
class PartyNationalSeats:
    """National seat distribution and El País tier counts for a party."""

    mean: float
    median: int
    p5: int
    p10: int
    p90: int
    p95: int
    seguros: int
    probables: int
    en_juego: int

    def __post_init__(self) -> None:
        if _require_float(self.mean, "mean") < 0.0:
            raise SchemaValidationError("mean seats must be >= 0.0")
        for attr in (
            "median",
            "p5",
            "p10",
            "p90",
            "p95",
            "seguros",
            "probables",
            "en_juego",
        ):
            if _require_int(getattr(self, attr), attr) < 0:
                raise SchemaValidationError(f"{attr} must be >= 0")
        if not (self.p5 <= self.p10 <= self.median <= self.p90 <= self.p95):
            raise SchemaValidationError("PartyNationalSeats quantile order violated")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PartyNationalSeats:
        d = _require_mapping(data, "PartyNationalSeats")
        return cls(
            mean=_require_float(d.get("mean"), "mean"),
            median=_require_int(d.get("median"), "median"),
            p5=_require_int(d.get("p5"), "p5"),
            p10=_require_int(d.get("p10"), "p10"),
            p90=_require_int(d.get("p90"), "p90"),
            p95=_require_int(d.get("p95"), "p95"),
            seguros=_require_int(d.get("seguros"), "seguros"),
            probables=_require_int(d.get("probables"), "probables"),
            en_juego=_require_int(d.get("en_juego"), "en_juego"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "p5": self.p5,
            "p10": self.p10,
            "p90": self.p90,
            "p95": self.p95,
            "seguros": self.seguros,
            "probables": self.probables,
            "en_juego": self.en_juego,
        }


@dataclass(frozen=True, slots=True)
class SeatSlot:
    """Marginal exceedance slot ``(party, k-th seat)`` classified into El País tiers."""

    party: str
    rank: int
    prob: float
    tier: SeatTier

    def __post_init__(self) -> None:
        _require_non_empty_str(self.party, "party")
        if _require_int(self.rank, "rank") < 1:
            raise SchemaValidationError(f"SeatSlot.rank must be >= 1, got {self.rank}")
        p = _require_float(self.prob, "prob")
        if p < 0.0 or p > 1.0:
            raise SchemaValidationError(f"SeatSlot.prob out of [0, 1]: {p}")
        if self.tier not in SEAT_TIERS:
            raise SchemaValidationError(f"Invalid SeatSlot.tier: {self.tier!r}")
        expected_tier = classify_seat_tier(p)
        if expected_tier is not None and expected_tier != self.tier:
            raise SchemaValidationError(
                f"SeatSlot.tier {self.tier!r} != {expected_tier!r} for prob={p}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeatSlot:
        d = _require_mapping(data, "SeatSlot")
        return cls(
            party=_require_non_empty_str(d.get("party"), "party"),
            rank=_require_int(d.get("rank"), "rank"),
            prob=_require_float(d.get("prob"), "prob"),
            tier=_parse_seat_tier(d.get("tier")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "party": self.party,
            "rank": self.rank,
            "prob": self.prob,
            "tier": self.tier,
        }


@dataclass(frozen=True, slots=True)
class LastSeatBattle:
    """Summary of the marginal M_c-th D'Hondt quotient battle in a constituency."""

    holder: str
    cutoff_quotient: float
    runners_up: tuple[RunnerUpGap, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_str(self.holder, "holder")
        if _require_float(self.cutoff_quotient, "cutoff_quotient") < 0.0:
            raise SchemaValidationError("cutoff_quotient must be >= 0.0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LastSeatBattle:
        d = _require_mapping(data, "LastSeatBattle")
        raw_ru = _require_sequence_not_str(d.get("runners_up", []), "runners_up")
        return cls(
            holder=_require_non_empty_str(d.get("holder"), "holder"),
            cutoff_quotient=_require_float(d.get("cutoff_quotient"), "cutoff_quotient"),
            runners_up=tuple(RunnerUpGap.from_dict(x) for x in raw_ru),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "holder": self.holder,
            "cutoff_quotient": self.cutoff_quotient,
            "runners_up": [r.to_dict() for r in self.runners_up],
        }


@dataclass(frozen=True, slots=True)
class PartyConstituencyProjection:
    """Per-party provincial vote-share and seat distribution summary."""

    vote_share_mean: float
    vote_share_p10: float
    vote_share_p90: float
    seats_mean: float
    seats_median: int
    seats_p5: int
    seats_p10: int
    seats_p90: int
    seats_p95: int
    seguros: int
    probables: int
    en_juego: int
    prob_at_least_one: float

    def __post_init__(self) -> None:
        for attr in (
            "vote_share_mean",
            "vote_share_p10",
            "vote_share_p90",
            "prob_at_least_one",
        ):
            v = _require_float(getattr(self, attr), attr)
            if v < 0.0 or v > 1.0:
                raise SchemaValidationError(f"{attr} out of [0, 1]: {v}")
        if _require_float(self.seats_mean, "seats_mean") < 0.0:
            raise SchemaValidationError("seats_mean must be >= 0.0")
        for attr in (
            "seats_median",
            "seats_p5",
            "seats_p10",
            "seats_p90",
            "seats_p95",
            "seguros",
            "probables",
            "en_juego",
        ):
            if _require_int(getattr(self, attr), attr) < 0:
                raise SchemaValidationError(f"{attr} must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PartyConstituencyProjection:
        d = _require_mapping(data, "PartyConstituencyProjection")
        return cls(
            vote_share_mean=_require_float(d.get("vote_share_mean"), "vote_share_mean"),
            vote_share_p10=_require_float(d.get("vote_share_p10"), "vote_share_p10"),
            vote_share_p90=_require_float(d.get("vote_share_p90"), "vote_share_p90"),
            seats_mean=_require_float(d.get("seats_mean"), "seats_mean"),
            seats_median=_require_int(d.get("seats_median"), "seats_median"),
            seats_p5=_require_int(d.get("seats_p5"), "seats_p5"),
            seats_p10=_require_int(d.get("seats_p10"), "seats_p10"),
            seats_p90=_require_int(d.get("seats_p90"), "seats_p90"),
            seats_p95=_require_int(d.get("seats_p95"), "seats_p95"),
            seguros=_require_int(d.get("seguros"), "seguros"),
            probables=_require_int(d.get("probables"), "probables"),
            en_juego=_require_int(d.get("en_juego"), "en_juego"),
            prob_at_least_one=_require_float(
                d.get("prob_at_least_one"), "prob_at_least_one"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vote_share_mean": self.vote_share_mean,
            "vote_share_p10": self.vote_share_p10,
            "vote_share_p90": self.vote_share_p90,
            "seats_mean": self.seats_mean,
            "seats_median": self.seats_median,
            "seats_p5": self.seats_p5,
            "seats_p10": self.seats_p10,
            "seats_p90": self.seats_p90,
            "seats_p95": self.seats_p95,
            "seguros": self.seguros,
            "probables": self.probables,
            "en_juego": self.en_juego,
            "prob_at_least_one": self.prob_at_least_one,
        }


@dataclass(frozen=True, slots=True)
class ConstituencyForecast:
    """Complete Stage 2 + Stage 3 forecast object for one constituency."""

    cpro: str
    name: str
    codauto: str
    seats_2026: int
    turnout_mean: float
    blank_share: float
    pre_shrinkage_ipf_shares: dict[str, float]
    final_shares: dict[str, float]
    party_projections: dict[str, PartyConstituencyProjection]
    seat_slots: tuple[SeatSlot, ...]
    dhondt_quotients: dict[str, list[float]]
    last_seat_battle: LastSeatBattle
    pre_shrinkage_blank_share: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpro", normalize_cpro(self.cpro))
        object.__setattr__(self, "codauto", normalize_codauto(self.codauto))
        _require_non_empty_str(self.name, "name")
        if _require_int(self.seats_2026, "seats_2026") < 1:
            raise SchemaValidationError("seats_2026 must be >= 1")
        tm = _require_float(self.turnout_mean, "turnout_mean")
        if tm <= 0.0 or tm > 1.0:
            raise SchemaValidationError(f"turnout_mean must be in (0, 1], got {tm}")
        bs = _require_float(self.blank_share, "blank_share")
        if bs < 0.0 or bs >= 1.0:
            raise SchemaValidationError(f"blank_share must be in [0, 1), got {bs}")

        pre_bs = (
            bs
            if self.pre_shrinkage_blank_share is None
            else _require_float(
                self.pre_shrinkage_blank_share, "pre_shrinkage_blank_share"
            )
        )
        for label, shares, b_val in (
            ("pre_shrinkage_ipf_shares", self.pre_shrinkage_ipf_shares, pre_bs),
            ("final_shares", self.final_shares, bs),
        ):
            m = _require_mapping(shares, label)
            if not m:
                raise SchemaValidationError(f"{label} must not be empty")
            for k, v in m.items():
                fv = _require_float(v, f"{label}[{k}]")
                if fv < 0.0 or fv > 1.0:
                    raise SchemaValidationError(f"{label}[{k}] out of [0, 1]: {fv}")
            tot = district_simplex_sum(m, b_val)
            if abs(tot - 1.0) > SIMPLEX_TOL:
                raise SchemaValidationError(
                    f"cpro={self.cpro} {label} sum {tot:.12f} != 1.0 +/- {SIMPLEX_TOL}"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstituencyForecast:
        d = _require_mapping(data, "ConstituencyForecast")
        raw_slots = _require_sequence_not_str(d.get("seat_slots", []), "seat_slots")
        raw_pp = _require_mapping(d.get("party_projections", {}), "party_projections")
        raw_dq = _require_mapping(d.get("dhondt_quotients", {}), "dhondt_quotients")
        pre_bs = d.get("pre_shrinkage_blank_share")
        return cls(
            cpro=normalize_cpro(d.get("cpro", "")),
            name=_require_non_empty_str(d.get("name"), "name"),
            codauto=normalize_codauto(d.get("codauto", "")),
            seats_2026=_require_int(d.get("seats_2026"), "seats_2026"),
            turnout_mean=_require_float(d.get("turnout_mean"), "turnout_mean"),
            blank_share=_require_float(d.get("blank_share"), "blank_share"),
            pre_shrinkage_ipf_shares={
                str(k): _require_float(v, f"pre_shrinkage_ipf_shares[{k}]")
                for k, v in _require_mapping(
                    d.get("pre_shrinkage_ipf_shares"), "pre_shrinkage_ipf_shares"
                ).items()
            },
            final_shares={
                str(k): _require_float(v, f"final_shares[{k}]")
                for k, v in _require_mapping(
                    d.get("final_shares"), "final_shares"
                ).items()
            },
            party_projections={
                str(k): PartyConstituencyProjection.from_dict(v)
                for k, v in raw_pp.items()
            },
            seat_slots=tuple(SeatSlot.from_dict(s) for s in raw_slots),
            dhondt_quotients={
                str(k): [
                    _require_float(x, f"dhondt_quotients[{k}]")
                    for x in _require_sequence_not_str(v, f"dhondt_quotients[{k}]")
                ]
                for k, v in raw_dq.items()
            },
            last_seat_battle=LastSeatBattle.from_dict(
                _require_mapping(d.get("last_seat_battle"), "last_seat_battle")
            ),
            pre_shrinkage_blank_share=(
                _require_float(pre_bs, "pre_shrinkage_blank_share")
                if pre_bs is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cpro": self.cpro,
            "name": self.name,
            "codauto": self.codauto,
            "seats_2026": self.seats_2026,
            "turnout_mean": self.turnout_mean,
            "blank_share": self.blank_share,
            "pre_shrinkage_ipf_shares": dict(self.pre_shrinkage_ipf_shares),
            "final_shares": dict(self.final_shares),
            "party_projections": {
                k: v.to_dict() for k, v in self.party_projections.items()
            },
            "seat_slots": [s.to_dict() for s in self.seat_slots],
            "dhondt_quotients": {k: list(v) for k, v in self.dhondt_quotients.items()},
            "last_seat_battle": self.last_seat_battle.to_dict(),
        }
        if self.pre_shrinkage_blank_share is not None:
            out["pre_shrinkage_blank_share"] = self.pre_shrinkage_blank_share
        return out


@dataclass(frozen=True, slots=True)
class PactProbability:
    """Art. 99.3 CE 1st-ballot absolute majority (>= 176 seats) pact summary."""

    label: str
    parties: tuple[str, ...]
    prob: float
    median_seats: int
    p10: int
    p90: int

    def __post_init__(self) -> None:
        _require_non_empty_str(self.label, "label")
        if not self.parties or len(self.parties) != len(set(self.parties)):
            raise SchemaValidationError(
                "PactProbability.parties must be non-empty and unique"
            )
        p = _require_float(self.prob, "prob")
        if p < 0.0 or p > 1.0:
            raise SchemaValidationError(f"PactProbability.prob out of [0, 1]: {p}")
        for attr in ("median_seats", "p10", "p90"):
            if _require_int(getattr(self, attr), attr) < 0:
                raise SchemaValidationError(f"{attr} must be >= 0")
        if not (self.p10 <= self.median_seats <= self.p90):
            raise SchemaValidationError(
                "PactProbability quantile order violated (p10 <= median <= p90)"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PactProbability:
        d = _require_mapping(data, "PactProbability")
        raw_p = _require_sequence_not_str(d.get("parties"), "parties")
        return cls(
            label=_require_non_empty_str(d.get("label"), "label"),
            parties=tuple(_require_non_empty_str(x, "party") for x in raw_p),
            prob=_require_float(d.get("prob"), "prob"),
            median_seats=_require_int(d.get("median_seats"), "median_seats"),
            p10=_require_int(d.get("p10"), "p10"),
            p90=_require_int(d.get("p90"), "p90"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "parties": list(self.parties),
            "prob": self.prob,
            "median_seats": self.median_seats,
            "p10": self.p10,
            "p90": self.p90,
        }


@dataclass(frozen=True, slots=True)
class SecondBallotScenario:
    """Art. 99.3 CE 2nd-ballot simple majority (strict Sí > No) scenario summary."""

    label: str
    yes: tuple[str, ...]
    no: tuple[str, ...]
    abstain: tuple[str, ...]
    prob: float
    median_margin: int

    def __post_init__(self) -> None:
        _require_non_empty_str(self.label, "label")
        if not self.yes:
            raise SchemaValidationError("SecondBallotScenario.yes must not be empty")
        s_yes, s_no, s_abs = set(self.yes), set(self.no), set(self.abstain)
        if (
            len(s_yes) != len(self.yes)
            or len(s_no) != len(self.no)
            or len(s_abs) != len(self.abstain)
        ):
            raise SchemaValidationError("Duplicate party inside yes/no/abstain list")
        if (s_yes & s_no) or (s_yes & s_abs) or (s_no & s_abs):
            raise SchemaValidationError(
                "SecondBallotScenario requires disjoint yes, no, and abstain sets"
            )
        p = _require_float(self.prob, "prob")
        if p < 0.0 or p > 1.0:
            raise SchemaValidationError(f"SecondBallotScenario.prob out of [0, 1]: {p}")
        _require_int(self.median_margin, "median_margin")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SecondBallotScenario:
        d = _require_mapping(data, "SecondBallotScenario")
        yes_seq = _require_sequence_not_str(d.get("yes"), "yes")
        no_seq = _require_sequence_not_str(d.get("no"), "no")
        abs_seq = _require_sequence_not_str(d.get("abstain"), "abstain")
        return cls(
            label=_require_non_empty_str(d.get("label"), "label"),
            yes=tuple(_require_non_empty_str(x, "yes") for x in yes_seq),
            no=tuple(_require_non_empty_str(x, "no") for x in no_seq),
            abstain=tuple(_require_non_empty_str(x, "abstain") for x in abs_seq),
            prob=_require_float(d.get("prob"), "prob"),
            median_margin=_require_int(d.get("median_margin"), "median_margin"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "yes": list(self.yes),
            "no": list(self.no),
            "abstain": list(self.abstain),
            "prob": self.prob,
            "median_margin": self.median_margin,
        }


@dataclass(frozen=True, slots=True)
class InvestitureSummary:
    """Art. 99 CE two-ballot and Art. 99.5 deadlock summary."""

    first_ballot: dict[str, PactProbability]
    second_ballot: dict[str, SecondBallotScenario]
    deadlock_prob: float

    def __post_init__(self) -> None:
        dp = _require_float(self.deadlock_prob, "deadlock_prob")
        if dp < 0.0 or dp > 1.0:
            raise SchemaValidationError(f"deadlock_prob out of [0, 1]: {dp}")
        if not self.first_ballot:
            raise SchemaValidationError("first_ballot must not be empty")
        if not self.second_ballot:
            raise SchemaValidationError("second_ballot must not be empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InvestitureSummary:
        d = _require_mapping(data, "InvestitureSummary")
        fb = _require_mapping(d.get("first_ballot"), "first_ballot")
        sb = _require_mapping(d.get("second_ballot"), "second_ballot")
        return cls(
            first_ballot={str(k): PactProbability.from_dict(v) for k, v in fb.items()},
            second_ballot={
                str(k): SecondBallotScenario.from_dict(v) for k, v in sb.items()
            },
            deadlock_prob=_require_float(d.get("deadlock_prob"), "deadlock_prob"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_ballot": {k: v.to_dict() for k, v in self.first_ballot.items()},
            "second_ballot": {k: v.to_dict() for k, v in self.second_ballot.items()},
            "deadlock_prob": self.deadlock_prob,
        }


@dataclass(frozen=True, slots=True)
class ForecastBundle:
    """Complete forecast output contract backing ``data/outputs/forecast_29n.json``."""

    metadata: ForecastMetadata
    national: dict[str, Any]
    tier_totals: dict[str, int]
    trend_series: dict[str, Any]
    constituencies: tuple[ConstituencyForecast, ...]
    battlegrounds: list[dict[str, Any]]
    investiture: InvestitureSummary
    pollster_scorecard: list[dict[str, Any]] = field(default_factory=list)
    sim_sample: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.constituencies) != 52:
            raise SchemaValidationError(
                f"ForecastBundle needs 52 districts, got {len(self.constituencies)}"
            )
        if {c.cpro for c in self.constituencies} != VALID_CPROS_SET:
            raise SchemaValidationError(
                "ForecastBundle constituencies must cover '01'..'52'"
            )
        if sum(c.seats_2026 for c in self.constituencies) != 350:
            raise SchemaValidationError("ForecastBundle sum(seats_2026) must equal 350")
        for req_key in ("vote_shares", "seats", "stage1_comparison", "cis_sensitivity"):
            if req_key not in self.national:
                raise SchemaValidationError(
                    f"ForecastBundle.national missing key {req_key!r}"
                )
        for tier_key in ("seguros", "probables", "en_juego"):
            if (
                _require_int(self.tier_totals.get(tier_key), f"tier_totals[{tier_key}]")
                < 0
            ):
                raise SchemaValidationError(f"tier_totals[{tier_key}] must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForecastBundle:
        d = _require_mapping(data, "ForecastBundle")
        raw_c = _require_sequence_not_str(d.get("constituencies"), "constituencies")
        raw_tt = _require_mapping(d.get("tier_totals"), "tier_totals")
        raw_bg = _require_sequence_not_str(d.get("battlegrounds", []), "battlegrounds")
        raw_ps = _require_sequence_not_str(
            d.get("pollster_scorecard", []), "pollster_scorecard"
        )
        raw_ss = _require_sequence_not_str(d.get("sim_sample", []), "sim_sample")
        return cls(
            metadata=ForecastMetadata.from_dict(
                _require_mapping(d.get("metadata"), "metadata")
            ),
            national=dict(_require_mapping(d.get("national"), "national")),
            tier_totals={
                str(k): _require_int(v, f"tier_totals[{k}]") for k, v in raw_tt.items()
            },
            trend_series=dict(_require_mapping(d.get("trend_series"), "trend_series")),
            constituencies=tuple(ConstituencyForecast.from_dict(x) for x in raw_c),
            battlegrounds=[dict(_require_mapping(x, "battleground")) for x in raw_bg],
            investiture=InvestitureSummary.from_dict(
                _require_mapping(d.get("investiture"), "investiture")
            ),
            pollster_scorecard=[dict(_require_mapping(x, "pollster")) for x in raw_ps],
            sim_sample=[dict(_require_mapping(x, "sim_sample")) for x in raw_ss],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "national": dict(self.national),
            "tier_totals": dict(self.tier_totals),
            "trend_series": dict(self.trend_series),
            "constituencies": [c.to_dict() for c in self.constituencies],
            "battlegrounds": list(self.battlegrounds),
            "investiture": self.investiture.to_dict(),
            "pollster_scorecard": list(self.pollster_scorecard),
            "sim_sample": list(self.sim_sample),
        }

    @classmethod
    def load_json(cls, path: Path) -> ForecastBundle:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class ModelBacktestMetrics:
    """Out-of-sample evaluation metrics for one Stage 1 engine in backtesting."""

    brier_score: float
    log_score: float
    vote_mae_national: float
    seat_mae_provincial: float
    seat_mae_national: float
    coverage_80_provincial_seats: float
    coverage_95_provincial_seats: float
    coverage_80_national_seats: float
    coverage_95_national_seats: float

    def __post_init__(self) -> None:
        for attr in (
            "brier_score",
            "log_score",
            "vote_mae_national",
            "seat_mae_provincial",
            "seat_mae_national",
        ):
            _require_float(getattr(self, attr), attr)
        if self.brier_score < 0.0:
            raise SchemaValidationError("brier_score must be >= 0.0")
        for attr in (
            "vote_mae_national",
            "seat_mae_provincial",
            "seat_mae_national",
        ):
            if getattr(self, attr) < 0.0:
                raise SchemaValidationError(f"{attr} must be >= 0.0")
        for attr in (
            "coverage_80_provincial_seats",
            "coverage_95_provincial_seats",
            "coverage_80_national_seats",
            "coverage_95_national_seats",
        ):
            val = _require_float(getattr(self, attr), attr)
            if val < 0.0 or val > 1.0:
                raise SchemaValidationError(f"{attr} out of [0, 1]: {val}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelBacktestMetrics:
        d = _require_mapping(data, "ModelBacktestMetrics")
        return cls(
            brier_score=_require_float(d.get("brier_score"), "brier_score"),
            log_score=_require_float(d.get("log_score"), "log_score"),
            vote_mae_national=_require_float(d.get("vote_mae_national"), "vote_mae_national"),
            seat_mae_provincial=_require_float(
                d.get("seat_mae_provincial"), "seat_mae_provincial"
            ),
            seat_mae_national=_require_float(d.get("seat_mae_national"), "seat_mae_national"),
            coverage_80_provincial_seats=_require_float(
                d.get("coverage_80_provincial_seats"), "coverage_80_provincial_seats"
            ),
            coverage_95_provincial_seats=_require_float(
                d.get("coverage_95_provincial_seats"), "coverage_95_provincial_seats"
            ),
            coverage_80_national_seats=_require_float(
                d.get("coverage_80_national_seats"), "coverage_80_national_seats"
            ),
            coverage_95_national_seats=_require_float(
                d.get("coverage_95_national_seats"), "coverage_95_national_seats"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "brier_score": self.brier_score,
            "log_score": self.log_score,
            "vote_mae_national": self.vote_mae_national,
            "seat_mae_provincial": self.seat_mae_provincial,
            "seat_mae_national": self.seat_mae_national,
            "coverage_80_provincial_seats": self.coverage_80_provincial_seats,
            "coverage_95_provincial_seats": self.coverage_95_provincial_seats,
            "coverage_80_national_seats": self.coverage_80_national_seats,
            "coverage_95_national_seats": self.coverage_95_national_seats,
        }


@dataclass(frozen=True, slots=True)
class TierCalibrationBucket:
    """Empirical hit rate for one seat-probability tier in backtesting."""

    total: int
    hits: int
    hit_rate: float

    def __post_init__(self) -> None:
        tot = _require_int(self.total, "total")
        hits = _require_int(self.hits, "hits")
        hr = _require_float(self.hit_rate, "hit_rate")
        if tot < 0 or hits < 0 or hits > tot:
            raise SchemaValidationError(f"Invalid bucket counts: hits={hits}, total={tot}")
        if hr < 0.0 or hr > 1.0:
            raise SchemaValidationError(f"hit_rate out of [0, 1]: {hr}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TierCalibrationBucket:
        d = _require_mapping(data, "TierCalibrationBucket")
        return cls(
            total=_require_int(d.get("total"), "total"),
            hits=_require_int(d.get("hits"), "hits"),
            hit_rate=_require_float(d.get("hit_rate"), "hit_rate"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "hits": self.hits, "hit_rate": self.hit_rate}


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    """Complete out-of-sample backtest summary backing ``data/outputs/backtest_summary.json``."""

    elections: tuple[str, ...]
    total_constituency_races: int
    total_seats: int
    horizons: dict[str, Any]
    models: dict[str, ModelBacktestMetrics]
    tier_calibration: dict[str, TierCalibrationBucket]
    calibrated_parameters: dict[str, Any]
    per_election: dict[str, Any]

    def __post_init__(self) -> None:
        raw_el = _require_sequence_not_str(self.elections, "elections")
        if not raw_el:
            raise SchemaValidationError("BacktestSummary.elections must not be empty")
        object.__setattr__(
            self, "elections", tuple(_require_non_empty_str(e, "election") for e in raw_el)
        )
        races = _require_int(self.total_constituency_races, "total_constituency_races")
        seats = _require_int(self.total_seats, "total_seats")
        if races < 1:
            raise SchemaValidationError("total_constituency_races must be >= 1")
        if seats < 1:
            raise SchemaValidationError("total_seats must be >= 1")
        _require_mapping(self.horizons, "horizons")
        models_m = _require_mapping(self.models, "models")
        if not models_m:
            raise SchemaValidationError("BacktestSummary.models must not be empty")
        tc_m = _require_mapping(self.tier_calibration, "tier_calibration")
        if not tc_m:
            raise SchemaValidationError("BacktestSummary.tier_calibration must not be empty")
        _require_mapping(self.calibrated_parameters, "calibrated_parameters")
        _require_mapping(self.per_election, "per_election")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BacktestSummary:
        d = _require_mapping(data, "BacktestSummary")
        return cls(
            elections=tuple(_require_sequence_not_str(d.get("elections"), "elections")),
            total_constituency_races=_require_int(
                d.get("total_constituency_races"), "total_constituency_races"
            ),
            total_seats=_require_int(d.get("total_seats"), "total_seats"),
            horizons=dict(_require_mapping(d.get("horizons", {}), "horizons")),
            models={
                str(k): ModelBacktestMetrics.from_dict(v)
                for k, v in _require_mapping(d.get("models"), "models").items()
            },
            tier_calibration={
                str(k): TierCalibrationBucket.from_dict(v)
                for k, v in _require_mapping(
                    d.get("tier_calibration"), "tier_calibration"
                ).items()
            },
            calibrated_parameters=dict(
                _require_mapping(d.get("calibrated_parameters", {}), "calibrated_parameters")
            ),
            per_election=dict(_require_mapping(d.get("per_election", {}), "per_election")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "elections": list(self.elections),
            "total_constituency_races": self.total_constituency_races,
            "total_seats": self.total_seats,
            "horizons": dict(self.horizons),
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "tier_calibration": {k: v.to_dict() for k, v in self.tier_calibration.items()},
            "calibrated_parameters": dict(self.calibrated_parameters),
            "per_election": dict(self.per_election),
        }

    @classmethod
    def load_json(cls, path: Path) -> BacktestSummary:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class PollsterRatingRecord:
    """Typed row for ``data/outputs/pollster_ratings.csv``."""

    pollster: str
    n_elections: int
    n_polls_2026: int
    shrunk_mae: float
    grade: str
    grade_weight: float
    herding_ratio: float
    herding_penalty: float
    delta_PP: float
    delta_PSOE: float
    delta_VOX: float
    delta_FA: float
    delta_PODEMOS: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pollster": self.pollster,
            "n_elections": self.n_elections,
            "n_polls_2026": self.n_polls_2026,
            "shrunk_mae": self.shrunk_mae,
            "grade": self.grade,
            "grade_weight": self.grade_weight,
            "herding_ratio": self.herding_ratio,
            "herding_penalty": self.herding_penalty,
            "delta_PP": self.delta_PP,
            "delta_PSOE": self.delta_PSOE,
            "delta_VOX": self.delta_VOX,
            "delta_FA": self.delta_FA,
            "delta_PODEMOS": self.delta_PODEMOS,
        }


def load_constituencies_catalog(path: Path | None = None) -> ConstituenciesCatalog:
    """Load and validate ``constituencies_2026.json``."""
    target = path
    if target is None:
        from elecciones.config import get_metadata_dir

        target = get_metadata_dir() / "constituencies_2026.json"
    return ConstituenciesCatalog.load_json(target)


def load_parties_catalog(path: Path | None = None) -> PartiesCatalog:
    """Load and validate ``parties_2026.json``."""
    target = path
    if target is None:
        from elecciones.config import get_metadata_dir

        target = get_metadata_dir() / "parties_2026.json"
    return PartiesCatalog.load_json(target)

