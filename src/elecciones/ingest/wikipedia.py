"""Wikipedia MediaWiki API scrapers & HTML poll table parsers (2015–2026).

Ingests and normalizes:
- ``data/polls/national_2015_2023.csv`` (2015-D, 2016-J, 2019-A, 2019-N, 2023-J)
- ``data/polls/national_2026.csv`` (live 29-N national polling, cutoff <= 2026-11-23)
- ``data/polls/regional_2026.csv`` (2024–2026 CCAA & provincial polling)
"""

from __future__ import annotations

import csv
import json
import re
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any, Final

from bs4 import BeautifulSoup, Tag

from elecciones.config import (
    BLACKOUT_CUTOFF_DATE_2026,
    HISTORICAL_ELECTION_DATES,
    get_polls_dir,
    get_raw_cache_dir,
    make_http_client,
)
from elecciones.ingest.elections import resolve_cpro_from_name, strip_accents
from elecciones.schemas import (
    PollRecord,
    RegionalPollRecord,
    SchemaValidationError,
)

# MediaWiki pages mapped to cache filenames
WIKI_POLL_PAGES: Final[dict[str, tuple[str, bool]]] = {
    # cache_key: (wiki_page_title, refresh_when_online)
    "2015-12": ("Opinion_polling_for_the_2015_Spanish_general_election", False),
    "2016-06": ("Opinion_polling_for_the_2016_Spanish_general_election", False),
    "2019-04": ("Opinion_polling_for_the_April_2019_Spanish_general_election", False),
    "2019-11": ("Opinion_polling_for_the_November_2019_Spanish_general_election", False),
    "2023-07-early": (
        "Opinion_polling_for_the_2021–2023_Spanish_general_election",
        False,
    ),
    "2023-07": ("Opinion_polling_for_the_2023_Spanish_general_election", False),
    "2026-11-early": (
        "Nationwide_opinion_polling_for_the_2026_Spanish_general_election_(2023–2024)",
        False,
    ),
    "2026-11": ("Opinion_polling_for_the_2026_Spanish_general_election", True),
    "2026-11-sub": (
        "Sub-national_opinion_polling_for_the_2026_Spanish_general_election",
        True,
    ),
}

# Expected chronological top-row years for each page's Voting Intention tables
_PAGE_TABLE_TOP_YEARS: Final[dict[str, tuple[int, ...]]] = {
    "2026-11": (2026, 2026, 2025),
    "2026-11-early": (2024, 2023),
    "2023-07": (2023, 2023, 2023, 2022),
    "2023-07-early": (2021, 2020, 2019),
    "2019-11": (2019,),
    "2019-04": (2019, 2018, 2017, 2016),
    "2016-06": (2016,),
    "2015-12": (2015, 2014, 2013, 2012, 2011),
}

# Exclusive lower / upper fieldwork_end window (prev_election, this_election - 1d)
_CYCLE_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "2015-12": (date(2011, 11, 21), date(2015, 12, 19)),
    "2016-06": (date(2015, 12, 21), date(2016, 6, 25)),
    "2019-04": (date(2016, 6, 27), date(2019, 4, 27)),
    "2019-11": (date(2019, 4, 29), date(2019, 11, 9)),
    "2023-07": (date(2019, 11, 11), date(2023, 7, 22)),
    "2026-11": (date(2023, 7, 24), BLACKOUT_CUTOFF_DATE_2026),
}

_VALID_VOTING_HEADINGS: Final[frozenset[str]] = frozenset(
    {"Voting intention estimates", "2024", "2023", "2021", "2020", "2019"}
)

_MONTHS: Final[dict[str, int]] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Canonical Wikipedia sub-national section title -> (scope_type, INE code)
_SUBNATIONAL_SCOPE_MAP: Final[dict[str, tuple[str, str]]] = {
    # 17 Autonomous Communities (codauto)
    "andalusia": ("ccaa", "01"),
    "aragon": ("ccaa", "02"),
    "asturias": ("ccaa", "03"),
    "balearic islands": ("ccaa", "04"),
    "canary islands": ("ccaa", "05"),
    "cantabria": ("ccaa", "06"),
    "castile and leon": ("ccaa", "07"),
    "castilla la mancha": ("ccaa", "08"),
    "catalonia": ("ccaa", "09"),
    "valencian community": ("ccaa", "10"),
    "extremadura": ("ccaa", "11"),
    "galicia": ("ccaa", "12"),
    "madrid": ("ccaa", "13"),
    "murcia": ("ccaa", "14"),
    "navarre": ("ccaa", "15"),
    "basque country": ("ccaa", "16"),
    "la rioja": ("ccaa", "17"),
    # Provinces & autonomous cities (cpro)
    "a coruna": ("province", "15"),
    "alava": ("province", "01"),
    "alicante": ("province", "03"),
    "biscay": ("province", "48"),
    "bizkaia": ("province", "48"),
    "cadiz": ("province", "11"),
    "castellon": ("province", "12"),
    "gipuzkoa": ("province", "20"),
    "lugo": ("province", "27"),
    "melilla": ("province", "52"),
    "ceuta": ("province", "51"),
    "ourense": ("province", "32"),
    "pontevedra": ("province", "36"),
    "valencia": ("province", "46"),
}


# ---------------------------------------------------------------------------
# Fetch & cache Wikipedia HTML via MediaWiki API
# ---------------------------------------------------------------------------


def ensure_wiki_html(
    cache_key: str,
    *,
    offline: bool = False,
    refresh_live: bool = True,
    raw_cache_dir: Path | None = None,
) -> Path:
    """Return local path to cached MediaWiki HTML, fetching if missing or live refresh."""
    title, is_live = WIKI_POLL_PAGES[cache_key]
    cache_dir = (raw_cache_dir or get_raw_cache_dir()) / "wiki_html"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{cache_key}.html"

    should_reuse = target.is_file() and target.stat().st_size > 0 and (
        offline or not (is_live and refresh_live)
    )
    if should_reuse:
        return target
    if offline:
        raise FileNotFoundError(
            f"Missing cached Wikipedia HTML {target} in offline mode (page={title!r})."
        )

    encoded_title = urllib.parse.quote(title)
    url = (
        "https://en.wikipedia.org/w/api.php?action=parse"
        f"&page={encoded_title}&prop=text&formatversion=2&format=json"
    )
    with make_http_client(timeout=45.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = json.loads(resp.content)
    html_body = payload["parse"]["text"]
    target.write_text(html_body, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Pollster & Party Header Normalization
# ---------------------------------------------------------------------------

_POLLSTER_CANONICAL_RULES: Final[tuple[tuple[str, str], ...]] = (
    ("40db", "40dB"),
    ("simple logica", "Simple Lógica"),
    ("sociometrica", "SocioMétrica"),
    ("sigma dos", "Sigma Dos"),
    ("sigmados", "Sigma Dos"),
    ("gad3", "GAD3"),
    ("gesop", "GESOP"),
    ("imop", "IMOP"),
    ("dym", "DYM"),
    ("celeste tel", "Celeste-Tel"),
    ("celeste-tel", "Celeste-Tel"),
    ("nc report", "NC Report"),
    ("em analytics", "EM-Analytics"),
    ("em-analytics", "EM-Analytics"),
    ("electopanel", "EM-Analytics"),
    ("electomania", "EM-Analytics"),
    ("sw demoscopia", "SW Demoscopia"),
    ("hamalgama metrica", "Hamalgama Métrica"),
    ("invymark", "InvyMark"),
    ("target point", "Target Point"),
    ("data10", "Data10"),
    ("keydata", "KeyData"),
    ("key data", "KeyData"),
    ("sondaxe", "Sondaxe"),
    ("metroscopia", "Metroscopia"),
    ("myword", "MyWord"),
    ("ipsos", "Ipsos"),
    ("jm a", "JM&A"),
    ("jm&a", "JM&A"),
    ("jaime miquel", "JM&A"),
    ("demoscopia y servicios", "Demoscopia y Servicios"),
    ("demoscopia servicios", "Demoscopia y Servicios"),
    ("sym consulting", "SyM Consulting"),
    ("opina 360", "Opina 360"),
    ("ateneo del dato", "Ateneo del Dato"),
    ("more in common", "More in Common"),
    ("yougov", "YouGov"),
    ("winston", "Winston"),
    ("cluster17", "Cluster17"),
    ("cluster 17", "Cluster17"),
    ("gipeyop", "GIPEyOP"),
    ("tns demoscopia", "TNS Demoscopia"),
    ("redondo asociados", "Redondo & Asociados"),
    ("intercampo", "Intercampo"),
    ("a m", "A+M"),
    ("feedback", "Feedback"),
    ("gizaker", "Gizaker"),
    ("ikerfel", "Ikerfel"),
    ("obede", "OBEDE"),
    ("cemop", "CEMOP"),
    ("asturbarometro", "Asturbarómetro"),
    ("encuestamos", "Encuestamos"),
    ("agora integral", "Ágora Integral"),
    ("advice strategic", "Advice Strategic"),
    ("sinaptica", "Sináptica"),
    ("opinionmetre", "Opinòmetre"),
    ("opinometre", "Opinòmetre"),
    ("abes", "IBES"),
    ("ibes", "IBES"),
    ("netquest", "Netquest"),
    ("deimos estadistica", "Deimos Estadística"),
    ("infortecnica", "Infortécnica"),
    ("llorente cuenca", "Llorente & Cuenca"),
    ("gi internacional", "GI Internacional"),
    ("estudio de sociologia", "Estudio de Sociología"),
    ("top position", "Top Position"),
    ("atlasintel", "AtlasIntel"),
    ("cis", "CIS"),
)


def strip_wiki_footnotes(text: str) -> str:
    """Remove bracketed footnotes like ``[ 1 ]``, ``[ a ]``, ``[12]`` and squeeze spaces."""
    cleaned = re.sub(r"\[\s*[a-zA-Z0-9]+\s*\]", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def split_pollster_and_media(raw_cell: str) -> tuple[str, str]:
    """Split Wikipedia ``Firm/Commissioner`` label into clean ``(firm, media)``."""
    cleaned = strip_wiki_footnotes(raw_cell)
    if "/" in cleaned:
        firm, media = cleaned.split("/", 1)
        return firm.strip(), media.strip()
    return cleaned.strip(), ""


def is_cis_third_party_reestimation(raw_firm: str) -> bool:
    """Return True for third-party re-cooked CIS rows like ``CIS (Opina 360)``."""
    stripped = strip_wiki_footnotes(raw_firm).strip()
    return bool(re.match(r"^CIS\s*\(", stripped, flags=re.I))


def normalize_pollster_key(raw_name: str) -> str:
    """Canonicalize pollster firm names consistently across 2015–2026 & external trackers."""
    firm, _ = split_pollster_and_media(raw_name)
    norm = strip_accents(firm).lower().strip(" .")
    norm = re.sub(r"[^a-z0-9&-]+", " ", norm).strip()
    if norm == "cis" or norm.startswith("cis "):
        return "CIS"
    collapsed = re.sub(r"[^a-z0-9]+", " ", norm).strip()
    for pat, canon in _POLLSTER_CANONICAL_RULES:
        if pat == "cis":
            continue
        if collapsed == pat or collapsed.startswith(f"{pat} ") or firm.lower().startswith(pat):
            return canon
    if norm == "jm":
        return "JM&A"
    return firm or raw_name.strip()


def _normalize_header_party(raw_header: str, *, context: str = "national") -> str | None:
    """Map a Wikipedia table column title to canonical party ID (or None if metadata)."""
    clean = strip_wiki_footnotes(raw_header).strip()
    low = strip_accents(clean).lower()
    if not low or low in {
        "polling firm commissioner",
        "polling firm/commissioner",
        "fieldwork date",
        "sample size",
        "turnout",
        "lead",
        "n",
        "?",
        "other none",
        "other/ none",
    }:
        return None
    if "polling firm" in low or "fieldwork" in low or "sample size" in low:
        return None

    # Exact & fuzzy party matching aligned with congress_*.csv & parties_2026.json
    compact = re.sub(r"[^a-z0-9]+", " ", low).strip()
    if compact in {"pp", "partido popular"}:
        return "PP"
    if compact in {"psoe", "psc", "pse ee psoe", "pse ee", "psdeg psoe", "psdeg"}:
        return "PSOE"
    if compact == "vox":
        return "VOX"
    if compact in {"fa", "frente amplio"}:
        return "FA"
    if compact == "cha":
        return "CHA"
    if compact in {"fv", "front valencia"}:
        return "COMPROMIS_FV"
    if compact == "salf":
        return "SALF"
    if compact in {"sumar", "sumem per guanyar"}:
        return "SUMAR"
    if compact in {"podemos", "podem"}:
        return "PODEMOS"
    if compact in {"unidos podemos", "unidas podemos"}:
        return "UP"
    if compact in {"iu lv", "iu upec", "iu"}:
        return "IU"
    if compact in {"c s", "cs"}:
        return "CS"
    if compact == "upyd":
        return "UPYD"
    if compact in {"mas pais", "errejon s party"}:
        return "MP"
    if compact == "compromis":
        return "COMPROMIS"
    if compact in {"erc", "erc catsi", "erc sobiranistes"}:
        return "ERC"
    if compact in {"junts", "jxcat", "ciu", "dil", "cdc", "pdecat"}:
        return "JUNTS"
    if compact in {"eh bildu", "amaiur", "bildu"}:
        return "EH_BILDU"
    if compact in {"pnv", "eaj pnv"}:
        return "PNV"
    if compact == "bng":
        return "BNG"
    if compact in {"cc", "cca", "cc nca"}:
        return "CCA"
    if compact == "nca":
        return "NC"
    if compact == "upn":
        return "UPN"
    if compact == "upn pp":
        return "UPN_PP"
    if compact == "na":
        return "NA_SUMA"
    if compact == "cup":
        return "CUP"
    if compact == "prc":
        return "PRC"
    if compact == "pacma":
        return "PACMA"
    if compact in {"gbai", "geroa bai"}:
        return "GEROA_BAI"
    if "adelante andalucia" in compact:
        return "ADELANTE"
    if compact == "ara mes":
        return "ARA_MES"
    if compact in {"teruel existe", "existe", "ev"}:
        return "TERUEL_EXISTE"
    if compact == "sy":
        return "SORIA_YA"
    if compact == "upc":
        return "UPC"
    if compact == "upl":
        return "UPL"
    if compact in {"alianca cat", "alianca catalana"}:
        return "AC"
    if compact == "cpm":
        return "CPM"
    if compact == "ie":
        return "IE"
    if context == "subnational" and clean.upper() == "FA":
        return "FA"
    return re.sub(r"[^A-Z0-9_]", "_", clean.upper()).strip("_") or None


# ---------------------------------------------------------------------------
# 2D Grid HTML Table Parser & Chronological Date Resolver
# ---------------------------------------------------------------------------


def _extract_header_text(cell: Tag) -> str:
    txt = re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
    a_tag = cell.find("a")
    img_tag = cell.find("img")
    if not txt:
        if a_tag and a_tag.get("title"):
            return str(a_tag["title"]).strip()
        if img_tag and img_tag.get("alt"):
            return str(img_tag["alt"]).strip()
    if len(txt) <= 2 and a_tag and a_tag.get("title"):
        return str(a_tag["title"]).strip()
    return txt


def _expand_wikitable_grid(
    table: Tag,
) -> tuple[list[dict[int, str]], list[list[tuple[int, int, int, str, bool]]]]:
    """Expand a wikitable into exact header column names and 2D body rows.

    Returns ``(header_cols_by_col, rows)``, where each row cell entry is
    ``(start_col, span_cols, rowspan, text, owns_cell)``.
    ``owns_cell=True`` indicates this HTML ``<tr>`` originally declared the
    ``<th>``/``<td>`` (rather than inheriting it via a previous row's ``rowspan``).
    """
    trs = table.find_all("tr")
    if not trs:
        return [], []

    # First pass: locate all HTML cells on an infinite (r, c) grid
    occupied: dict[tuple[int, int], tuple[str, int, int, Tag, bool]] = {}
    max_col = 0
    for r_idx, tr in enumerate(trs):
        col = 0
        for cell in tr.find_all(["th", "td"], recursive=False):
            while (r_idx, col) in occupied:
                col += 1
            rs = max(1, int(cell.get("rowspan", "1") or 1))
            cs = max(1, int(cell.get("colspan", "1") or 1))
            raw_txt = re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
            hdr_txt = _extract_header_text(cell) if cell.name == "th" else raw_txt
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r_idx + dr, col + dc)] = (
                        hdr_txt,
                        cs,
                        rs,
                        cell,
                        dr == 0 and dc == 0,
                    )
            col += cs
            if col > max_col:
                max_col = col

    # Identify header row depth (rows before first data row containing Pollster/date td)
    header_rows_end = 1
    for r_idx, tr in enumerate(trs[:4]):
        tds = tr.find_all("td", recursive=False)
        ths = tr.find_all("th", recursive=False)
        if ths and not tds:
            header_rows_end = r_idx + 1
        elif r_idx == 1 and not tds:
            header_rows_end = 2
        elif tds:
            break

    col_headers: list[dict[int, str]] = [{}]
    for c in range(max_col):
        # Prefer last non-empty header label across top header rows
        chosen = ""
        for hr in range(header_rows_end - 1, -1, -1):
            entry = occupied.get((hr, c))
            if entry is not None and entry[0]:
                chosen = entry[0]
                break
        col_headers[0][c] = chosen

    body_rows: list[list[tuple[int, int, int, str, bool]]] = []
    for r_idx in range(header_rows_end, len(trs)):
        row_cells: list[tuple[int, int, int, str, bool]] = []
        c = 0
        while c < max_col:
            entry = occupied.get((r_idx, c))
            if entry is None:
                c += 1
                continue
            txt, cs, rs, cell_tag, owns_origin = entry
            if not txt and cell_tag.name == "th":
                txt = _extract_header_text(cell_tag)
            raw_text = re.sub(r"\s+", " ", cell_tag.get_text(" ", strip=True))
            owns_row = (occupied.get((r_idx, c), ("", 1, 1, cell_tag, False))[4]) or (
                # Keep rowspan origin flag for column 0 to detect scenario groups
                False
            )
            _ = owns_row
            row_cells.append((c, cs, rs, raw_text, owns_origin))
            c += cs
        if row_cells:
            body_rows.append(row_cells)

    return col_headers, body_rows


def _parse_fieldwork_dates(
    raw_date: str,
    default_year: int,
    prev_end_month: int | None,
) -> tuple[date, date, int] | None:
    """Parse Wikipedia fieldwork date strings into ``(start, end, new_year)``."""
    txt = strip_wiki_footnotes(raw_date).replace("\u2014", "\u2013").replace("-", "\u2013")
    txt = txt.replace("–", "\u2013").strip()
    if not txt or txt in {"?", "N/a", "n/a"}:
        return None
    if re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", txt, re.I) is None:
        return None

    # Split range on en-dash or " to "
    parts = [p.strip() for p in re.split(r"\u2013|\s+to\s+", txt) if p.strip()]
    if len(parts) == 1:
        end_str = parts[0]
        start_str = parts[0]
    elif len(parts) == 2:
        start_str, end_str = parts
    else:
        return None

    def _parse_half(s: str) -> tuple[int, int | None, int | None] | None:
        m_full = re.match(r"^(\d{1,2})\s+([A-Za-z]{3,9})(?:\s+(\d{4}))?$", s)
        if m_full:
            d_val = int(m_full.group(1))
            m_val = _MONTHS.get(m_full.group(2).lower())
            y_val = int(m_full.group(3)) if m_full.group(3) else None
            return (d_val, m_val, y_val) if m_val is not None else None
        m_day = re.match(r"^(\d{1,2})$", s)
        if m_day:
            return (int(m_day.group(1)), None, None)
        m_my = re.match(r"^([A-Za-z]{3,9})(?:\s+(\d{4}))?$", s)
        if m_my:
            m_val = _MONTHS.get(m_my.group(1).lower())
            y_val = int(m_my.group(2)) if m_my.group(2) else None
            return (15, m_val, y_val) if m_val is not None else None
        return None

    parsed_end = _parse_half(end_str)
    parsed_start = _parse_half(start_str)
    if parsed_end is None or parsed_start is None or parsed_end[1] is None:
        return None

    ed, em, ey = parsed_end
    sd, sm, sy = parsed_start
    if sm is None:
        sm = em

    cur_year = ey if ey is not None else default_year
    if ey is None and prev_end_month is not None and em > prev_end_month + 5:
        # Table scrolled backwards across a Dec -> Jan boundary without explicit year
        cur_year -= 1

    end_year = ey if ey is not None else cur_year
    if sy is not None:
        start_year = sy
    elif sm > em:
        start_year = end_year - 1
    else:
        start_year = end_year

    try:
        d_start = date(start_year, sm, sd)
        d_end = date(end_year, em, ed)
    except ValueError:
        return None
    if d_end < d_start:
        return None
    return d_start, d_end, end_year


def _parse_sample_size(raw: str) -> int | None:
    cleaned = strip_wiki_footnotes(raw).replace(",", "").replace(".", "").replace(" ", "")
    if not cleaned or cleaned in {"?", "-", "–", "—", "Na", "N/a"}:
        return None
    m = re.match(r"^(\d+)$", cleaned)
    if not m:
        return None
    val = int(m.group(1))
    return val if val > 0 else None


def _parse_turnout(raw: str) -> float | None:
    cleaned = strip_wiki_footnotes(raw).replace("%", "").strip()
    if not cleaned or cleaned in {"?", "-", "–", "—"}:
        return None
    m_range = re.match(r"^(\d+(?:\.\d+)?)\s*[\u2013\-]\s*(\d+(?:\.\d+)?)$", cleaned)
    if m_range:
        mid = 0.5 * (float(m_range.group(1)) + float(m_range.group(2)))
        return mid / 100.0 if 0.0 <= mid <= 100.0 else None
    m_num = re.match(r"^(\d+(?:\.\d+)?)$", cleaned)
    if m_num:
        val = float(m_num.group(1))
        return val / 100.0 if 0.0 <= val <= 100.0 else None
    return None


def _parse_party_pct(
    raw: str, *, require_decimal: bool = False
) -> tuple[float | None, bool]:
    """Parse vote share `%` from cell text like ``31.5 128`` or ``6.9 10/13``.

    Returns ``(pct_0_100, has_explicit_decimal)``.
    Cells starting with ``?`` (seat-only projections) always return ``(None, False)``.
    """
    cleaned = strip_wiki_footnotes(raw).strip()
    if not cleaned or cleaned.startswith("?") or cleaned in {"–", "—", "-", "N/a", "Tie"}:
        return None, False
    # Reject seat counts like "14" or "11/12" when cell has no decimal vote share
    if re.match(r"^\d{1,3}/\d{1,3}$", cleaned):
        return None, False
    m_dec = re.match(r"^(\d{1,2}\.\d+)(?:\s|$|%|\()", cleaned)
    if m_dec:
        val = float(m_dec.group(1))
        if 0.0 <= val <= 100.0:
            return val, True
        return None, False
    if require_decimal:
        return None, False
    m_int = re.match(r"^(\d{1,2})(?:\s+\d{1,3}(?:/\d{1,3})?|\s*%?)$", cleaned)
    if m_int:
        val = float(m_int.group(1))
        if 0.0 <= val <= 100.0:
            return val, False
    return None, False


def _select_scenario_rows(
    candidates: list[dict[str, Any]], election_id: str
) -> list[dict[str, Any]]:
    """Collapse ``rowspan > 1`` scenario clusters on the same poll to one canonical row."""
    by_group: dict[int, list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []
    for item in candidates:
        gid = item.get("_scenario_gid")
        if gid is None:
            ungrouped.append(item)
        else:
            by_group.setdefault(gid, []).append(item)

    selected: list[dict[str, Any]] = list(ungrouped)
    for group in by_group.values():
        if len(group) == 1:
            selected.append(group[0])
            continue

        # In June/Apr/May 2023, prefer the unified SUMAR row matching the July ballot
        if election_id == "2023-07":
            unified = [
                g
                for g in group
                if g["shares"].get("SUMAR", 0.0) > 10.0 and "PODEMOS" not in g["shares"]
            ]
            if unified:
                selected.append(unified[0])
                continue

        # If scenarios vary by Turnout (e.g., 72 / 68 / 63), pick middle-turnout (~68%) row
        def _score(item: dict[str, Any]) -> tuple[int, float, int]:
            t = item.get("turnout")
            t_dist = abs(t - 0.68) if t is not None else 0.5
            n_parties = len(item["shares"])
            return (-n_parties, t_dist, item.get("_row_order", 0))

        group_sorted = sorted(group, key=_score)
        selected.append(group_sorted[0])

    selected.sort(key=lambda r: (r["fieldwork_end"], r["pollster"], r.get("_row_order", 0)))
    return selected


# ---------------------------------------------------------------------------
# National Poll Table Parser
# ---------------------------------------------------------------------------


def parse_national_wiki_html(
    html_path: Path,
    *,
    page_key: str,
    target_election_id: str,
) -> list[PollRecord]:
    """Parse all valid ``Voting intention estimates`` tables from a Wikipedia poll page."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    tables = soup.find_all("table", class_="wikitable")
    top_years = _PAGE_TABLE_TOP_YEARS[page_key]
    low_date, high_date = _CYCLE_WINDOWS[target_election_id]

    voting_tables: list[Tag] = []
    for tbl in tables:
        prev_h = tbl.find_previous(["h2", "h3", "h4"])
        h_txt = strip_wiki_footnotes(prev_h.get_text(" ", strip=True)) if prev_h else ""
        if h_txt in _VALID_VOTING_HEADINGS:
            voting_tables.append(tbl)

    if not voting_tables:
        raise SchemaValidationError(f"Found 0 Voting Intention tables in {html_path}")

    raw_rows_collected: list[dict[str, Any]] = []
    scenario_seq = 0
    row_order = 0

    for tbl_idx, tbl in enumerate(voting_tables):
        cur_year = top_years[tbl_idx] if tbl_idx < len(top_years) else top_years[-1]
        col_headers_list, body_rows = _expand_wikitable_grid(tbl)
        if not col_headers_list:
            continue
        col_headers = col_headers_list[0]
        prev_end_month: int | None = None
        last_scenario_gid: int | None = None
        last_col0_span_remaining = 0

        for row_cells in body_rows:
            row_order += 1
            by_col = {c: (cs, rs, txt, owns) for c, cs, rs, txt, owns in row_cells}
            if 0 not in by_col or 1 not in by_col:
                continue

            cs0, rs0, raw_pollster, owns0 = by_col[0]
            _ = cs0
            _, _, raw_date, _ = by_col[1]
            _, _, raw_n, _ = by_col.get(2, (1, 1, "", False))
            _, _, raw_turnout, _ = by_col.get(3, (1, 1, "", False))

            clean_p0 = strip_wiki_footnotes(raw_pollster)
            if not clean_p0:
                continue
            low_p0 = clean_p0.lower()
            if (
                "general election" in low_p0
                or "polling firm" in low_p0
                or "exit poll" in low_p0
                or bool(re.search(r"\b(?:elections?|referendum)\b", low_p0))
            ):
                continue

            # Skip third-party CIS re-estimations ("CIS (Opina 360)", "CIS (Logoslab)", etc.)
            if is_cis_third_party_reestimation(clean_p0):
                continue

            # Track scenario groups when column 0 itself spans multiple rows
            if owns0 and rs0 > 1:
                scenario_seq += 1
                last_scenario_gid = scenario_seq
                last_col0_span_remaining = rs0 - 1
                row_gid: int | None = last_scenario_gid
            elif not owns0 and last_col0_span_remaining > 0:
                last_col0_span_remaining -= 1
                row_gid = last_scenario_gid
            else:
                last_scenario_gid = None
                last_col0_span_remaining = 0
                row_gid = None

            date_parsed = _parse_fieldwork_dates(raw_date, cur_year, prev_end_month)
            if date_parsed is None:
                continue
            d_start, d_end, cur_year = date_parsed
            prev_end_month = d_end.month

            if not (low_date <= d_end <= high_date):
                continue

            pollster = normalize_pollster_key(clean_p0)
            _, media = split_pollster_and_media(clean_p0)
            sample_n = _parse_sample_size(raw_n)
            turnout = _parse_turnout(raw_turnout)
            is_cis = pollster == "CIS"

            # Exclude RTVE-FORTA exit polls on closing weekend (sample > 50,000)
            if (
                sample_n is not None
                and sample_n > 50_000
                and d_end == HISTORICAL_ELECTION_DATES.get(target_election_id)
            ):
                continue

            shares: dict[str, float] = {}
            has_decimal_in_row = False
            for c, cs, _rs, txt, _owns in row_cells:
                if c < 4:
                    continue
                if cs > 1:
                    span_cols = [
                        _normalize_header_party(col_headers.get(c + k, ""))
                        for k in range(cs)
                    ]
                    if set(span_cols) == {"FA", "COMPROMIS_FV"}:
                        pct, has_dec = _parse_party_pct(txt, require_decimal=True)
                        if pct is not None:
                            shares["FA_FV"] = pct / 100.0
                            has_decimal_in_row = has_decimal_in_row or has_dec
                    continue

                party_id = _normalize_header_party(col_headers.get(c, ""))
                if party_id is None:
                    continue
                pct, has_dec = _parse_party_pct(txt, require_decimal=False)
                if pct is not None:
                    shares[party_id] = shares.get(party_id, 0.0) + (pct / 100.0)
                    has_decimal_in_row = has_decimal_in_row or has_dec

            if not shares or not has_decimal_in_row:
                continue
            if sum(shares.values()) > 1.05:
                continue

            raw_rows_collected.append(
                {
                    "election_id": target_election_id,
                    "pollster": pollster,
                    "media": media,
                    "fieldwork_start": d_start,
                    "fieldwork_end": d_end,
                    "sample_size": sample_n,
                    "turnout": turnout,
                    "is_cis": is_cis,
                    "shares": shares,
                    "_scenario_gid": row_gid,
                    "_row_order": row_order,
                }
            )

    collapsed = _select_scenario_rows(raw_rows_collected, target_election_id)

    # Deduplicate exact poll signatures (pollster, fieldwork_start, fieldwork_end, shares)
    deduped: list[PollRecord] = []
    seen_keys: set[tuple[str, date, date, str]] = set()
    for item in collapsed:
        sig = (
            item["pollster"],
            item["fieldwork_start"],
            item["fieldwork_end"],
            item["media"],
        )
        if sig in seen_keys:
            continue
        seen_keys.add(sig)
        deduped.append(
            PollRecord(
                election_id=item["election_id"],
                pollster=item["pollster"],
                media=item["media"],
                fieldwork_start=item["fieldwork_start"],
                fieldwork_end=item["fieldwork_end"],
                pub_date=None,
                sample_size=item["sample_size"],
                turnout=item["turnout"],
                is_cis=item["is_cis"],
                party_shares=item["shares"],
            )
        )

    deduped.sort(key=lambda r: (r.fieldwork_end, r.fieldwork_start, r.pollster))
    return deduped


# ---------------------------------------------------------------------------
# Sub-national Poll Table Parser (2026 CCAA & Provincial Polls)
# ---------------------------------------------------------------------------


def _resolve_subnational_scope(heading: str) -> tuple[str, str] | None:
    key = strip_accents(strip_wiki_footnotes(heading)).lower()
    key = re.sub(r"[^a-z0-9]+", " ", key).strip()
    if key in _SUBNATIONAL_SCOPE_MAP:
        return _SUBNATIONAL_SCOPE_MAP[key]
    try:
        return ("province", resolve_cpro_from_name(heading))
    except SchemaValidationError:
        return None


def parse_regional_wiki_html(html_path: Path) -> list[RegionalPollRecord]:
    """Parse ``2026-11-sub.html`` keeping validated vote-% polls only (no seat-only rows)."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    tables = soup.find_all("table", class_="wikitable")
    records: list[RegionalPollRecord] = []
    seen: set[tuple[str, str, str, date, date]] = set()

    for tbl in tables:
        prev_h = tbl.find_previous(["h2", "h3", "h4"])
        if prev_h is None:
            continue
        scope = _resolve_subnational_scope(prev_h.get_text(" ", strip=True))
        if scope is None:
            continue
        scope_type, scope_code = scope

        col_headers_list, body_rows = _expand_wikitable_grid(tbl)
        if not col_headers_list:
            continue
        col_headers = col_headers_list[0]
        cur_year = 2026
        prev_end_month: int | None = None

        for row_cells in body_rows:
            by_col = {c: (cs, rs, txt, owns) for c, cs, rs, txt, owns in row_cells}
            if 0 not in by_col or 1 not in by_col:
                continue
            _, _, raw_p0, _ = by_col[0]
            _, _, raw_date, _ = by_col[1]
            _, _, raw_n, _ = by_col.get(2, (1, 1, "", False))
            clean_p0 = strip_wiki_footnotes(raw_p0)
            low_p0 = clean_p0.lower()
            if (
                not clean_p0
                or "polling firm" in low_p0
                or bool(re.search(r"\b(?:elections?|referendum|exit poll)\b", low_p0))
                or is_cis_third_party_reestimation(clean_p0)
            ):
                continue

            date_parsed = _parse_fieldwork_dates(raw_date, cur_year, prev_end_month)
            if date_parsed is None:
                continue
            d_start, d_end, cur_year = date_parsed
            prev_end_month = d_end.month

            if d_end > BLACKOUT_CUTOFF_DATE_2026 or d_end < date(2023, 7, 24):
                continue

            # Require strict decimal percentage in at least one party cell so seat-only rows fail
            shares: dict[str, float] = {}
            has_decimal = False
            for c, cs, _rs, txt, _owns in row_cells:
                if c < 4:
                    continue
                if cs > 1:
                    span_cols = [
                        _normalize_header_party(col_headers.get(c + k, ""), context="subnational")
                        for k in range(cs)
                    ]
                    if set(span_cols) == {"FA", "COMPROMIS_FV"}:
                        pct, is_dec = _parse_party_pct(txt, require_decimal=True)
                        if pct is not None:
                            shares["FA_FV"] = pct / 100.0
                            has_decimal = has_decimal or is_dec
                    continue
                pid = _normalize_header_party(col_headers.get(c, ""), context="subnational")
                if pid is None:
                    continue
                pct, is_dec = _parse_party_pct(txt, require_decimal=True)
                if pct is not None and is_dec:
                    shares[pid] = shares.get(pid, 0.0) + (pct / 100.0)
                    has_decimal = True

            if len(shares) < 2 or not has_decimal or sum(shares.values()) > 1.05:
                continue

            pollster = normalize_pollster_key(clean_p0)
            _, media = split_pollster_and_media(clean_p0)
            sample_n = _parse_sample_size(raw_n)
            sig = (scope_type, scope_code, pollster, d_start, d_end)
            if sig in seen:
                continue
            seen.add(sig)
            records.append(
                RegionalPollRecord(
                    scope_type=scope_type,  # type: ignore[arg-type]
                    scope_code=scope_code,
                    pollster=pollster,
                    fieldwork_start=d_start,
                    fieldwork_end=d_end,
                    pub_date=None,
                    sample_size=sample_n,
                    media=media,
                    is_cis=(pollster == "CIS"),
                    party_shares=shares,
                )
            )

    records.sort(key=lambda r: (r.scope_type, r.scope_code, r.fieldwork_end, r.pollster))
    return records


# ---------------------------------------------------------------------------
# CSV Writers & High-Level Builders
# ---------------------------------------------------------------------------


def _write_national_polls_csv(records: list[PollRecord], out_path: Path) -> None:
    party_freq: dict[str, int] = {}
    for rec in records:
        for p in rec.party_shares:
            party_freq[p] = party_freq.get(p, 0) + 1
    major_order = [
        "PP",
        "PSOE",
        "VOX",
        "FA",
        "COMPROMIS_FV",
        "FA_FV",
        "SUMAR",
        "PODEMOS",
        "UP",
        "CS",
        "IU",
        "MP",
        "UPYD",
        "SALF",
        "ERC",
        "JUNTS",
        "EH_BILDU",
        "PNV",
        "BNG",
        "CCA",
        "UPN_PP",
        "UPN",
        "NA_SUMA",
        "ADELANTE",
        "COMPROMIS",
        "CUP",
        "TERUEL_EXISTE",
        "GEROA_BAI",
        "PACMA",
        "PRC",
    ]
    rank = {p: i for i, p in enumerate(major_order)}
    party_cols = sorted(
        party_freq.keys(),
        key=lambda p: (rank.get(p, 999), -party_freq[p], p),
    )
    fieldnames = [
        "election_id",
        "pollster",
        "media",
        "fieldwork_start",
        "fieldwork_end",
        "pub_date",
        "sample_size",
        "turnout",
        "is_cis",
        *party_cols,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(rec.to_csv_row(party_cols))


def _write_regional_polls_csv(records: list[RegionalPollRecord], out_path: Path) -> None:
    party_freq: dict[str, int] = {}
    for rec in records:
        for p in rec.party_shares:
            party_freq[p] = party_freq.get(p, 0) + 1
    major_order = [
        "PP",
        "PSOE",
        "VOX",
        "FA",
        "COMPROMIS_FV",
        "FA_FV",
        "SUMAR",
        "PODEMOS",
        "SALF",
        "AC",
        "ERC",
        "JUNTS",
        "EH_BILDU",
        "PNV",
        "BNG",
        "CCA",
        "UPN",
        "ADELANTE",
        "COMPROMIS",
        "CUP",
        "GEROA_BAI",
        "TERUEL_EXISTE",
        "SORIA_YA",
        "UPL",
        "ARA_MES",
        "UPC",
        "NC",
        "CHA",
        "CPM",
    ]
    rank = {p: i for i, p in enumerate(major_order)}
    party_cols = sorted(
        party_freq.keys(),
        key=lambda p: (rank.get(p, 999), -party_freq[p], p),
    )
    fieldnames = [
        "scope_type",
        "scope_code",
        "pollster",
        "media",
        "fieldwork_start",
        "fieldwork_end",
        "pub_date",
        "sample_size",
        "is_cis",
        *party_cols,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            row = rec.to_csv_row(party_cols)
            row["media"] = rec.media
            row["is_cis"] = "true" if rec.is_cis else "false"
            w.writerow(row)


def build_national_2015_2023_polls(
    *,
    since: int = 2015,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
    polls_dir: Path | None = None,
    write_file: bool = True,
) -> list[PollRecord]:
    """Build ``data/polls/national_2015_2023.csv`` across selected general elections."""
    out_path = (polls_dir or get_polls_dir()) / "national_2015_2023.csv"
    try:
        all_records: list[PollRecord] = []
        if since <= 2015:
            p15 = ensure_wiki_html("2015-12", offline=offline, raw_cache_dir=raw_cache_dir)
            all_records.extend(
                parse_national_wiki_html(p15, page_key="2015-12", target_election_id="2015-12")
            )
        if since <= 2016:
            p16 = ensure_wiki_html("2016-06", offline=offline, raw_cache_dir=raw_cache_dir)
            all_records.extend(
                parse_national_wiki_html(p16, page_key="2016-06", target_election_id="2016-06")
            )
        if since <= 2019:
            p19a = ensure_wiki_html("2019-04", offline=offline, raw_cache_dir=raw_cache_dir)
            all_records.extend(
                parse_national_wiki_html(p19a, page_key="2019-04", target_election_id="2019-04")
            )
            p19n = ensure_wiki_html("2019-11", offline=offline, raw_cache_dir=raw_cache_dir)
            all_records.extend(
                parse_national_wiki_html(p19n, page_key="2019-11", target_election_id="2019-11")
            )
        if since <= 2023:
            p23_early = ensure_wiki_html(
                "2023-07-early", offline=offline, raw_cache_dir=raw_cache_dir
            )
            all_records.extend(
                parse_national_wiki_html(
                    p23_early, page_key="2023-07-early", target_election_id="2023-07"
                )
            )
            p23 = ensure_wiki_html("2023-07", offline=offline, raw_cache_dir=raw_cache_dir)
            all_records.extend(
                parse_national_wiki_html(p23, page_key="2023-07", target_election_id="2023-07")
            )
    except FileNotFoundError:
        if offline and out_path.is_file():
            return [r for r in load_national_polls(out_path) if int(r.election_id[:4]) >= since]
        raise

    all_records.sort(key=lambda r: (r.election_id, r.fieldwork_end, r.fieldwork_start, r.pollster))
    if write_file:
        _write_national_polls_csv(all_records, out_path)
    return all_records


def build_national_2026_polls(
    *,
    offline: bool = False,
    refresh_live: bool = True,
    raw_cache_dir: Path | None = None,
    polls_dir: Path | None = None,
    write_file: bool = True,
) -> list[PollRecord]:
    """Build ``data/polls/national_2026.csv`` from live Wikipedia 2026 tables."""
    out_path = (polls_dir or get_polls_dir()) / "national_2026.csv"
    try:
        p26_early = ensure_wiki_html(
            "2026-11-early",
            offline=offline,
            refresh_live=refresh_live,
            raw_cache_dir=raw_cache_dir,
        )
        p26 = ensure_wiki_html(
            "2026-11",
            offline=offline,
            refresh_live=refresh_live,
            raw_cache_dir=raw_cache_dir,
        )
    except FileNotFoundError:
        if offline and out_path.is_file():
            return load_national_polls(out_path)
        raise
    early_records = parse_national_wiki_html(
        p26_early, page_key="2026-11-early", target_election_id="2026-11"
    )
    live_records = parse_national_wiki_html(p26, page_key="2026-11", target_election_id="2026-11")
    by_key: dict[tuple[str, date, date], PollRecord] = {}
    for rec in early_records + live_records:
        by_key[(rec.pollster, rec.fieldwork_start, rec.fieldwork_end)] = rec
    records = sorted(
        by_key.values(),
        key=lambda r: (r.fieldwork_end, r.fieldwork_start, r.pollster),
    )
    if write_file:
        _write_national_polls_csv(records, out_path)
    return records


def build_regional_2026_polls(
    *,
    offline: bool = False,
    refresh_live: bool = True,
    raw_cache_dir: Path | None = None,
    polls_dir: Path | None = None,
    write_file: bool = True,
) -> list[RegionalPollRecord]:
    """Build ``data/polls/regional_2026.csv`` from sub-national Wikipedia tables."""
    out_path = (polls_dir or get_polls_dir()) / "regional_2026.csv"
    try:
        psub = ensure_wiki_html(
            "2026-11-sub",
            offline=offline,
            refresh_live=refresh_live,
            raw_cache_dir=raw_cache_dir,
        )
    except FileNotFoundError:
        if offline and out_path.is_file():
            return load_regional_polls(out_path)
        raise
    records = parse_regional_wiki_html(psub)
    if write_file:
        _write_regional_polls_csv(records, out_path)
    return records


def load_national_polls(
    csv_path: Path,
) -> list[PollRecord]:
    """Load and validate a national poll CSV snapshot into ``PollRecord`` objects."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing national polls CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return [PollRecord.from_csv_row(row) for row in csv.DictReader(fh)]


def load_regional_polls(csv_path: Path) -> list[RegionalPollRecord]:
    """Load and validate a regional/provincial poll CSV into ``RegionalPollRecord``."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing regional polls CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return [RegionalPollRecord.from_csv_row(row) for row in csv.DictReader(fh)]
