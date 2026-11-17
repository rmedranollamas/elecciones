"""Official election result & BOE metadata builders for Spain (2015–2026).

Builds and validates:
- ``data/metadata/constituencies_2026.json`` (BOE-A-2026-20742 + 23-J Cuadro I)
- ``data/metadata/parties_2026.json`` (BOE-A-2026-23103 + corr. BOE-A-2026-23383)
- ``data/elections/congress_{2015_12,2016_06,2019_04,2019_11,2023_07}.csv``
- ``data/elections/europarl_2024_06.csv`` (BOE-A-2024-13092 + errata BOE-A-2024-15344)
- ``data/elections/regional_2024_2026.csv`` (2024–2026 CCAA & provincial results)
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from elecciones.config import (
    CONGRESS_TOTAL_SEATS,
    get_elections_dir,
    get_metadata_dir,
    get_raw_cache_dir,
    make_http_client,
)
from elecciones.schemas import (
    ALL_CPROS,
    AUTONOMOUS_CITY_CPROS,
    ConstituenciesCatalog,
    Constituency,
    ConstituencyElectionResult,
    DistrictType,
    PartiesCatalog,
    PartySpec,
    SchemaValidationError,
    SenateSpec,
    SubdistrictSenateSpec,
    load_constituencies_catalog,
    load_parties_catalog,
)

# ---------------------------------------------------------------------------
# Authoritative URLs & statutory fallback tables
# ---------------------------------------------------------------------------

BOE_URLS: Final[dict[str, str]] = {
    "boe_2026_convocatoria.xml": "https://www.boe.es/diario_boe/xml.php?id=BOE-A-2026-20742",
    "boe_2026_candidaturas.xml": "https://www.boe.es/diario_boe/xml.php?id=BOE-A-2026-23103",
    "boe_2026_cand_corr.xml": "https://www.boe.es/diario_boe/xml.php?id=BOE-A-2026-23383",
    "boe_2023_07.xml": "https://www.boe.es/diario_boe/xml.php?id=BOE-A-2023-18907",
    "boe_2023_07.pdf": "https://www.boe.es/boe/dias/2023/09/01/pdfs/BOE-A-2023-18907.pdf",
    "boe_2024_ep.xml": "https://www.boe.es/diario_boe/xml.php?id=BOE-A-2024-13092",
    "boe_2024_ep_errata.xml": "https://www.boe.es/diario_boe/xml.php?id=BOE-A-2024-15344",
}

INFOELECTORAL_DAT_URLS: Final[dict[str, str]] = {
    "02201512": "https://raw.githubusercontent.com/JaimeObregon/infoelectoral/main/archivos/02_201512_1.zip",
    "02201606": "https://raw.githubusercontent.com/JaimeObregon/infoelectoral/main/archivos/02_201606_1.zip",
    "02201904": "https://raw.githubusercontent.com/JaimeObregon/infoelectoral/main/archivos/02_201904_1.zip",
    "02201911": "https://raw.githubusercontent.com/JaimeObregon/infoelectoral/main/archivos/02_201911_1.zip",
}

WIKI_REGIONAL_TITLES: Final[tuple[str, ...]] = (
    "2024 Galician regional election",
    "2024 Basque regional election",
    "2024 Catalan regional election",
    "2025 Extremaduran regional election",
    "2026 Aragonese regional election",
    "2026 Castilian-Leonese regional election",
    "2026 Andalusian regional election",
)

CCAA_NAMES: Final[dict[str, str]] = {
    "01": "Andalucía",
    "02": "Aragón",
    "03": "Principado de Asturias",
    "04": "Illes Balears",
    "05": "Canarias",
    "06": "Cantabria",
    "07": "Castilla y León",
    "08": "Castilla-La Mancha",
    "09": "Cataluña",
    "10": "Comunitat Valenciana",
    "11": "Extremadura",
    "12": "Galicia",
    "13": "Comunidad de Madrid",
    "14": "Región de Murcia",
    "15": "Comunidad Foral de Navarra",
    "16": "País Vasco",
    "17": "La Rioja",
    "18": "Ceuta",
    "19": "Melilla",
}

# Canonical 52 constituencies:
# (cpro, name, short_name, codauto, seats_2015, seats_2016, seats_2019, seats_2023,
#  cartogram_row, cartogram_col)
PROVINCE_SPECS: Final[tuple[tuple[str, str, str, str, int, int, int, int, int, int], ...]] = (
    ("01", "Araba/Álava", "Álava", "16", 4, 4, 4, 4, 1, 5),
    ("02", "Albacete", "Albacete", "08", 4, 4, 4, 4, 5, 5),
    ("03", "Alicante/Alacant", "Alicante", "10", 12, 12, 12, 12, 5, 7),
    ("04", "Almería", "Almería", "01", 6, 6, 6, 6, 7, 6),
    ("05", "Ávila", "Ávila", "07", 3, 3, 3, 3, 3, 3),
    ("06", "Badajoz", "Badajoz", "11", 6, 6, 6, 5, 5, 2),
    ("07", "Illes Balears", "Illes Balears", "04", 8, 8, 8, 8, 3, 9),
    ("08", "Barcelona", "Barcelona", "09", 31, 31, 32, 32, 1, 9),
    ("09", "Burgos", "Burgos", "07", 4, 4, 4, 4, 1, 4),
    ("10", "Cáceres", "Cáceres", "11", 4, 4, 4, 4, 4, 2),
    ("11", "Cádiz", "Cádiz", "01", 9, 9, 9, 9, 7, 3),
    ("12", "Castellón/Castelló", "Castellón", "10", 5, 5, 5, 5, 3, 7),
    ("13", "Ciudad Real", "C. Real", "08", 5, 5, 5, 5, 5, 4),
    ("14", "Córdoba", "Córdoba", "01", 6, 6, 6, 6, 6, 4),
    ("15", "A Coruña", "A Coruña", "12", 8, 8, 8, 8, 0, 0),
    ("16", "Cuenca", "Cuenca", "08", 3, 3, 3, 3, 4, 5),
    ("17", "Girona", "Girona", "09", 6, 6, 6, 6, 0, 9),
    ("18", "Granada", "Granada", "01", 7, 7, 7, 7, 7, 5),
    ("19", "Guadalajara", "Guadalajara", "08", 3, 3, 3, 3, 3, 5),
    ("20", "Gipuzkoa", "Gipuzkoa", "16", 6, 6, 6, 6, 0, 5),
    ("21", "Huelva", "Huelva", "01", 5, 5, 5, 5, 6, 2),
    ("22", "Huesca", "Huesca", "02", 3, 3, 3, 3, 0, 7),
    ("23", "Jaén", "Jaén", "01", 5, 5, 5, 5, 6, 5),
    ("24", "León", "León", "07", 5, 4, 4, 4, 1, 2),
    ("25", "Lleida", "Lleida", "09", 4, 4, 4, 4, 1, 8),
    ("26", "La Rioja", "La Rioja", "17", 4, 4, 4, 4, 1, 6),
    ("27", "Lugo", "Lugo", "12", 4, 4, 4, 4, 0, 1),
    ("28", "Madrid", "Madrid", "13", 36, 36, 37, 37, 3, 4),
    ("29", "Málaga", "Málaga", "01", 11, 11, 11, 11, 7, 4),
    ("30", "Murcia", "Murcia", "14", 10, 10, 10, 10, 6, 6),
    ("31", "Navarra", "Navarra", "15", 5, 5, 5, 5, 0, 6),
    ("32", "Ourense", "Ourense", "12", 4, 4, 4, 4, 1, 1),
    ("33", "Asturias", "Asturias", "03", 8, 8, 7, 7, 0, 2),
    ("34", "Palencia", "Palencia", "07", 3, 3, 3, 3, 1, 3),
    ("35", "Las Palmas", "Las Palmas", "05", 8, 8, 8, 8, 8, 1),
    ("36", "Pontevedra", "Pontevedra", "12", 7, 7, 7, 7, 1, 0),
    ("37", "Salamanca", "Salamanca", "07", 4, 4, 4, 4, 3, 2),
    ("38", "Santa Cruz de Tenerife", "S. C. Tenerife", "05", 7, 7, 7, 7, 8, 0),
    ("39", "Cantabria", "Cantabria", "06", 5, 5, 5, 5, 0, 3),
    ("40", "Segovia", "Segovia", "07", 3, 3, 3, 3, 2, 4),
    ("41", "Sevilla", "Sevilla", "01", 12, 12, 12, 12, 6, 3),
    ("42", "Soria", "Soria", "07", 2, 2, 2, 2, 2, 5),
    ("43", "Tarragona", "Tarragona", "09", 6, 6, 6, 6, 2, 8),
    ("44", "Teruel", "Teruel", "02", 3, 3, 3, 3, 2, 7),
    ("45", "Toledo", "Toledo", "08", 6, 6, 6, 6, 4, 4),
    ("46", "Valencia/València", "Valencia", "10", 15, 16, 15, 16, 4, 7),
    ("47", "Valladolid", "Valladolid", "07", 5, 5, 5, 5, 2, 3),
    ("48", "Bizkaia", "Bizkaia", "16", 8, 8, 8, 8, 0, 4),
    ("49", "Zamora", "Zamora", "07", 3, 3, 3, 3, 2, 2),
    ("50", "Zaragoza", "Zaragoza", "02", 7, 7, 7, 7, 1, 7),
    ("51", "Ceuta", "Ceuta", "18", 1, 1, 1, 1, 8, 7),
    ("52", "Melilla", "Melilla", "19", 1, 1, 1, 1, 8, 8),
)

INSULAR_SENATE: Final[dict[str, dict[str, Any]]] = {
    "07": {
        "direct_seats": 5,
        "max_votes_per_ballot": 2,
        "subdistricts": [
            {"id": "07-MALLORCA", "name": "Mallorca", "direct_seats": 3, "max_votes_per_ballot": 2},
            {"id": "07-MENORCA", "name": "Menorca", "direct_seats": 1, "max_votes_per_ballot": 1},
            {
                "id": "07-IBIZA_FORMENTERA",
                "name": "Ibiza-Formentera",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
        ],
    },
    "35": {
        "direct_seats": 5,
        "max_votes_per_ballot": 2,
        "subdistricts": [
            {
                "id": "35-GRAN_CANARIA",
                "name": "Gran Canaria",
                "direct_seats": 3,
                "max_votes_per_ballot": 2,
            },
            {
                "id": "35-FUERTEVENTURA",
                "name": "Fuerteventura",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
            {
                "id": "35-LANZAROTE",
                "name": "Lanzarote",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
        ],
    },
    "38": {
        "direct_seats": 6,
        "max_votes_per_ballot": 2,
        "subdistricts": [
            {"id": "38-TENERIFE", "name": "Tenerife", "direct_seats": 3, "max_votes_per_ballot": 2},
            {"id": "38-LA_PALMA", "name": "La Palma", "direct_seats": 1, "max_votes_per_ballot": 1},
            {
                "id": "38-LA_GOMERA",
                "name": "La Gomera",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
            {
                "id": "38-EL_HIERRO",
                "name": "El Hierro",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
        ],
    },
}

SPANISH_NUMBER_WORDS: Final[dict[str, int]] = {
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "dieciseis.": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "treinta y uno": 31,
    "treinta y dos": 32,
    "treinta y seis": 36,
    "treinta y siete": 37,
    "treinta y ocho": 38,
}


def strip_accents(text: str) -> str:
    """Fold Unicode accents/diacritics to ASCII lowercase form for resilient matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_province_key(raw: str) -> str:
    """Normalize arbitrary Spanish/Wikipedia/BOE province string to a lookup token."""
    cleaned = re.sub(r"\.\d+$", "", raw.strip())
    cleaned = re.sub(
        r"^(?:junta\s+electoral\s+(?:provincial\s+)?de\s+)", "", cleaned, flags=re.I
    )
    cleaned = strip_accents(cleaned).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    return cleaned


_PROVINCE_ALIASES: Final[dict[str, str]] = {
    "araba alava": "01",
    "araba": "01",
    "alava": "01",
    "albacete": "02",
    "alicante alacant": "03",
    "alicante": "03",
    "alacant": "03",
    "almeria": "04",
    "avila": "05",
    "badajoz": "06",
    "illes balears": "07",
    "balears illes": "07",
    "islas baleares": "07",
    "balears": "07",
    "baleares": "07",
    "barcelona": "08",
    "burgos": "09",
    "caceres": "10",
    "cadiz": "11",
    "castellon castello": "12",
    "castellon": "12",
    "castello": "12",
    "ciudad real": "13",
    "cordoba": "14",
    "a coruna": "15",
    "coruna a": "15",
    "coruna la": "15",
    "la coruna": "15",
    "cuenca": "16",
    "girona": "17",
    "gerona": "17",
    "granada": "18",
    "guadalajara": "19",
    "gipuzkoa": "20",
    "guipuzcoa": "20",
    "huelva": "21",
    "huesca": "22",
    "jaen": "23",
    "leon": "24",
    "lleida": "25",
    "lerida": "25",
    "la rioja": "26",
    "rioja la": "26",
    "rioja": "26",
    "lugo": "27",
    "madrid": "28",
    "malaga": "29",
    "murcia": "30",
    "navarra": "31",
    "ourense": "32",
    "orense": "32",
    "asturias": "33",
    "principado de asturias": "33",
    "palencia": "34",
    "las palmas": "35",
    "palmas las": "35",
    "pontevedra": "36",
    "salamanca": "37",
    "santa cruz de tenerife": "38",
    "s c tenerife": "38",
    "tenerife": "38",
    "cantabria": "39",
    "segovia": "40",
    "sevilla": "41",
    "seville": "41",
    "soria": "42",
    "tarragona": "43",
    "teruel": "44",
    "toledo": "45",
    "valencia valencia": "46",
    "valencia": "46",
    "valladolid": "47",
    "bizkaia": "48",
    "vizcaya": "48",
    "biscay": "48",
    "zamora": "49",
    "zaragoza": "50",
    "ceuta": "51",
    "melilla": "52",
}


def resolve_cpro_from_name(raw_name: str) -> str:
    """Resolve an arbitrary province/city name to INE ``cpro`` ``'01'..'52'``."""
    tok = normalize_province_key(raw_name)
    if tok in _PROVINCE_ALIASES:
        return _PROVINCE_ALIASES[tok]
    for alias, cpro in _PROVINCE_ALIASES.items():
        if alias in tok:
            return cpro
    raise SchemaValidationError(f"Unable to map province name {raw_name!r} to INE cpro")


def _parse_es_int(raw: str | None) -> int:
    if raw is None:
        return 0
    cleaned = re.sub(r"[.\s,\xa0]", "", raw.strip())
    if not cleaned or cleaned in {"-", "—", "–"}:
        return 0
    return int(cleaned)


def ensure_cached_file(
    filename: str,
    url: str,
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
) -> Path:
    """Return local path inside ``data/raw_cache/``, downloading if missing and online."""
    cache_dir = raw_cache_dir or get_raw_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    if offline:
        raise FileNotFoundError(
            f"Missing cached file {target} in offline mode (URL: {url})."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with make_http_client(timeout=45.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    return target


def _ensure_infoelectoral_dir(
    election_code: str,
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
) -> Path:
    cache_dir = raw_cache_dir or get_raw_cache_dir()
    subdir = cache_dir / "infoelectoral" / election_code
    if subdir.is_dir() and list(subdir.glob("08*.DAT")) and list(subdir.glob("07*.DAT")):
        return subdir
    if offline:
        raise FileNotFoundError(
            f"Missing Infoelectoral cache in {subdir} in offline mode."
        )
    subdir.mkdir(parents=True, exist_ok=True)
    url = INFOELECTORAL_DAT_URLS[election_code]
    with make_http_client(timeout=60.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(subdir)
    return subdir


# ---------------------------------------------------------------------------
# Metadata Builders: constituencies_2026.json & parties_2026.json
# ---------------------------------------------------------------------------


def parse_boe_2026_seats(xml_path: Path) -> dict[str, int]:
    """Parse exact 52-district Congress apportionment from ``BOE-A-2026-20742`` XML."""
    root = ET.parse(xml_path).getroot()
    seats_map: dict[str, int] = {}
    for tb in root.findall(".//table"):
        for tr in tb.findall(".//tr"):
            cells = [re.sub(r"\s+", " ", "".join(c.itertext()).strip()) for c in tr]
            if len(cells) < 2:
                continue
            prov_txt, num_txt = cells[0].rstrip("."), cells[1].rstrip(".")
            word_key = strip_accents(num_txt).lower().strip()
            if word_key in SPANISH_NUMBER_WORDS:
                cpro = resolve_cpro_from_name(prov_txt)
                seats_map[cpro] = SPANISH_NUMBER_WORDS[word_key]
    for cpro in ALL_CPROS:
        if cpro in AUTONOMOUS_CITY_CPROS:
            seats_map.setdefault(cpro, 1)
    if len(seats_map) != 52 or sum(seats_map.values()) != CONGRESS_TOTAL_SEATS:
        raise SchemaValidationError(
            f"BOE 2026 seats invalid: len={len(seats_map)}, sum={sum(seats_map.values())}"
        )
    return seats_map


def parse_boe_2023_cuadro_uno(pdf_path: Path) -> dict[str, dict[str, int]]:
    """Extract exact 23-J ``censo, votantes, validos, blancos, nulos`` via ``pdftotext``."""
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    out: dict[str, dict[str, int]] = {}
    line_re = re.compile(
        r"^\s*([^\d]+?)\s+"
        r"(\d{1,3}(?:\.\d{3})+|\d+)\s+"
        r"(\d{1,3}(?:\.\d{3})+|\d+)\s+"
        r"(\d{1,3}(?:\.\d{3})+|\d+)\s+"
        r"(\d{1,3}(?:\.\d{3})+|\d+)\s+"
        r"(\d{1,3}(?:\.\d{3})+|\d+)\s+"
        r"(\d{1,3}(?:\.\d{3})+|\d+)\s*$"
    )
    in_cuadro_uno = False
    for ln in proc.stdout.splitlines():
        if "Votos a Candidaturas" in ln or ln.strip() == "CUADRO I":
            in_cuadro_uno = True
            continue
        if not in_cuadro_uno:
            continue
        if ln.strip() == "CUADRO II":
            break
        m = line_re.match(ln)
        if not m:
            continue
        prov_name = m.group(1).strip()
        if "total" in strip_accents(prov_name).lower():
            continue
        try:
            cpro = resolve_cpro_from_name(prov_name)
        except SchemaValidationError:
            continue
        censo = _parse_es_int(m.group(2))
        votantes = _parse_es_int(m.group(3))
        validos = _parse_es_int(m.group(4))
        candidaturas = _parse_es_int(m.group(5))
        blancos = _parse_es_int(m.group(6))
        nulos = _parse_es_int(m.group(7))
        if validos == candidaturas + blancos and votantes == validos + nulos:
            out[cpro] = {
                "censo": censo,
                "votantes": votantes,
                "validos": validos,
                "candidaturas": candidaturas,
                "blancos": blancos,
                "nulos": nulos,
            }
    if len(out) != 52:
        raise SchemaValidationError(
            f"BOE 2023 Cuadro I parsed {len(out)} provinces != 52 from {pdf_path}"
        )
    return out


def build_constituencies_catalog(
    *,
    offline: bool = False,
    metadata_dir: Path | None = None,
    raw_cache_dir: Path | None = None,
    write_file: bool = True,
) -> ConstituenciesCatalog:
    """Build and validate ``data/metadata/constituencies_2026.json``."""
    out_path = (metadata_dir or get_metadata_dir()) / "constituencies_2026.json"
    try:
        conv_xml = ensure_cached_file(
            "boe_2026_convocatoria.xml",
            BOE_URLS["boe_2026_convocatoria.xml"],
            offline=offline,
            raw_cache_dir=raw_cache_dir,
        )
        pdf_2023 = ensure_cached_file(
            "boe_2023_07.pdf",
            BOE_URLS["boe_2023_07.pdf"],
            offline=offline,
            raw_cache_dir=raw_cache_dir,
        )
    except FileNotFoundError:
        if offline and out_path.is_file():
            return load_constituencies_catalog(out_path)
        raise
    seats_2026_map = parse_boe_2026_seats(conv_xml)
    cuadro_2023 = parse_boe_2023_cuadro_uno(pdf_2023)

    constituencies: list[Constituency] = []
    for cpro, name, short_name, codauto, _s15, _s16, s19, s23, row, col in PROVINCE_SPECS:
        is_city = cpro in AUTONOMOUS_CITY_CPROS
        dtype: DistrictType = "autonomous_city" if is_city else "province"
        if is_city:
            senate = SenateSpec(direct_seats=2, max_votes_per_ballot=2, subdistricts=())
        elif cpro in INSULAR_SENATE:
            ins = INSULAR_SENATE[cpro]
            senate = SenateSpec(
                direct_seats=int(ins["direct_seats"]),
                max_votes_per_ballot=int(ins["max_votes_per_ballot"]),
                subdistricts=tuple(
                    SubdistrictSenateSpec.from_dict(s) for s in ins["subdistricts"]
                ),
            )
        else:
            senate = SenateSpec(direct_seats=4, max_votes_per_ballot=3, subdistricts=())

        c23 = cuadro_2023[cpro]
        constituencies.append(
            Constituency(
                cpro=cpro,
                name=name,
                short_name=short_name,
                codauto=codauto,
                ccaa_name=CCAA_NAMES[codauto],
                district_type=dtype,
                threshold=0.0 if is_city else 0.03,
                seats_2019=s19,
                seats_2023=s23,
                seats_2026=seats_2026_map[cpro],
                cartogram_row=row,
                cartogram_col=col,
                senate=senate,
                censo_2023=c23["censo"],
                validos_2023=c23["validos"],
            )
        )

    catalog = ConstituenciesCatalog(
        constituencies=tuple(constituencies),
        chamber="congress",
        election_date="2026-11-29",
        total_seats=350,
        boe_source="BOE-A-2026-20742",
        total_senate_direct_seats=208,
    )
    if write_file:
        out_path = (metadata_dir or get_metadata_dir()) / "constituencies_2026.json"
        catalog.save_json(out_path)
    return catalog


def _classify_boe_2026_candidacy(raw_name: str) -> str:
    """Classify a proclaimed 29-N BOE Congress list string to a canonical party ID."""
    up = strip_accents(raw_name).upper().strip()
    up = re.sub(r"^\d+\.\s*", "", up)

    if "UPN-PP" in up or ("UNION DEL PUEBLO NAVARRO" in up and "PARTIDO POPULAR" in up):
        return "UPN_PP"
    if "COMPROMIS" in up and "FRONT VALENCIA" in up:
        return "COMPROMIS_FV"
    if "UNIDAD DEL PUEBLO CANARIO" in up or "(NC-BC, PODEMOS" in up or "UPC (" in up:
        return "UPC"
    if "ARA MES" in up or "ARAMES" in up:
        return "ARA_MES"
    if "ADELANTE ANDALUCIA" in up:
        return "ADELANTE"
    if "ALHORA" in up:
        return "ALHORA"
    if (
        "FRENTE AMPLIO" in up
        or "FRENTE AMPLIU" in up
        or "COMUNS-FRONT AMPLI" in up
        or "FRONTE AMPLA" in up
        or "FRONTE ZABALA" in up
    ):
        return "FA"
    if "PODEMOS" in up:
        return "PODEMOS"
    if "PARTIDO POPULAR" in up or "PARTIT POPULAR" in up or up.endswith("(PP)"):
        return "PP"
    if (
        "SOCIALISTA OBRERO ESPANOL" in up
        or "SOCIALISTES DE CATALUNYA" in up
        or "SOCIALISTA DE EUSKADI" in up
        or "SOCIALISTAS DE GALICIA" in up
        or "SOCIALISTA DE LAS ISLAS BALEARES" in up
        or "SOCIALISTA DE NAVARRA" in up
        or "(PSOE)" in up
        or "(PSC)" in up
        or "PSDEG-PSOE" in up
        or "PSIB-PSOE" in up
        or "PSN-PSOE" in up
    ):
        return "PSOE"
    if up == "VOX" or "(VOX)" in up or up.startswith("VOX "):
        return "VOX"
    if "SE ACABO LA FIESTA" in up or "(SALF)" in up:
        return "SALF"
    if "ESQUERRA REPUBLICANA DE CATALUNYA" in up or "(ERC)" in up:
        return "ERC"
    if "JUNTS PER CATALUNYA" in up or "(JXCAT)" in up or "(JUNTS)" in up:
        return "JUNTS"
    if "EUSKAL HERRIA BILDU" in up or "(EH BILDU)" in up:
        return "EH_BILDU"
    if "PARTIDO NACIONALISTA VASCO" in up or "(EAJ-PNV)" in up:
        return "PNV"
    if "GEROA BAI" in up or "(GBAI)" in up:
        return "GEROA_BAI"
    if "BLOQUE NACIONALISTA GALEGO" in up or "(BNG)" in up:
        return "BNG"
    if "COALICION CANARIA" in up or "CCA-1" in up or "MUNICIPALISTAS PRIMERO CANARIAS" in up:
        return "CCA"
    if "TERUEL EXISTE" in up:
        return "TERUEL_EXISTE"
    if "SORIA" in up and "YA" in up:
        return "SORIA_YA"
    if "PARTIDO ANIMALISTA" in up or "(PACMA)" in up:
        return "PACMA"
    if "FRENTE OBRERO" in up or "(FO)" in up:
        return "FO"
    return "OTROS"


def parse_boe_2026_candidacy_mask(
    cand_xml: Path,
    corr_xml: Path | None = None,
) -> dict[str, dict[str, int]]:
    """Parse exact 52-district ``M_{c,p}`` candidacy mask from BOE Nov 3 + Nov 7 XML."""
    root = ET.parse(cand_xml).getroot()
    txt = root.find("texto")
    if txt is None:
        raise SchemaValidationError(f"Missing <texto> in {cand_xml}")

    by_cpro: dict[str, dict[str, str]] = {c: {} for c in ALL_CPROS}
    cur_cpro: str | None = None
    cur_chamber: str | None = None
    last_cid: str | None = None

    for p in txt.findall(".//p"):
        cls = p.attrib.get("class", "")
        raw = "".join(p.itertext()).strip().replace("\u2003", " ").replace("\xa0", " ")
        line = re.sub(r"\s+", " ", raw)
        if not line:
            continue
        if cls == "centro_negrita" and "JUNTA ELECTORAL" in strip_accents(line).upper():
            cur_cpro = resolve_cpro_from_name(line)
            cur_chamber = None
            last_cid = None
        elif cls == "centro_cursiva" and cur_cpro is not None:
            cur_chamber = strip_accents(line).upper()
            last_cid = None
        elif (
            cls == "centro_redonda"
            and cur_cpro is not None
            and cur_chamber
            and "CONGRESO" in cur_chamber
        ):
            m_num = re.match(r"^(\d+)\.\s*(.+)$", line)
            list_key = m_num.group(1) if m_num else line
            list_label = m_num.group(2) if m_num else line
            by_cpro[cur_cpro][list_key] = list_label
            last_cid = list_key
        elif (
            cur_cpro is not None
            and cur_chamber
            and "CONGRESO" in cur_chamber
            and last_cid is not None
            and "no proclamada" in strip_accents(line).lower()
        ):
            by_cpro[cur_cpro].pop(last_cid, None)
            last_cid = None

    # Apply BOE-A-2026-23383 errata / label corrections (e.g. Navarra EH BILDU-ERC -> EH BILDU)
    if corr_xml is not None and corr_xml.is_file():
        croot = ET.parse(corr_xml).getroot()
        ctxt = croot.find("texto")
        if ctxt is not None:
            cur_cpro = None
            cur_chamber = None
            waiting_debe = False
            replace_target: str | None = None
            for p in ctxt.findall(".//p"):
                cls = p.attrib.get("class", "")
                line = re.sub(
                    r"\s+",
                    " ",
                    "".join(p.itertext()).strip().replace("\u2003", " ").replace("\xa0", " "),
                )
                if not line:
                    continue
                if cls == "centro_negrita" and "JUNTA ELECTORAL" in strip_accents(line).upper():
                    cur_cpro = resolve_cpro_from_name(line)
                    cur_chamber = None
                elif cls == "centro_cursiva" and cur_cpro is not None:
                    cur_chamber = strip_accents(line).upper()
                elif cur_cpro is not None and cur_chamber and "CONGRESO" in cur_chamber:
                    low = strip_accents(line).lower()
                    if low.startswith("donde dice"):
                        waiting_debe = False
                    elif low.startswith("debe decir"):
                        waiting_debe = True
                    elif cls == "centro_redonda":
                        clean_l = line.strip("«» ")
                        m_num = re.match(r"^(\d+)\.\s*(.+)$", clean_l)
                        if m_num:
                            lk, ll = m_num.group(1), m_num.group(2)
                            if waiting_debe and replace_target is not None:
                                by_cpro[cur_cpro][lk] = ll
                                replace_target = None
                            else:
                                replace_target = lk

    mask: dict[str, dict[str, int]] = {}
    for cpro, entries in by_cpro.items():
        if not entries:
            raise SchemaValidationError(f"Zero proclaimed Congress lists parsed for cpro={cpro}")
        present: set[str] = {_classify_boe_2026_candidacy(lbl) for lbl in entries.values()}
        mask[cpro] = {p: 1 if p in present else 0 for p in present}
    return mask


# Canonical 2026 party definitions:
# (id, name, short_name, color, national_group, bloc, aliases)
CANONICAL_PARTY_SPECS: Final[
    tuple[tuple[str, str, str, str, str, str, tuple[str, ...]], ...]
] = (
    (
        "PP",
        "Partido Popular",
        "PP",
        "#0055A7",
        "PP",
        "right",
        ("PP", "PP-PAR", "PP-FORO", "Partit Popular"),
    ),
    (
        "PSOE",
        "Partido Socialista Obrero Español",
        "PSOE",
        "#E30613",
        "PSOE",
        "left",
        ("PSOE", "PSC", "PSE-EE", "PSdeG-PSOE", "PSIB-PSOE", "PSN-PSOE", "PSC–PSOE"),
    ),
    ("VOX", "VOX", "Vox", "#63BE21", "VOX", "right", ("VOX", "Vox")),
    (
        "FA",
        "Frente Amplio",
        "F. Amplio",
        "#7B2D8E",
        "IZQUIERDAS",
        "left",
        ("FA", "Frente Amplio", "Comuns", "CHA-FA", "Frente Amplio-Fronte Ampla"),
    ),
    (
        "PODEMOS",
        "Podemos",
        "Podemos",
        "#612D62",
        "IZQUIERDAS",
        "left",
        ("PODEMOS", "Podemos", "Podem"),
    ),
    (
        "COMPROMIS_FV",
        "Compromís: Front Valencià",
        "Compromís-FV",
        "#E86C1E",
        "IZQUIERDAS",
        "left",
        ("FV", "COMPROMIS_FV", "Compromís", "Compromís-Front Valencià"),
    ),
    (
        "UPC",
        "Unidad del Pueblo Canario",
        "UPC",
        "#2A9D8F",
        "IZQUIERDAS",
        "left",
        ("UPC", "Unidad del Pueblo Canario"),
    ),
    (
        "ARA_MES",
        "Ara Més-Front Ampli",
        "Ara Més-FA",
        "#9B5DE5",
        "IZQUIERDAS",
        "left",
        ("ARA_MES", "Ara Més", "Ara Més-Front Ampli"),
    ),
    (
        "ADELANTE",
        "Adelante Andalucía",
        "Adelante",
        "#2E9E5B",
        "IZQUIERDAS",
        "left",
        ("ADELANTE", "Adelante Andalucía", "Adelante Andalucía (2021)", "AA"),
    ),
    (
        "SALF",
        "Se Acabó La Fiesta",
        "SALF",
        "#6B4226",
        "OTROS",
        "right",
        ("SALF", "Se Acabó la Fiesta", "Se Acabó La Fiesta"),
    ),
    (
        "ERC",
        "Esquerra Republicana de Catalunya",
        "ERC",
        "#FFB232",
        "OTROS",
        "left_regional",
        ("ERC", "ERC-CatSí", "ERC–Sobiranistes"),
    ),
    (
        "JUNTS",
        "Junts per Catalunya",
        "Junts",
        "#00C3B2",
        "OTROS",
        "right_regional",
        ("JUNTS", "Junts", "JxCat", "JUNTS UE"),
    ),
    (
        "EH_BILDU",
        "Euskal Herria Bildu",
        "EH Bildu",
        "#B5CF18",
        "OTROS",
        "left_regional",
        ("EH_BILDU", "EH Bildu", "Bildu"),
    ),
    ("PNV", "EAJ-PNV", "PNV", "#008000", "OTROS", "center_regional", ("PNV", "EAJ-PNV")),
    (
        "GEROA_BAI",
        "Geroa Bai",
        "Geroa Bai",
        "#F76524",
        "OTROS",
        "center_regional",
        ("GEROA_BAI", "Geroa Bai", "GBai"),
    ),
    ("BNG", "Bloque Nacionalista Galego", "BNG", "#76B3DD", "OTROS", "left_regional", ("BNG",)),
    (
        "CCA",
        "Coalición Canaria",
        "CCa",
        "#FFD700",
        "OTROS",
        "center_regional",
        ("CCA", "CCa", "CC", "CCa-1.ºCAN"),
    ),
    (
        "UPN_PP",
        "Unión del Pueblo Navarro - Partido Popular",
        "UPN-PP",
        "#1D4E89",
        "PP",
        "right",
        ("UPN_PP", "UPN-PP", "UPN–PP"),
    ),
    (
        "TERUEL_EXISTE",
        "Teruel Existe",
        "Teruel Existe",
        "#006D77",
        "OTROS",
        "local_regional",
        ("TERUEL_EXISTE", "Teruel Existe", "EXISTE", "Aragón Existe"),
    ),
    (
        "SORIA_YA",
        "Soria ¡Ya!",
        "Soria ¡Ya!",
        "#4B3F72",
        "OTROS",
        "local_regional",
        ("SORIA_YA", "Soria Ya", "Soria ¡Ya!", "SY"),
    ),
    (
        "PACMA",
        "Partido Animalista con el Medio Ambiente",
        "PACMA",
        "#8BBF3C",
        "OTROS",
        "other",
        ("PACMA",),
    ),
    (
        "FO",
        "Frente Obrero",
        "Frente Obrero",
        "#7A0019",
        "OTROS",
        "other",
        ("FO", "Frente Obrero"),
    ),
    ("ALHORA", "Alhora", "Alhora", "#4F6D7A", "OTROS", "right_regional", ("ALHORA", "Alhora")),
    (
        "AC",
        "Aliança Catalana",
        "Aliança.cat",
        "#1B365D",
        "OTROS",
        "right_regional",
        ("AC", "Aliança.cat", "Aliança Catalana", "Aliança"),
    ),
    (
        "CUP",
        "Candidatura d'Unitat Popular",
        "CUP",
        "#FFEA00",
        "OTROS",
        "left_regional",
        ("CUP", "CUP-PR", "CUP–DT"),
    ),
    ("UPL", "Unión del Pueblo Leonés", "UPL", "#B91C6B", "OTROS", "local_regional", ("UPL",)),
    (
        "OTROS",
        "Otras candidaturas",
        "Otros",
        "#7F8C8D",
        "OTROS",
        "other",
        ("OTROS", "Otros", "Otro partido"),
    ),
)


def build_parties_catalog(
    *,
    offline: bool = False,
    metadata_dir: Path | None = None,
    raw_cache_dir: Path | None = None,
    write_file: bool = True,
) -> PartiesCatalog:
    """Build and validate ``data/metadata/parties_2026.json`` from 3 Nov 2026 BOE."""
    out_path = (metadata_dir or get_metadata_dir()) / "parties_2026.json"
    try:
        cand_xml = ensure_cached_file(
            "boe_2026_candidaturas.xml",
            BOE_URLS["boe_2026_candidaturas.xml"],
            offline=offline,
            raw_cache_dir=raw_cache_dir,
        )
        corr_xml = ensure_cached_file(
            "boe_2026_cand_corr.xml",
            BOE_URLS["boe_2026_cand_corr.xml"],
            offline=offline,
            raw_cache_dir=raw_cache_dir,
        )
    except FileNotFoundError:
        if offline and out_path.is_file():
            return load_parties_catalog(out_path)
        raise
    mask_by_cpro = parse_boe_2026_candidacy_mask(cand_xml, corr_xml)

    specs: list[PartySpec] = []
    for pid, name, short, color, ngroup, bloc, aliases in CANONICAL_PARTY_SPECS:
        pmask = {cpro: int(mask_by_cpro[cpro].get(pid, 0)) for cpro in ALL_CPROS}
        specs.append(
            PartySpec(
                id=pid,
                name=name,
                short_name=short,
                color=color,
                national_group=ngroup,  # type: ignore[arg-type]
                bloc=bloc,
                boe_mask_2026=pmask,
                aliases=aliases,
            )
        )

    catalog = PartiesCatalog(
        parties=tuple(specs),
        election_date="2026-11-29",
        boe_source="BOE-A-2026-23103",
    )
    if write_file:
        out_path = (metadata_dir or get_metadata_dir()) / "parties_2026.json"
        catalog.save_json(out_path)
    return catalog


# ---------------------------------------------------------------------------
# Historical Congress Builders (2015-D, 2016-J, 2019-A, 2019-N, 2023-J)
# ---------------------------------------------------------------------------


def _map_dat_candidacy_to_canonical(
    election_id: str,
    cpro: str,
    siglas: str,
    denom: str,
    seats_won: int,
) -> str:
    """Map an Interior ``03*.DAT`` candidacy row onto canonical historical party IDs."""
    s = strip_accents(siglas).upper().strip()
    d = strip_accents(denom).upper().strip()
    comb = f"{s} || {d}"

    # Navarre joint lists match Wikipedia polling columns directly
    if cpro == "31":
        if election_id in ("2019-04", "2019-11") and ("NA+" in s or "NAVARRA SUMA" in d):
            return "NA_SUMA"
        if "UPN" in s or "UNION DEL PUEBLO NAVARRO" in d:
            return "PP" if election_id in ("2015-12", "2016-06") else "UPN"

    if (
        "PARTIDO POPULAR" in d
        or "PARTIT POPULAR" in d
        or s in {"PP", "PP-PAR", "PP-FORO", "UPN-PP"}
        or s.startswith("PP-")
    ):
        return "PP"
    if (
        "PARTIDO SOCIALISTA" in d
        or "PARTIT DELS SOCIALISTES" in d
        or "PARTIDO DOS SOCIALISTAS" in d
        or s in {
            "PSOE",
            "PSC-PSOE",
            "PSC",
            "PSDEG-PSOE",
            "PSE-EE (PSOE)",
            "PSE-EE (PSO",
            "PSE-EE",
            "PSOE-NCA",
        }
        or s.startswith("PSOE")
    ):
        return "PSOE"
    if s == "VOX" or d == "VOX":
        return "VOX"
    s_norm = re.sub(r"[´`’]", "'", s)
    if (
        s_norm in {"C'S", "CS", "C S"}
        or "CIUDADANOS-PARTIDO" in d
        or "CIUDADANOS - PARTIDO" in d
        or "CIUDADANOS, PARTIDO" in d
        or "CIUDADANOS PARTIDO" in d
        or ("CIUTADANS" in d and "CENTRO DEMOCRATICO" not in d)
        or d == "CIUDADANOS"
    ):
        return "CS"

    # Left / Unidas Podemos / IU / En Comú / En Marea / Compromís / Más País
    if election_id == "2015-12":
        if "PODEMOS" in d or "EN COMU PODEM" in d or "EN MAREA" in d or "ES EL MOMENT" in d:
            return "PODEMOS"
        if "UNIDAD POPULAR" in d or "IZQUIERDA UNIDA" in d or "UNITAT POPULAR" in d or "UPEC" in s:
            return "IU"
        if "UNION PROGRESO Y DEMOCRACIA" in d or s == "UPYD":
            return "UPYD"
    else:
        if "MAS PAIS" in d or s.startswith("MAS PAIS") or s.startswith("MÁS PAÍS"):
            return "MP"
        if (
            "MES COMPROMIS" in d
            or "MÉS COMPROMÍS" in d
            or "COMPROMIS 2019" in d
            or s.startswith("COMPROMIS")
            or s.startswith("MES COMPROM")
        ):
            return "COMPROMIS"
        if election_id == "2019-04" and s == "EN MAREA":
            return "EN_MAREA"
        if (
            "UNIDOS PODEMOS" in d
            or "UNIDAS PODEMOS" in d
            or "EN COMU PODEM" in d
            or "EN COMUN-UNIDAS" in d
            or "A LA VALENCIANA" in d
            or (election_id == "2016-06" and "EN MAREA" in d)
            or "ELKARREKIN PODEMOS" in d
            or s.startswith("PODEMOS")
            or s.startswith("ECP")
        ):
            return "UP"

    if "ESQUERRA REPUBLICANA" in d or s.startswith("ERC"):
        return "ERC"
    if (
        "JUNTS" in d
        or "CONVERGENCIA DEMOCRATICA DE CATALUNYA" in d
        or "DEMOCRACIA I LLIBERTAT" in d
        or s in {"CDC", "DL", "JXCAT-JUNTS", "JXCAT"}
    ):
        return "JUNTS"
    if "EUSKAL HERRIA BILDU" in d or "BILDU" in s:
        return "EH_BILDU"
    if "NACIONALISTA VASCO" in d or s == "EAJ-PNV":
        return "PNV"
    if "BLOQUE NACIONALISTA GALEGO" in d or "BNG" in s or s == "NOS":
        return "BNG"
    if s in {"NC", "NCA"} or ("NUEVA CANARIAS" in d and "COALICION CANARIA" not in d):
        return "NC"
    if (
        "COALICION CANARIA" in d
        or "NUEVA CANARIAS" in d
        or "CCA-PNC" in s
        or "NC-CCA" in s
    ):
        return "CCA"
    if "PARTIDO REGIONALISTA DE CANTABRIA" in d or s == "PRC":
        return "PRC"
    if "CANDIDATURA D'UNITAT POPULAR" in d or "CUP-PR" in s:
        return "CUP"
    if "TERUEL EXISTE" in comb:
        return "TERUEL_EXISTE"
    if "PLATAFORMA DEL PUEBLO SORIANO" in d or "PPSO" in s:
        return "SORIA_YA"
    if "CHUNTA ARAGONESISTA" in d or s == "CHA":
        return "CHA"
    if "UNION DEL PUEBLO LEONES" in d or s == "UPL":
        return "UPL"
    if "POR AVILA" in d or s == "XAV":
        return "XAV"
    if (
        s in {"MES", "ARA-MES-ESQ", "MES-ESQUERR"}
        or "VEUS PROGRESSISTES" in d
        or "MES ESQUERRA" in d
    ):
        return "MES"
    if "EL PI" in s or "PROPOSTA PER LES ILLES" in d:
        return "EL_PI"
    if "UNIO.CAT" in s or "UNIO DEMOCRATICA DE CATALUNYA" in d:
        return "UNIO"
    if "FRONT REPUB" in s or "POBLE LLIURE" in d:
        return "FRONT_REPUB"
    if "COALICION POR MELILLA" in d or s == "CPM":
        return "CPM"
    if "DIGNIDAD Y LA CIUDADANIA" in d or s == "MDYC":
        return "MDYC"
    if "PACMA" in s or "ANIMALISTA" in d:
        return "PACMA"
    if "GEROA BAI" in comb:
        return "GEROA_BAI"
    if "FRENTE OBRERO" in comb:
        return "FO"

    if seats_won > 0:
        raise SchemaValidationError(
            f"Unmapped seat-winning candidacy in {election_id} cpro={cpro}: {comb}"
        )
    return "OTROS"


def build_congress_from_dat(
    election_id: str,
    election_code: str,
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
    elections_dir: Path | None = None,
    write_file: bool = True,
) -> list[ConstituencyElectionResult]:
    """Build and verify a 52-district historical Congress CSV from Infoelectoral ``.DAT`` files."""
    snap_path = (elections_dir or get_elections_dir()) / (
        f"congress_{election_id.replace('-', '_')}.csv"
    )
    try:
        dat_dir = _ensure_infoelectoral_dir(
            election_code, offline=offline, raw_cache_dir=raw_cache_dir
        )
    except FileNotFoundError:
        if offline and snap_path.is_file():
            return load_congress_results(election_id, elections_dir=elections_dir)
        raise
    f03 = next(dat_dir.glob("03*.DAT"))
    f07 = next(dat_dir.glob("07*.DAT"))
    f08 = next(dat_dir.glob("08*.DAT"))

    cand_info: dict[str, tuple[str, str]] = {}
    for ln in f03.read_text(encoding="latin-1").splitlines():
        if len(ln) >= 65:
            cand_info[ln[8:14]] = (ln[14:64].strip(), ln[64:214].strip())

    prov_totals: dict[str, dict[str, Any]] = {}
    for ln in f07.read_text(encoding="latin-1").splitlines():
        if len(ln) >= 155 and ln[9:11] != "99" and ln[11:13] != "99" and ln[13:14] == "9":
            cpro = ln[11:13]
            prov_totals[cpro] = {
                "name": ln[14:64].strip(),
                "censo": int(ln[85:93]),
                "blancos": int(ln[125:133]),
                "nulos": int(ln[133:141]),
                "votos_cand": int(ln[141:149]),
                "seats": int(ln[149:155]),
            }

    party_votes_by_cpro: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    party_seats_by_cpro: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen_non_other: set[tuple[str, str]] = set()

    for ln in f08.read_text(encoding="latin-1").splitlines():
        if len(ln) >= 33 and ln[9:11] != "99" and ln[11:13] != "99" and ln[13:14] == "9":
            cpro = ln[11:13]
            cid = ln[14:20]
            votes = int(ln[20:28])
            seats = int(ln[28:33])
            siglas, denom = cand_info.get(cid, ("", ""))
            pid = _map_dat_candidacy_to_canonical(election_id, cpro, siglas, denom, seats)
            if pid != "OTROS":
                if (cpro, pid) in seen_non_other and votes > 0:
                    raise SchemaValidationError(
                        f"Collision on {(election_id, cpro, pid)} from cid={cid} ({siglas})"
                    )
                seen_non_other.add((cpro, pid))
            party_votes_by_cpro[cpro][pid] += votes
            party_seats_by_cpro[cpro][pid] += seats

    if set(prov_totals.keys()) != set(ALL_CPROS):
        raise SchemaValidationError(f"{election_id}: parsed {len(prov_totals)} provinces != 52")

    from elecciones.models.dhondt import allocate_dhondt_loreg

    results: list[ConstituencyElectionResult] = []
    seat_mismatches = 0
    for cpro in ALL_CPROS:
        pt = prov_totals[cpro]
        pv = dict(party_votes_by_cpro[cpro])
        ps = dict(party_seats_by_cpro[cpro])
        cand_sum = sum(pv.values())
        if cand_sum != pt["votos_cand"]:
            raise SchemaValidationError(
                f"{election_id} cpro={cpro}: party sum {cand_sum} != votos_cand {pt['votos_cand']}"
            )
        blancos = pt["blancos"]
        validos = blancos + cand_sum
        nulos = pt["nulos"]
        votantes = validos + nulos
        res = ConstituencyElectionResult(
            election_id=election_id,
            cpro=cpro,
            constituency=pt["name"],
            seats=pt["seats"],
            censo=pt["censo"],
            votantes=votantes,
            nulos=nulos,
            validos=validos,
            blancos=blancos,
            party_votes=pv,
            party_seats=ps,
        )
        # Verify exact LOREG Art. 163 D'Hondt reproduction
        alloc = allocate_dhondt_loreg(
            res.to_allocation_input(), raise_on_exhaustion=True, include_quotients=False
        )
        for p in set(ps) | set(alloc.seats):
            if ps.get(p, 0) != alloc.seats.get(p, 0):
                seat_mismatches += 1
        results.append(res)

    if seat_mismatches != 0:
        raise SchemaValidationError(
            f"{election_id}: LOREG verification found {seat_mismatches} seat discrepancies"
        )

    if write_file:
        _write_congress_csv(
            results,
            (elections_dir or get_elections_dir())
            / f"congress_{election_id.replace('-', '_')}.csv",
        )
    return results


def _map_boe_2023_party(header_label: str) -> str:
    """Map BOE-A-2023-18907 table header label onto canonical 2023 Congress party ID."""
    up = strip_accents(header_label).upper().strip()
    if "PARTIDO POPULAR (PP)" in up:
        return "PP"
    if (
        "PARTIDO SOCIALISTA OBRERO ESPANOL (PSOE)" in up
        or "PARTIT DELS SOCIALISTES DE CATALUNYA" in up
    ):
        return "PSOE"
    if "VOX" in up:
        return "VOX"
    if "SUMAR" in up:
        return "SUMAR"
    if "ESQUERRA REPUBLICANA DE CATALUNYA" in up:
        return "ERC"
    if "JUNTS" in up:
        return "JUNTS"
    if "EUSKAL HERRIA BILDU" in up:
        return "EH_BILDU"
    if "EUZKO ALDERDI JELTZALEA" in up or "EAJ-PNV" in up:
        return "PNV"
    if "BLOQUE NACIONALISTA GALEGO" in up or "(BNG)" in up:
        return "BNG"
    if "COALICION CANARIA" in up or "(CCA)" in up:
        return "CCA"
    if "UNION DEL PUEBLO NAVARRO" in up or "(UPN)" in up:
        return "UPN"
    if "NUEVA CANARIAS" in up or "(NC-BC)" in up:
        return "NC"
    if "CANDIDATURA D'UNITAT POPULAR" in up or "(CUP-PR)" in up:
        return "CUP"
    if "PARTIDO ANIMALISTA" in up or "(PACMA)" in up:
        return "PACMA"
    if "FRENTE OBRERO" in up or "(FO)" in up:
        return "FO"
    if "COALICION EXISTE" in up or "(EXISTE)" in up:
        return "TERUEL_EXISTE"
    if "SORIA" in up and "YA" in up:
        return "SORIA_YA"
    if "ADELANTE ANDALUCIA" in up:
        return "ADELANTE"
    if "GEROA BAI" in up:
        return "GEROA_BAI"
    if "UNION DEL PUEBLO LEONES" in up or "(UPL)" in up:
        return "UPL"
    if "JAEN MERECE MAS" in up or "(JM+)" in up:
        return "JM"
    if "POR AVILA" in up or "(XAV)" in up:
        return "XAV"
    if "PARTIT DEMOCRATA EUROPEU CATALA" in up or "(PDECAT" in up:
        return "PDECAT"
    return "OTROS"


def build_congress_2023_from_boe(
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
    elections_dir: Path | None = None,
    write_file: bool = True,
) -> list[ConstituencyElectionResult]:
    """Build and verify ``congress_2023_07.csv`` from official BOE XML + Cuadro I PDF."""
    snap_path = (elections_dir or get_elections_dir()) / "congress_2023_07.csv"
    try:
        xml_path = ensure_cached_file(
            "boe_2023_07.xml",
            BOE_URLS["boe_2023_07.xml"],
            offline=offline,
            raw_cache_dir=raw_cache_dir,
        )
        pdf_path = ensure_cached_file(
            "boe_2023_07.pdf",
            BOE_URLS["boe_2023_07.pdf"],
            offline=offline,
            raw_cache_dir=raw_cache_dir,
        )
    except FileNotFoundError:
        if offline and snap_path.is_file():
            return load_congress_results("2023-07", elections_dir=elections_dir)
        raise
    cuadro = parse_boe_2023_cuadro_uno(pdf_path)

    root = ET.parse(xml_path).getroot()
    party_votes_by_cpro: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    party_seats_by_cpro: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seats_total_by_cpro: dict[str, int] = {}

    for tb in root.findall(".//table"):
        thead_trs = tb.findall(".//thead/tr")
        if len(thead_trs) < 2:
            continue
        r0 = thead_trs[0].findall("th")
        has_total_escanos = len(r0) > 1 and "Total escaños" in "".join(r0[1].itertext())
        party_cols: list[tuple[str, bool]] = []
        start_idx = 2 if has_total_escanos else 1
        for th in r0[start_idx:]:
            lbl = re.sub(r"\s+", " ", "".join(th.itertext()).replace("\xa0", " ").strip())
            cs = int(th.get("colspan", "1"))
            party_cols.append((_map_boe_2023_party(lbl), cs == 2))

        for tr in tb.findall(".//tbody/tr"):
            cells = [
                re.sub(r"\s+", " ", "".join(c.itertext()).strip())
                for c in tr.findall("*")
            ]
            if not cells:
                continue
            prov_label = cells[0].rstrip(".")
            if "total" in strip_accents(prov_label).lower():
                continue
            cpro = resolve_cpro_from_name(prov_label)
            idx = 1
            if has_total_escanos:
                seats_total_by_cpro[cpro] = _parse_es_int(cells[idx])
                idx += 1
            for pid, has_seats in party_cols:
                v = _parse_es_int(cells[idx])
                idx += 1
                s = _parse_es_int(cells[idx]) if has_seats else 0
                if has_seats:
                    idx += 1
                party_votes_by_cpro[cpro][pid] += v
                party_seats_by_cpro[cpro][pid] += s

    if set(seats_total_by_cpro.keys()) != set(ALL_CPROS):
        raise SchemaValidationError(f"2023-07 BOE tables covered {len(seats_total_by_cpro)} != 52")

    from elecciones.models.dhondt import allocate_dhondt_loreg

    spec_by_cpro = {row[0]: row for row in PROVINCE_SPECS}
    results: list[ConstituencyElectionResult] = []
    seat_mismatches = 0
    for cpro in ALL_CPROS:
        cu = cuadro[cpro]
        pv = dict(party_votes_by_cpro[cpro])
        ps = dict(party_seats_by_cpro[cpro])
        cand_sum = sum(pv.values())
        if cand_sum != cu["candidaturas"]:
            raise SchemaValidationError(
                f"2023-07 cpro={cpro}: XML sum {cand_sum} != "
                f"Cuadro I candidaturas {cu['candidaturas']}"
            )
        res = ConstituencyElectionResult(
            election_id="2023-07",
            cpro=cpro,
            constituency=spec_by_cpro[cpro][1],
            seats=seats_total_by_cpro[cpro],
            censo=cu["censo"],
            votantes=cu["votantes"],
            nulos=cu["nulos"],
            validos=cu["validos"],
            blancos=cu["blancos"],
            party_votes=pv,
            party_seats=ps,
        )
        alloc = allocate_dhondt_loreg(
            res.to_allocation_input(), raise_on_exhaustion=True, include_quotients=False
        )
        for p in set(ps) | set(alloc.seats):
            if ps.get(p, 0) != alloc.seats.get(p, 0):
                seat_mismatches += 1
        results.append(res)

    if seat_mismatches != 0:
        raise SchemaValidationError(
            f"2023-07 BOE LOREG verification found {seat_mismatches} seat discrepancies"
        )

    if write_file:
        _write_congress_csv(
            results, (elections_dir or get_elections_dir()) / "congress_2023_07.csv"
        )
    return results


def _write_congress_csv(results: Sequence[ConstituencyElectionResult], out_path: Path) -> None:
    """Write deterministic wide CSV with all parties appearing in ``results``."""
    party_totals: dict[str, int] = defaultdict(int)
    for r in results:
        for p, v in r.party_votes.items():
            party_totals[p] += v
    party_order = [
        p
        for p, _ in sorted(
            party_totals.items(), key=lambda kv: (kv[0] == "OTROS", -kv[1], kv[0])
        )
    ]
    fieldnames = [
        "election_id",
        "cpro",
        "constituency",
        "seats",
        "censo",
        "votantes",
        "nulos",
        "validos",
        "blancos",
        *[f"votes_{p}" for p in party_order],
        *[f"seats_{p}" for p in party_order],
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_csv_row(party_order))


# ---------------------------------------------------------------------------
# 2024 European Parliament Builder (BOE-A-2024-13092 + Errata BOE-A-2024-15344)
# ---------------------------------------------------------------------------


def _map_ep_2024_party(acronym_or_title: str) -> str:
    up = strip_accents(acronym_or_title).upper().strip()
    if up == "PP" or "PARTIDO POPULAR" in up:
        return "PP"
    if up == "PSOE" or "SOCIALISTA OBRERO ESPANOL" in up:
        return "PSOE"
    if up == "VOX":
        return "VOX"
    if "AHORA REPUBLICAS" in up:
        return "AHORA_REPUBLICAS"
    if "SUMAR" in up:
        return "SUMAR"
    if "SE ACABO LA FIESTA" in up or up == "SALF":
        return "SALF"
    if up == "PODEMOS":
        return "PODEMOS"
    if "JUNTS" in up:
        return "JUNTS"
    if "CEUS" in up or "EUROPA SOLIDARIA" in up:
        return "CEUS"
    if "PACMA" in up or "ANIMALISTA" in up:
        return "PACMA"
    if up == "CS" or "CIUDADANOS" in up:
        return "CS"
    if up == "FO" or "FRENTE OBRERO" in up:
        return "FO"
    if "EXISTE" in up:
        return "EXISTE"
    return "OTROS"


def build_europarl_2024(
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
    elections_dir: Path | None = None,
    write_file: bool = True,
) -> list[dict[str, Any]]:
    """Build ``data/elections/europarl_2024_06.csv`` from BOE XML + errata verification."""
    snap_path = (elections_dir or get_elections_dir()) / "europarl_2024_06.csv"
    try:
        ep_xml = ensure_cached_file(
            "boe_2024_ep.xml",
            BOE_URLS["boe_2024_ep.xml"],
            offline=offline,
            raw_cache_dir=raw_cache_dir,
        )
        # Also ensure errata file is fetched/cached and audited against provincial sums
        err_xml = ensure_cached_file(
            "boe_2024_ep_errata.xml",
            BOE_URLS["boe_2024_ep_errata.xml"],
            offline=offline,
            raw_cache_dir=raw_cache_dir,
        )
    except FileNotFoundError:
        if offline and snap_path.is_file():
            int_cols = {"censo", "votantes", "nulos", "validos", "blancos"}
            loaded: list[dict[str, Any]] = []
            with snap_path.open("r", encoding="utf-8", newline="") as fh:
                for raw in csv.DictReader(fh):
                    row_d: dict[str, Any] = {}
                    for k, v in raw.items():
                        if k in int_cols or k.startswith("votes_"):
                            row_d[k] = int(v or 0)
                        else:
                            row_d[k] = v
                    loaded.append(row_d)
            if len(loaded) != 52:
                raise SchemaValidationError(
                    f"Expected 52 rows in {snap_path}, got {len(loaded)}"
                ) from None
            return loaded
        raise
    _ = ET.parse(err_xml)

    root = ET.parse(ep_xml).getroot()
    tables = root.findall(".//table")
    if len(tables) < 3:
        raise SchemaValidationError(f"Expected >= 3 tables in {ep_xml}, got {len(tables)}")

    cuadro: dict[str, dict[str, int]] = {}
    for tr in tables[0].findall(".//tr")[1:]:
        cells = [re.sub(r"\s+", " ", "".join(c.itertext()).strip()) for c in tr]
        if len(cells) < 7:
            continue
        prov = re.sub(r"\.\d+$", "", cells[0].rstrip(".")).strip()
        if "total" in strip_accents(prov).lower():
            continue
        cpro = resolve_cpro_from_name(prov)
        cuadro[cpro] = {
            "censo": _parse_es_int(cells[1]),
            "votantes": _parse_es_int(cells[2]),
            "validos": _parse_es_int(cells[3]),
            "candidaturas": _parse_es_int(cells[4]),
            "blancos": _parse_es_int(cells[5]),
            "nulos": _parse_es_int(cells[6]),
        }

    votes_by_cpro: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tb in tables[1:]:
        thead_trs = tb.findall(".//thead/tr")
        if not thead_trs:
            thead_trs = tb.findall(".//tr")[:3]
        if not thead_trs:
            continue
        ths0 = thead_trs[0].findall("th")
        ths1 = thead_trs[1].findall("th") if len(thead_trs) > 1 else []
        r0 = [re.sub(r"\s+", " ", "".join(c.itertext()).strip()) for c in ths0]
        r1 = [re.sub(r"\s+", " ", "".join(c.itertext()).strip()) for c in ths1]
        has_total_esc = len(r0) > 1 and "total escanos" in strip_accents(r0[1]).lower()
        start_idx = 2 if has_total_esc else 1
        col_specs: list[tuple[str, int]] = []
        for pos, idx in enumerate(range(start_idx, len(ths0))):
            hint = r1[pos] if pos < len(r1) and r1[pos] else r0[idx]
            cs = int(ths0[idx].attrib.get("colspan", "1"))
            col_specs.append((_map_ep_2024_party(hint), cs))

        tbody_rows = tb.findall(".//tbody/tr")
        if not tbody_rows:
            tbody_rows = tb.findall(".//tr")[3:]
        for tr in tbody_rows:
            cells = [re.sub(r"\s+", " ", "".join(c.itertext()).strip()) for c in tr]
            if not cells or not cells[0]:
                continue
            prov_raw = re.sub(r"\.\d+$", "", cells[0].rstrip(".")).strip()
            low_p = strip_accents(prov_raw).lower()
            if "total" in low_p or "porcentaje" in low_p or "votos" in low_p:
                continue
            cpro = resolve_cpro_from_name(prov_raw)
            val_idx = 2 if has_total_esc else 1
            for pid, cs in col_specs:
                if val_idx < len(cells):
                    votes_by_cpro[cpro][pid] += _parse_es_int(cells[val_idx])
                val_idx += cs

    if set(cuadro.keys()) != set(ALL_CPROS):
        raise SchemaValidationError(f"2024-06 EP parsed {len(cuadro)} != 52 provinces")

    # Verify BOE-A-2024-15344 provincial totals (PSOE=5,291,102, JUNTS=442,140)
    psoe_sum = sum(votes_by_cpro[c]["PSOE"] for c in ALL_CPROS)
    junts_sum = sum(votes_by_cpro[c]["JUNTS"] for c in ALL_CPROS)
    if psoe_sum != 5_291_102 or junts_sum != 442_140:
        raise SchemaValidationError(
            f"2024-06 EP errata verification failed: PSOE={psoe_sum}, JUNTS={junts_sum}"
        )

    spec_by_cpro = {row[0]: row for row in PROVINCE_SPECS}
    ep_parties = [
        "PP",
        "PSOE",
        "VOX",
        "AHORA_REPUBLICAS",
        "SUMAR",
        "SALF",
        "PODEMOS",
        "JUNTS",
        "CEUS",
        "PACMA",
        "CS",
        "FO",
        "EXISTE",
        "OTROS",
    ]
    rows: list[dict[str, Any]] = []
    for cpro in ALL_CPROS:
        cu = cuadro[cpro]
        pv = votes_by_cpro[cpro]
        cand_sum = sum(pv.values())
        if cand_sum != cu["candidaturas"]:
            raise SchemaValidationError(
                f"2024-06 EP cpro={cpro}: sum {cand_sum} != candidaturas {cu['candidaturas']}"
            )
        row: dict[str, Any] = {
            "election_id": "2024-06",
            "cpro": cpro,
            "constituency": spec_by_cpro[cpro][1],
            "censo": cu["censo"],
            "votantes": cu["votantes"],
            "nulos": cu["nulos"],
            "validos": cu["validos"],
            "blancos": cu["blancos"],
        }
        for p in ep_parties:
            row[f"votes_{p}"] = pv.get(p, 0)
        rows.append(row)

    if write_file:
        out_path = (elections_dir or get_elections_dir()) / "europarl_2024_06.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# 2024-2026 Regional Elections Builder (Wikipedia MediaWiki JSON)
# ---------------------------------------------------------------------------

_REGIONAL_META: Final[dict[str, tuple[str, str, str]]] = {
    "2024 Galician regional election": ("2024-02-18", "12", "Galicia"),
    "2024 Basque regional election": ("2024-04-21", "16", "País Vasco"),
    "2024 Catalan regional election": ("2024-05-12", "09", "Cataluña"),
    "2025 Extremaduran regional election": ("2025-12-21", "11", "Extremadura"),
    "2026 Aragonese regional election": ("2026-02-08", "02", "Aragón"),
    "2026 Castilian-Leonese regional election": ("2026-03-15", "07", "Castilla y León"),
    "2026 Andalusian regional election": ("2026-05-17", "01", "Andalucía"),
}


def _map_regional_party_token(raw: str, title: str) -> str | None:
    up = strip_accents(raw).upper().strip()
    if not up or any(
        k in up
        for k in (
            "INVALID",
            "BLANK",
            "TOTAL",
            "REGISTERED",
            "TURNOUT",
            "SOURCE",
            "VALID VOTES",
            "ABSTENTION",
            "FOOTNOTE",
            "{{HIDDEN",
        )
    ):
        return None
    if "PP" in re.findall(r"[A-Z]+", up) or "PEOPLE'S PARTY" in up:
        return "PP"
    if (
        "PSOE" in up
        or "PSC" in up
        or "PSDEG" in up
        or "PSE" in up
        or "SOCIALIST" in up
    ):
        return "PSOE"
    if "VOX" in up:
        return "VOX"
    if "SE ACABO" in up or "THE PARTY IS OVER" in up or "SALF" in up:
        return "SALF"
    if "ALIANCA" in up or "CATALAN ALLIANCE" in up or "ALIANZA CATALANA" in up:
        return "AC"
    if (
        "COMUNS" in up
        or "POR ANDALUCIA" in up
        or "FOR ANDALUSIA" in up
        or "PORA" in up
        or "UNIDAS POR EXTREMADURA" in up
        or "UNITED FOR EXTREMADURA" in up
        or "UXE" in up
    ):
        return "FA"
    if "CHUNTA ARAGONESISTA" in up or up == "CHA" or "CHA" in re.findall(r"[A-Z]+", up):
        return "FA"
    if ("UNITED LEFT" in up or "IU" in re.findall(r"[A-Z]+", up)) and "PODEMOS" not in up:
        return "FA"
    if "SUMAR" in up:
        return "FA" if "2026" in title or "2025" in title else "SUMAR"
    if "WE CAN" in up or "PODEMOS" in up or "PODEM" in up:
        return "PODEMOS"
    if "ADELANTE" in up or up == "AA":
        return "ADELANTE"
    if "JUNTS" in up or "TOGETHER FOR CATALONIA" in up:
        return "JUNTS"
    if "ERC" in re.findall(r"[A-Z]+", up) or "REPUBLICAN LEFT OF CATALONIA" in up:
        return "ERC"
    if "CUP" in re.findall(r"[A-Z]+", up) or "POPULAR UNITY CANDIDACY" in up:
        return "CUP"
    if "BILDU" in up:
        return "EH_BILDU"
    if "PNV" in up or "BASQUE NATIONALIST" in up:
        return "PNV"
    if "BNG" in up or "GALICIAN NATIONALIST" in up:
        return "BNG"
    if (
        "TERUEL EXISTE" in up
        or "ARAGON EXISTE" in up
        or "EXISTS COALITION" in up
        or up in {"EXISTE", "EXISTS"}
    ):
        return "TERUEL_EXISTE"
    if ("SORIA" in up and ("YA" in up or "NOW" in up)) or up == "SY" or "(SY)" in up:
        return "SORIA_YA"
    if up == "UPL" or "LEONESE PEOPLE" in up:
        return "UPL"
    if "OURENSAN" in up or up == "DO":
        return "DO"
    if "FOR AVILA" in up or up == "XAV":
        return "XAV"
    if "ARAGONESE PARTY" in up or up == "PAR":
        return "PAR"
    if "PACMA" in up or "ANIMALIST" in up:
        return "PACMA"
    return "OTROS"


def _wikitext_cell_value(raw_cell: str) -> str:
    """Split optional wikitext cell attributes before ``|`` at bracket depth 0, then strip."""
    depth_curly = 0
    depth_sq = 0
    split_pos = -1
    i = 0
    n = len(raw_cell)
    while i < n:
        ch = raw_cell[i]
        two = raw_cell[i : i + 2]
        if two == "{{":
            depth_curly += 1
            i += 2
            continue
        if two == "}}" and depth_curly > 0:
            depth_curly -= 1
            i += 2
            continue
        if two == "[[":
            depth_sq += 1
            i += 2
            continue
        if two == "]]" and depth_sq > 0:
            depth_sq -= 1
            i += 2
            continue
        if ch == "|" and depth_curly == 0 and depth_sq == 0:
            split_pos = i
            break
        i += 1
    body = raw_cell[split_pos + 1 :] if split_pos >= 0 else raw_cell
    return _strip_wiki_markup(body)


def _strip_wiki_markup(cell: str) -> str:
    txt = re.sub(r"<ref[^>]*>.*?</ref>", "", cell, flags=re.S)
    txt = re.sub(r"<sup[^>]*>.*?</sup>", "", txt, flags=re.S)
    txt = re.sub(r"'''?", "", txt)
    txt = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", txt)
    txt = re.sub(r"\[\[([^\]]+)\]\]", r"\1", txt)
    txt = re.sub(r"\{\{[^}]+\}\}", "", txt)
    return re.sub(r"\s+", " ", txt).strip()


def _parse_wikitext_regional_constituencies(
    title: str, rev: str, election_date: str, codauto: str
) -> list[dict[str, Any]]:
    m_dist = re.search(r"===Distribution by constituency===(.*?)(?=\n==[^=]|\Z)", rev, re.S)
    if not m_dist:
        raise SchemaValidationError(f"Missing Distribution by constituency in {title}")
    block = m_dist.group(1)
    raw_rows = re.split(r"\n\|-\s*", block)
    header_parties: list[str] = []
    data_cells_by_row: list[list[tuple[int, int, str]]] = []

    for r_idx, raw_r in enumerate(raw_rows):
        lines = [
            ln.strip()
            for ln in raw_r.splitlines()
            if ln.strip()
            and not ln.strip().startswith("{|")
            and not ln.strip().startswith("|}")
        ]
        if not lines:
            continue
        if r_idx == 0 or (lines[0].startswith("!") and not header_parties):
            for ln in lines:
                if 'colspan="2"' in ln and "background:" not in ln:
                    label = _wikitext_cell_value(ln.lstrip("!"))
                    if label:
                        pid = _map_regional_party_token(label, title) or "OTROS"
                        header_parties.append(pid)
            continue
        if lines[0].startswith("!"):
            continue

        cells: list[tuple[int, int, str]] = []
        for ln in lines:
            if not ln.startswith("|"):
                continue
            parts = ln[1:].split("||")
            for pt in parts:
                cs_m = re.search(r'colspan="(\d+)"', pt)
                rs_m = re.search(r'rowspan="(\d+)"', pt)
                cs = int(cs_m.group(1)) if cs_m else 1
                rs = int(rs_m.group(1)) if rs_m else 1
                cells.append((cs, rs, _wikitext_cell_value(pt)))
        if cells:
            data_cells_by_row.append(cells)

    # Expand 2D occupancy grid across rowspan & colspan
    grid: dict[tuple[int, int], str] = {}
    for dr, cells in enumerate(data_cells_by_row):
        col = 0
        for cs, rs, txt in cells:
            while (dr, col) in grid:
                col += 1
            for r_off in range(rs):
                for c_off in range(cs):
                    grid[(dr + r_off, col + c_off)] = txt
            col += cs

    out_rows: list[dict[str, Any]] = []
    for dr in range(len(data_cells_by_row)):
        label0 = grid.get((dr, 0), "").strip()
        if not label0 or any(k in strip_accents(label0).lower() for k in ("total", "source")):
            continue
        cpro = resolve_cpro_from_name(label0)
        shares: dict[str, float] = {}
        seats: dict[str, int] = {}
        for idx, pid in enumerate(header_parties):
            pct_txt = grid.get((dr, 1 + 2 * idx), "")
            seat_txt = grid.get((dr, 2 + 2 * idx), "")
            m_pct = re.search(r"(\d+(?:\.\d+)?)", pct_txt)
            if m_pct:
                shares[pid] = shares.get(pid, 0.0) + float(m_pct.group(1))
            m_s = re.search(r"(\d+)", seat_txt)
            if m_s:
                seats[pid] = seats.get(pid, 0) + int(m_s.group(1))
        if shares:
            out_rows.append(
                {
                    "election_id": f"reg-{codauto}-{election_date[:4]}",
                    "election_date": election_date,
                    "scope_type": "province",
                    "scope_code": cpro,
                    "scope_name": label0,
                    "shares": shares,
                    "seats": seats,
                }
            )
    return out_rows


def _parse_wikitext_regional_overall(
    title: str, rev: str, election_date: str, codauto: str, ccaa_name: str
) -> dict[str, Any] | None:
    m_ov = re.search(r"===Overall===(.*?)(?=\n===|\Z)", rev, re.S)
    if not m_ov:
        return None
    raw_rows = re.split(r"\n\|-\s*", m_ov.group(1))
    shares: dict[str, float] = {}
    seats: dict[str, int] = {}
    for raw_r in raw_rows:
        if 'align="left"' not in raw_r and "text-align:left" not in raw_r:
            continue
        lines = [
            ln.strip()
            for ln in raw_r.splitlines()
            if ln.strip().startswith("|") and not ln.strip().startswith("|}")
        ]
        if len(lines) < 2:
            continue
        cell_texts: list[str] = []
        party_txt = ""
        for ln in lines:
            for pt in ln[1:].split("||"):
                val = _wikitext_cell_value(pt)
                if not party_txt and (
                    'align="left"' in pt or "text-align:left" in pt or "[[" in pt
                ):
                    party_txt = val
                elif val:
                    cell_texts.append(val)
        if not party_txt:
            continue
        pid = _map_regional_party_token(party_txt, title)
        if pid is None:
            continue
        # First strictly numeric non-negative cell containing '.' in [0, 100] is vote %
        pct_val: float | None = None
        int_vals: list[int] = []
        for txt in cell_texts:
            if re.match(r"^\d+\.\d+$", txt):
                fv = float(txt)
                if 0.0 <= fv <= 100.0 and pct_val is None:
                    pct_val = fv
            elif re.match(r"^\d{1,3}$", txt):
                int_vals.append(int(txt))
        if pct_val is not None:
            shares[pid] = shares.get(pid, 0.0) + pct_val
        if int_vals:
            seats[pid] = seats.get(pid, 0) + int_vals[0]

    if not shares:
        return None
    return {
        "election_id": f"reg-{codauto}-{election_date[:4]}",
        "election_date": election_date,
        "scope_type": "ccaa",
        "scope_code": codauto,
        "scope_name": ccaa_name,
        "shares": shares,
        "seats": seats,
    }


def build_regional_2024_2026(
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
    elections_dir: Path | None = None,
    write_file: bool = True,
) -> list[dict[str, Any]]:
    """Build ``data/elections/regional_2024_2026.csv`` for all 7 2024–2026 CCAA cycles."""
    cache_dir = raw_cache_dir or get_raw_cache_dir()
    cache_file = cache_dir / "wikipedia_regional_elections.json"
    snap_path = (elections_dir or get_elections_dir()) / "regional_2024_2026.csv"
    if not (cache_file.is_file() and cache_file.stat().st_size > 0):
        if offline:
            if snap_path.is_file():
                with snap_path.open("r", encoding="utf-8", newline="") as fh:
                    rows = list(csv.DictReader(fh))
                if len(rows) < 35:
                    raise SchemaValidationError(
                        f"Expected >= 35 rows in {snap_path}, got {len(rows)}"
                    )
                return rows
            raise FileNotFoundError(f"Missing {cache_file} in offline mode.")
        titles_q = urllib.parse.quote("|".join(WIKI_REGIONAL_TITLES))
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&prop=revisions"
            f"&titles={titles_q}&rvslots=main&rvprop=content&formatversion=2&format=json"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        with make_http_client(timeout=45.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            cache_file.write_bytes(resp.content)

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    pages = payload["query"]["pages"]
    all_records: list[dict[str, Any]] = []
    all_parties: set[str] = set()

    for pdata in pages:
        title = pdata["title"]
        if title not in _REGIONAL_META:
            continue
        edate, codauto, ccaa_name = _REGIONAL_META[title]
        rev = pdata["revisions"][0]["slots"]["main"]["content"]
        ov = _parse_wikitext_regional_overall(title, rev, edate, codauto, ccaa_name)
        if ov is not None:
            all_records.append(ov)
            all_parties.update(ov["shares"].keys())
        prov_rows = _parse_wikitext_regional_constituencies(title, rev, edate, codauto)
        for pr in prov_rows:
            all_records.append(pr)
            all_parties.update(pr["shares"].keys())

    if not all_records:
        raise SchemaValidationError("Parsed 0 regional election records from Wikipedia JSON")

    party_order = sorted(
        all_parties,
        key=lambda p: (
            p not in {"PP", "PSOE", "VOX", "FA", "PODEMOS", "SUMAR", "SALF"},
            p == "OTROS",
            p,
        ),
    )
    all_records.sort(key=lambda r: (r["election_date"], r["scope_type"], r["scope_code"]))
    flat_rows: list[dict[str, Any]] = []
    for r in all_records:
        out: dict[str, Any] = {
            "election_id": r["election_id"],
            "election_date": r["election_date"],
            "scope_type": r["scope_type"],
            "scope_code": r["scope_code"],
            "scope_name": r["scope_name"],
        }
        for p in party_order:
            val = r["shares"].get(p)
            out[f"pct_{p}"] = f"{val:.2f}" if val is not None else ""
        flat_rows.append(out)

    if write_file:
        out_path = (elections_dir or get_elections_dir()) / "regional_2024_2026.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(flat_rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(flat_rows)
    return flat_rows


def build_all_elections(
    *,
    offline: bool = False,
    raw_cache_dir: Path | None = None,
    elections_dir: Path | None = None,
) -> dict[str, int]:
    """Build and verify all 7 election snapshots; returns row count per file key."""
    counts: dict[str, int] = {}
    for eid, code in (
        ("2015-12", "02201512"),
        ("2016-06", "02201606"),
        ("2019-04", "02201904"),
        ("2019-11", "02201911"),
    ):
        res = build_congress_from_dat(
            eid,
            code,
            offline=offline,
            raw_cache_dir=raw_cache_dir,
            elections_dir=elections_dir,
        )
        counts[f"congress_{eid}"] = len(res)

    res_23 = build_congress_2023_from_boe(
        offline=offline,
        raw_cache_dir=raw_cache_dir,
        elections_dir=elections_dir,
    )
    counts["congress_2023-07"] = len(res_23)

    ep = build_europarl_2024(
        offline=offline,
        raw_cache_dir=raw_cache_dir,
        elections_dir=elections_dir,
    )
    counts["europarl_2024-06"] = len(ep)

    reg = build_regional_2024_2026(
        offline=offline,
        raw_cache_dir=raw_cache_dir,
        elections_dir=elections_dir,
    )
    counts["regional_2024_2026"] = len(reg)
    return counts


def load_congress_results(
    election_id: str,
    elections_dir: Path | None = None,
) -> list[ConstituencyElectionResult]:
    """Load and schema-validate a committed ``congress_*.csv`` snapshot."""
    target = (elections_dir or get_elections_dir()) / (
        f"congress_{election_id.replace('-', '_')}.csv"
    )
    if not target.is_file():
        raise FileNotFoundError(f"Missing Congress snapshot: {target}")
    with target.open("r", encoding="utf-8", newline="") as fh:
        return [ConstituencyElectionResult.from_csv_row(row) for row in csv.DictReader(fh)]
