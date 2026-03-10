"""
OSM Population Distribution Plot
Fetches city, town, and village nodes from Overpass API in a single
request and plots their population distributions as a strip/jitter chart.
Also writes a GeoJSON file with per-feature misclassification analysis using
a hybrid of log-normal distribution likelihood and cross-type percentile ranking.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
from countries import Country, countries

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PLACE_TYPES = ["city", "town", "village"]

COLORS = {
    "city": "#e15759",
    "town": "#4e79a7",
    "village": "#59a14f",
}

# Descending order of expected population size — used for adjacent-type lookups.
TYPE_ORDER = ["city", "town", "village"]

PlaceRecord = dict  # {"name": str, "population": int, "lat": float, "lon": float, ...}
PlaceData = dict[str, list[PlaceRecord]]

COUNTRIES_BY_CODE: dict[str, Country] = {c.code: c for c in countries}


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------
def build_query(osm_id: int) -> str:
    return f"""
[out:json][timeout:950];
rel({osm_id});map_to_area->.searchCountry;
(
  {"\n".join([f"node(area.searchCountry)[population][place={place_type}];" for place_type in TYPE_ORDER])}
);
out body;
"""


# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
def fetch_all_populations(country: Country) -> PlaceData:
    query = build_query(country.osm_id)
    print(f"Querying Overpass API for {country.name} …")

    max_retries = 6
    backoff = 6  # seconds to wait on first 429
    for attempt in range(max_retries):
        OVERPASS_URL = (
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
            if attempt == 5
            else "https://overpass-api.de/api/interpreter"
        )

        wait = backoff * 2**attempt
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=1000)
        except requests.exceptions.ConnectionError:
            print(
                f"  Connection error — retrying in {wait}s (attempt {attempt + 1}/{max_retries}) …"
            )
            time.sleep(wait)
            continue
        if (
            resp.status_code == 429
            or resp.status_code >= 500
            and attempt < max_retries - 1
        ):
            print(
                f"  HTTP {resp.status_code} — retrying in {wait}s (attempt {attempt + 1}/{max_retries}) …"
            )
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()

    elements = resp.json().get("elements", [])

    data: PlaceData = {pt: [] for pt in PLACE_TYPES}
    for el in elements:
        tags = el.get("tags", {})
        place_type = tags.get("place", "")
        if place_type not in data:
            continue
        raw = tags.get("population", "")
        name = tags.get("name", "")
        lat = el.get("lat")
        lon = el.get("lon")
        osm_type = el.get("type")
        osm_id = el.get("id")
        wikidata = tags.get("wikidata")
        try:
            pop = int(raw.replace(",", "").replace(" ", "").split(".")[0])
            if pop <= 0:
                continue
            data[place_type].append(
                {
                    "name": name,
                    "population": pop,
                    "lat": lat,
                    "lon": lon,
                    "osm_type": osm_type,
                    "osm_id": osm_id,
                    "wikidata": wikidata,
                }
            )
        except ValueError:
            pass

    for pt, records in data.items():
        print(f"  {pt.capitalize()}: {len(records)} records")
    return data


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def _percentile_of_score(arr: np.ndarray, score: float) -> float:
    """Fraction of values in *arr* that are <= *score*, expressed as 0–100."""
    return float(np.mean(arr <= score) * 100)


def _norm_logpdf(x: float, mu: float, sigma: float) -> float:
    """Log-PDF of N(mu, sigma²) evaluated at *x*."""
    return float(
        -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2
    )


def _normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to [0, 1] range given min and max bounds."""
    if max_val <= min_val:
        return 0.0
    clamped = max(min_val, min(max_val, value))
    return (clamped - min_val) / (max_val - min_val)


def _compute_local_rank_percentile(
    lat: float,
    lon: float,
    pop: float,
    neighbors: list[PlaceRecord],
    distance_threshold_km: float,
) -> float:
    """
    Compute percentile rank of a place among neighbors within distance_threshold_km.
    Uses Haversine distance for accurate lat/lon calculations.
    Returns percentile 0-100.

    Args:
        lat, lon: coordinates of the place to rank
        pop: population of the place to rank
        neighbors: list of all neighboring places
        distance_threshold_km: maximum distance to consider

    Returns:
        percentile (0-100) of the place's population among nearby places
    """
    if not neighbors:
        return 50.0  # No neighbors, default to middle

    # Haversine distance calculation
    def haversine_km(lat1, lon1, lat2, lon2):
        """Calculate distance in km between two lat/lon points."""
        R = 6371.0  # Earth radius in km
        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    # Filter neighbors by distance
    nearby = [
        n
        for n in neighbors
        if haversine_km(lat, lon, n["lat"], n["lon"]) <= distance_threshold_km
    ]

    if not nearby:
        return 50.0

    # Get populations of nearby places
    nearby_pops = np.array([n["population"] for n in nearby], dtype=float)

    # Percentile: fraction of neighbors with lower or equal population
    percentile = _percentile_of_score(nearby_pops, pop)
    return percentile


def _make_explanation(
    pop: int,
    place_type: str,
    upper_type: str | None,
    lower_type: str | None,
    own_pct: float,
    up_pct: float | None,
    down_pct: float | None,
    flag: str,
    confidence: str,
) -> str:
    conf_phrases = {
        "strong": "a strong candidate for review",
        "weak": "a weak candidate for review",
        "monitor": "worth monitoring",
        "consistent": "consistent with its classification",
    }
    base = f"Population of {pop:,} sits at the {own_pct:.0f}th percentile among {place_type}s."
    if flag == "upgrade" and up_pct is not None:
        detail = (
            f" It would rank at the {up_pct:.0f}th percentile among {upper_type}s, "
            f"making it {conf_phrases[confidence]} for upgrade."
        )
    elif flag == "downgrade" and down_pct is not None:
        detail = (
            f" It would rank at the {down_pct:.0f}th percentile among {lower_type}s, "
            f"making it {conf_phrases[confidence]} for downgrade."
        )
    else:
        detail = f" No cross-type overlap detected — {conf_phrases[confidence]}."
    return base + detail


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_classification(data: PlaceData) -> PlaceData:
    """
    Hybrid misclassification analysis with scoring system and spatial component.

    For each record, mutates in-place adding:
      - own_percentile       – percentile rank within its own type (0–100)
      - upgrade_percentile   – percentile rank within the next-higher type, or None
      - downgrade_percentile – percentile rank within the next-lower type, or None
      - local_rank_percentile – percentile rank among neighbors within distance threshold

      - llr_up_score        – normalized [0-1] LLR score for upgrade
      - llr_down_score      – normalized [0-1] LLR score for downgrade
      - local_rank_score    – normalized [0-1] local rank signal
      - percentile_up_bonus – normalized [0-1] percentile corroboration for upgrade
      - percentile_down_bonus – normalized [0-1] percentile corroboration for downgrade

      - upgrade_score       – combined upgrade signal [0-1]
      - downgrade_score     – combined downgrade signal [0-1]
      - net_score          – upgrade_score - downgrade_score
      - strength           – absolute value of net_score

      - flag                – "upgrade" | "downgrade" | "consistent"
      - confidence          – "strong" | "weak" | "monitor" | "consistent"
      - explanation         – human-readable summary

    Approach:
      1. Fit log-normal distribution per type.
      2. Compute LLR (log-likelihood ratio) for upgrade/downgrade signals.
      3. Compute percentile ranks (global and local spatial).
      4. Normalize all signals to [0-1] scale.
      5. Combine signals equally weighted to get final scores.
      6. Determine flag + confidence based on net_score strength.
    """
    # Distance thresholds by place type (km)
    DISTANCE_THRESHOLDS = {
        "city": 100,
        "town": 50,
        "village": 25,
    }

    # ── Step 1: fit log-normal params and collect raw pop arrays per type ──
    log_params: dict[str, tuple[float, float]] = {}
    all_pops: dict[str, np.ndarray] = {}

    for place_type, records in data.items():
        if len(records) < 2:
            continue
        pops = np.array([r["population"] for r in records], dtype=float)
        log_pops = np.log10(pops)
        mu = float(np.mean(log_pops))
        sigma = float(np.std(log_pops, ddof=1))
        log_params[place_type] = (mu, sigma)
        all_pops[place_type] = pops
        print(
            f"  {place_type.capitalize()} – "
            f"log10 mean: {mu:.3f}  |  log10 std: {sigma:.3f}  |  "
            f"geometric mean pop: {10**mu:,.0f}"
        )

    # Flatten all places for spatial lookups
    all_places_flat = []
    for place_type, records in data.items():
        all_places_flat.extend(records)

    # ── Step 2: score every record using scoring system ──
    for place_type, records in data.items():
        type_idx = TYPE_ORDER.index(place_type)
        upper_type = TYPE_ORDER[type_idx - 1] if type_idx > 0 else None
        lower_type = (
            TYPE_ORDER[type_idx + 1] if type_idx < len(TYPE_ORDER) - 1 else None
        )

        own_pops = all_pops.get(place_type, np.array([]))
        upper_pops = all_pops.get(upper_type) if upper_type else None
        lower_pops = all_pops.get(lower_type) if lower_type else None

        mu_curr, sigma_curr = log_params.get(place_type, (0.0, 1.0))

        for r in records:
            pop = r["population"]
            log_pop = np.log10(pop)
            lat = r["lat"]
            lon = r["lon"]

            # ── Global percentile ranks ──
            own_pct = _percentile_of_score(own_pops, pop)
            up_pct = (
                _percentile_of_score(upper_pops, pop)
                if upper_pops is not None
                else None
            )
            down_pct = (
                _percentile_of_score(lower_pops, pop)
                if lower_pops is not None
                else None
            )

            # ── Local spatial percentile rank ──
            distance_threshold = DISTANCE_THRESHOLDS.get(place_type, 50)
            local_rank_pct = _compute_local_rank_percentile(
                lat, lon, pop, all_places_flat, distance_threshold
            )

            # ── Log-likelihood ratios vs adjacent types ──
            curr_ll = _norm_logpdf(log_pop, mu_curr, sigma_curr)

            llr_up: float | None = None
            if upper_type and upper_type in log_params:
                mu_up, sigma_up = log_params[upper_type]
                llr_up = _norm_logpdf(log_pop, mu_up, sigma_up) - curr_ll

            llr_down: float | None = None
            if lower_type and lower_type in log_params:
                mu_down, sigma_down = log_params[lower_type]
                llr_down = _norm_logpdf(log_pop, mu_down, sigma_down) - curr_ll

            # ── Normalize LLR scores to [0, 1] ──
            # LLR range: typically [-5, 5], normalize to [0, 1]
            llr_up_score = (
                _normalize_score(llr_up or 0.0, -3.0, 3.0)
                if llr_up is not None
                else 0.0
            )
            llr_down_score = (
                _normalize_score(llr_down or 0.0, -3.0, 3.0)
                if llr_down is not None
                else 0.0
            )

            # ── Local rank score: signal based on local vs global importance ──
            # A place that is important locally but labeled as low-tier might be mislabeled
            # Conversely, a place that is unimportant locally but labeled as high-tier might be mislabeled
            local_rank_score = 0.0
            if place_type == "village":
                # Villages: being locally important suggests upgrade
                # If in top 50% locally, signal for upgrade
                if local_rank_pct > 50:
                    local_rank_score = _normalize_score(local_rank_pct, 50.0, 100.0)
                # If in bottom 10% locally, signal for downgrade (incorrect classification)
                elif local_rank_pct < 10:
                    local_rank_score = -_normalize_score(local_rank_pct, 0.0, 10.0)
            elif place_type == "city":
                # Cities: being locally dominant is GOOD (confirms classification)
                # If in bottom 30% locally, that's suspicious - signal for downgrade
                if local_rank_pct < 30:
                    local_rank_score = _normalize_score(
                        30.0 - local_rank_pct, 0.0, 30.0
                    )
                # High local rank is GOOD for cities - no signal, it confirms the classification
            elif place_type == "town":
                # Towns: being locally important suggests upgrade to city
                if local_rank_pct > 70:
                    # High locally, might be a city
                    local_rank_score = (
                        _normalize_score(local_rank_pct, 70.0, 100.0) * 0.6
                    )
                # Being locally unimportant suggests downgrade to village
                elif local_rank_pct < 30:
                    # Low locally, might be a village
                    local_rank_score = (
                        -_normalize_score(local_rank_pct, 0.0, 30.0) * 0.6
                    )

            # ── Percentile corroboration bonuses ──
            percentile_up_bonus = 0.0
            if up_pct is not None and own_pct >= 75:
                # High in own type, check if reasonable in target type
                if 20 <= up_pct <= 65:
                    percentile_up_bonus = _normalize_score(own_pct, 75.0, 100.0)

            percentile_down_bonus = 0.0
            if down_pct is not None and own_pct <= 25:
                # Low in own type, check if reasonable in target type
                if 35 <= down_pct <= 80:
                    percentile_down_bonus = _normalize_score(25.0 - own_pct, 0.0, 25.0)

            # ── Combine scores (equal weighting) ──
            upgrade_score = (
                llr_up_score + max(0.0, local_rank_score) + percentile_up_bonus
            ) / 3.0
            downgrade_score = (
                llr_down_score + max(0.0, -local_rank_score) + percentile_down_bonus
            ) / 3.0

            net_score = upgrade_score - downgrade_score
            strength = abs(net_score)

            # ── Determine flag ──
            if net_score > 0.1:
                flag = "upgrade"
            elif net_score < -0.1:
                flag = "downgrade"
            else:
                flag = "consistent"

            # ── Determine confidence ──
            if strength >= 0.67:
                confidence = "strong"
            elif strength >= 0.33:
                confidence = "weak"
            elif flag == "consistent" and (
                (upper_type and own_pct >= 90) or (lower_type and own_pct <= 10)
            ):
                confidence = "monitor"
            else:
                confidence = "consistent"

            # Store all results
            r["own_percentile"] = round(own_pct, 1)
            r["upgrade_percentile"] = round(up_pct, 1) if up_pct is not None else None
            r["downgrade_percentile"] = (
                round(down_pct, 1) if down_pct is not None else None
            )
            r["local_rank_percentile"] = round(local_rank_pct, 1)

            # Intermediate scores
            r["llr_up_score"] = round(llr_up_score, 3)
            r["llr_down_score"] = round(llr_down_score, 3)
            r["local_rank_score"] = round(local_rank_score, 3)
            r["percentile_up_bonus"] = round(percentile_up_bonus, 3)
            r["percentile_down_bonus"] = round(percentile_down_bonus, 3)

            # Final scores
            r["upgrade_score"] = round(upgrade_score, 3)
            r["downgrade_score"] = round(downgrade_score, 3)
            r["net_score"] = round(net_score, 3)
            r["strength"] = round(strength, 3)

            r["flag"] = flag
            r["confidence"] = confidence
            r["explanation"] = _make_explanation(
                pop,
                place_type,
                upper_type,
                lower_type,
                own_pct,
                up_pct,
                down_pct,
                flag,
                confidence,
            )

    return data


# ---------------------------------------------------------------------------
# GeoJSON output
# ---------------------------------------------------------------------------
def write_geojson(data: PlaceData, country_code: str) -> None:
    features: list[dict] = []

    for place_type, records in data.items():
        for r in records:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [r["lon"], r["lat"]],
                    },
                    "properties": {
                        "name": r["name"],
                        "place": place_type,
                        "population": r["population"],
                        "own_percentile": r.get("own_percentile"),
                        "upgrade_percentile": r.get("upgrade_percentile"),
                        "downgrade_percentile": r.get("downgrade_percentile"),
                        "local_rank_percentile": r.get("local_rank_percentile"),
                        # Intermediate scores
                        "llr_up_score": r.get("llr_up_score"),
                        "llr_down_score": r.get("llr_down_score"),
                        "local_rank_score": r.get("local_rank_score"),
                        "percentile_up_bonus": r.get("percentile_up_bonus"),
                        "percentile_down_bonus": r.get("percentile_down_bonus"),
                        # Final scores
                        "upgrade_score": r.get("upgrade_score"),
                        "downgrade_score": r.get("downgrade_score"),
                        "net_score": r.get("net_score"),
                        "strength": r.get("strength"),
                        "flag": r.get("flag", "consistent"),
                        "confidence": r.get("confidence", "consistent"),
                        "osm_type": r.get("osm_type"),
                        "osm_id": r.get("osm_id"),
                        "wikidata": r.get("wikidata"),
                    },
                }
            )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    out_path = f"output/geojson/osm_places_{country_code}.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"GeoJSON saved to {out_path}  ({len(features):,} features)")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dark-theme palette (mirrors the website CSS variables)
# ---------------------------------------------------------------------------
_BG = "#0e0f13"  # --bg
_SURFACE = "#16181f"  # --surface
_SURFACE2 = "#1e2029"  # --surface-2
_BORDER = "#2a2d3a"  # --border
_TEXT = "#e8eaf0"  # --text
_TEXT_MUTED = "#6b7280"  # --text-muted


def plot(data: PlaceData, country: Country, generated_at: str | None = None) -> None:
    # ── Dark-theme rcParams ──────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "figure.facecolor": _BG,
            "axes.facecolor": _SURFACE,
            "axes.edgecolor": _BORDER,
            "axes.labelcolor": _TEXT,
            "xtick.color": _TEXT_MUTED,
            "ytick.color": _TEXT,
            "text.color": _TEXT,
            "grid.color": _BORDER,
            "legend.facecolor": _SURFACE2,
            "legend.edgecolor": _BORDER,
            "legend.labelcolor": _TEXT,
            "savefig.facecolor": _BG,
            "savefig.edgecolor": _BG,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    y_positions = {pt: i for i, pt in enumerate(PLACE_TYPES)}
    jitter_amount = 0.25
    rng = np.random.default_rng(42)

    for place_type, records in data.items():
        if not records:
            continue

        color = COLORS[place_type]
        y = y_positions[place_type]
        pops_arr = np.array([r["population"] for r in records], dtype=float)
        jitter = rng.uniform(-jitter_amount, jitter_amount, size=len(pops_arr))

        ax.scatter(
            pops_arr,
            y + jitter,
            alpha=0.45,
            s=10,
            color=color,
            linewidths=0,
        )

        # Median line
        median = np.median(pops_arr)
        ax.plot(
            [median, median],
            [y - jitter_amount - 0.05, y + jitter_amount + 0.05],
            color=color,
            linewidth=2.5,
            solid_capstyle="round",
        )
        ax.text(
            median,
            y + jitter_amount + 0.12,
            f"median\n{int(median):,}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )

        # Label the smallest and largest places
        sorted_records = sorted(records, key=lambda r: r["population"])
        for rec, valign, offset_sign in [
            (sorted_records[0], "top", -1),
            (sorted_records[-1], "bottom", +1),
        ]:
            name, pop = rec["name"], rec["population"]
            if not name:
                continue
            ax.annotate(
                f"{name}\n({pop:,})",
                xy=(pop, y),
                xytext=(pop, y + offset_sign * (jitter_amount + 0.18)),
                ha="center",
                va=valign,
                fontsize=7,
                color=color,
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=0.8,
                    shrinkA=0,
                    shrinkB=2,
                ),
            )

    # Axes
    ax.set_xscale("log")
    ax.set_xlabel("Population (log scale)", fontsize=12, color=_TEXT_MUTED)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([pt.capitalize() for pt in PLACE_TYPES], fontsize=12)
    ax.set_ylim(-0.75, len(PLACE_TYPES) - 0.25)
    ax.set_title(
        f"OSM place= population distributions — {country.name} ({country.code})"
        "\n(city / town / village)",
        fontsize=14,
        pad=14,
        color=_TEXT,
    )

    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)

    # Grid
    ax.xaxis.grid(
        True, which="both", linestyle="--", linewidth=0.5, alpha=0.4, color=_BORDER
    )
    ax.set_axisbelow(True)

    # Legend / counts
    patches = [
        mpatches.Patch(
            color=COLORS[pt], label=f"{pt.capitalize()}  (n={len(data[pt]):,})"
        )
        for pt in PLACE_TYPES
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)

    # ── Timestamp annotation ─────────────────────────────────────────────────
    ts = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.text(
        0.99,
        0.01,
        f"Data as of {ts}",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=_TEXT_MUTED,
        fontstyle="italic",
    )

    fig.tight_layout()
    out_path = f"output/img/osm_population_plot_{country.code}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close(fig)

    # Reset rcParams so subsequent calls (if any) are not affected
    plt.rcParams.update(plt.rcParamsDefault)


# ---------------------------------------------------------------------------
# Per-country runner
# ---------------------------------------------------------------------------
def run_for_country(country: Country, *, show_plot: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {country.name} ({country.code})")
    print(f"{'=' * 60}")

    data = fetch_all_populations(country)

    total = sum(len(v) for v in data.values())
    if total == 0:
        print(
            f"No data returned for {country.name} — "
            "skipping (no population tags, or country not in OSM)."
        )
        return

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"\nTotal records: {total:,}")
    print("\nFitting log-normal distributions and scoring classification …")
    analyze_classification(data)

    write_geojson(data, country.code)
    plot(data, country, generated_at=generated_at)

    if show_plot:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch OSM place population data and produce a GeoJSON + strip-chart "
            "for one or all countries."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c",
        "--country",
        metavar="CODE",
        help=(
            "Two-letter ISO 3166-1 alpha-2 country code to process (e.g. CO, US, DE)."
        ),
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="all_countries",
        help="Process every country in the countries list sequentially.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        default=False,
        help="Display each plot interactively in addition to saving it (default: off).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.all_countries:
        failed: list[str] = []
        for country in countries:
            try:
                run_for_country(country, show_plot=args.show_plot)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR processing {country.name} ({country.code}): {exc}")
                failed.append(country.code)
            time.sleep(5)

        print(f"\n{'=' * 60}")
        print(f"Done. Processed {len(countries)} countries.")
        if failed:
            print(f"Failed ({len(failed)}): {', '.join(failed)}")
    else:
        code = args.country.upper()
        country = COUNTRIES_BY_CODE.get(code)
        if country is None:
            print(
                f"Error: '{code}' is not a recognised country code in countries.py.",
                file=sys.stderr,
            )
            sys.exit(1)
        run_for_country(country, show_plot=args.show_plot)


if __name__ == "__main__":
    main()
