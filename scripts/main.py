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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
from countries import Country, countries

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
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
def build_query(country_code: str) -> str:
    return f"""
[out:json][timeout:120];
area["ISO3166-1"="{country_code}"][admin_level=2];
(
  node(area)["place"~"^(city|town|village)$"][population];
);
out body;
"""


# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
def fetch_all_populations(country_code: str) -> PlaceData:
    query = build_query(country_code)
    print(f"Querying Overpass API for {country_code} …")

    max_retries = 6
    backoff = 60  # seconds to wait on first 429
    for attempt in range(max_retries):
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=150)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = backoff * 2**attempt
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
    Hybrid misclassification analysis run independently per place type.

    For each record, mutates in-place adding:
      - own_percentile       – percentile rank within its own type (0–100)
      - upgrade_percentile   – percentile rank within the next-higher type, or None
      - downgrade_percentile – percentile rank within the next-lower type, or None
      - flag                 – "upgrade" | "downgrade" | "consistent"
      - confidence           – "strong" | "weak" | "monitor" | "consistent"
      - explanation          – human-readable summary

    Approach:
      1. Fit a log-normal distribution (mean/std of log10 populations) per type.
      2. Compute log-likelihood ratio (LLR) of each record under adjacent-type
         distributions vs its own (approach 2 — cross-type overlap).
      3. Compute percentile rank within own type and adjacent types
         (approach 3 — cross-type percentile comparison).
      4. Combine both signals to assign flag + confidence tier.
    """
    # ── Step 1: fit log-normal params and collect raw pop arrays per type ──
    log_params: dict[
        str, tuple[float, float]
    ] = {}  # type -> (mu, sigma) in log10 space
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

    # ── Step 2 & 3: score every record ──
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

            # Percentile ranks
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

            # Log-likelihood ratios vs adjacent types
            curr_ll = _norm_logpdf(log_pop, mu_curr, sigma_curr)

            llr_up: float | None = None
            if upper_type and upper_type in log_params:
                mu_up, sigma_up = log_params[upper_type]
                llr_up = round(_norm_logpdf(log_pop, mu_up, sigma_up) - curr_ll, 3)

            llr_down: float | None = None
            if lower_type and lower_type in log_params:
                mu_down, sigma_down = log_params[lower_type]
                llr_down = round(
                    _norm_logpdf(log_pop, mu_down, sigma_down) - curr_ll, 3
                )

            # ── Determine flag ──
            upgrade_ll_signal = llr_up is not None and llr_up > 0
            downgrade_ll_signal = llr_down is not None and llr_down > 0

            if upgrade_ll_signal and downgrade_ll_signal:
                # Both fire — follow the stronger LLR
                flag = "upgrade" if (llr_up or 0) >= (llr_down or 0) else "downgrade"
            elif upgrade_ll_signal:
                flag = "upgrade"
            elif downgrade_ll_signal:
                flag = "downgrade"
            else:
                flag = "consistent"

            # ── Determine confidence ──
            # "strong"    – LLR > 1.0 and percentile rank corroborates
            # "weak"      – LLR > 0 (distribution signal present)
            # "monitor"   – no LLR signal but extreme percentile in own type
            # "consistent"– no signal
            if flag == "upgrade":
                pct_corroborates = up_pct is not None and 20 <= up_pct <= 65
                if (llr_up or 0) > 1.0 and own_pct >= 75 and pct_corroborates:
                    confidence = "strong"
                else:
                    confidence = "weak"
            elif flag == "downgrade":
                pct_corroborates = down_pct is not None and 35 <= down_pct <= 80
                if (llr_down or 0) > 1.0 and own_pct <= 25 and pct_corroborates:
                    confidence = "strong"
                else:
                    confidence = "weak"
            else:
                # No LLR signal — check for extreme percentile worth monitoring
                if (upper_type and own_pct >= 90) or (lower_type and own_pct <= 10):
                    flag = "consistent"
                    confidence = "monitor"
                else:
                    confidence = "consistent"

            r["own_percentile"] = round(own_pct, 1)
            r["upgrade_percentile"] = round(up_pct, 1) if up_pct is not None else None
            r["downgrade_percentile"] = (
                round(down_pct, 1) if down_pct is not None else None
            )
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
            if r.get("lat") is None or r.get("lon") is None:
                continue
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
                        "flag": r.get("flag", "consistent"),
                        "confidence": r.get("confidence", "consistent"),
                        "explanation": r.get("explanation", ""),
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
def plot(data: PlaceData, country: Country) -> None:
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
            alpha=0.35,
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
    ax.set_xlabel("Population (log scale)", fontsize=12)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([pt.capitalize() for pt in PLACE_TYPES], fontsize=12)
    ax.set_ylim(-0.75, len(PLACE_TYPES) - 0.25)
    ax.set_title(
        f"OSM place= population distributions — {country.name} ({country.code})"
        "\n(city / town / village)",
        fontsize=14,
        pad=14,
    )

    # Grid
    ax.xaxis.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Legend / counts
    patches = [
        mpatches.Patch(
            color=COLORS[pt], label=f"{pt.capitalize()}  (n={len(data[pt]):,})"
        )
        for pt in PLACE_TYPES
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    out_path = f"output/img/osm_population_plot_{country.code}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-country runner
# ---------------------------------------------------------------------------
def run_for_country(country: Country, *, show_plot: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {country.name} ({country.code})")
    print(f"{'=' * 60}")

    data = fetch_all_populations(country.code)

    total = sum(len(v) for v in data.values())
    if total == 0:
        print(
            f"No data returned for {country.name} — "
            "skipping (no population tags, or country not in OSM)."
        )
        return

    print(f"\nTotal records: {total:,}")
    print("\nFitting log-normal distributions and scoring classification …")
    analyze_classification(data)

    write_geojson(data, country.code)
    plot(data, country)

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
