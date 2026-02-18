"""
OSM Population Distribution Plot
Fetches city, town, and village nodes from Overpass API in a single
request and plots their population distributions as a strip/jitter chart.
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PLACE_TYPES = ["city", "town", "village"]
QUERY = """
[out:json][timeout:120];
area["ISO3166-1"="CO"][admin_level=2];
(
  node(area)["place"~"^(city|town|village)$"][population];
);
out body;
"""

COLORS = {
    "city": "#e15759",
    "town": "#4e79a7",
    "village": "#59a14f",
}


# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
def fetch_all_populations() -> dict[str, list[tuple[str, int]]]:
    print("Querying Overpass API (single request) …")
    resp = requests.post(OVERPASS_URL, data={"data": QUERY}, timeout=150)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    data: dict[str, list[tuple[str, int]]] = {pt: [] for pt in PLACE_TYPES}
    for el in elements:
        tags = el.get("tags", {})
        place_type = tags.get("place", "")
        if place_type not in data:
            continue
        raw = tags.get("population", "")
        name = tags.get("name", "")
        try:
            pop = int(raw.replace(",", "").replace(" ", "").split(".")[0])
            data[place_type].append((name, pop))
        except ValueError:
            pass

    for pt, records in data.items():
        print(f"  {pt.capitalize()}: {len(records)} records")
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(data: dict[str, list[tuple[str, int]]]):
    fig, ax = plt.subplots(figsize=(14, 6))

    y_positions = {pt: i for i, pt in enumerate(PLACE_TYPES)}
    jitter_amount = 0.25
    rng = np.random.default_rng(42)

    for place_type, records in data.items():
        if not records:
            continue

        color = COLORS[place_type]
        y = y_positions[place_type]
        pops_arr = np.array([r[1] for r in records], dtype=float)
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
        sorted_records = sorted(records, key=lambda r: r[1])
        for name, pop, valign, offset_sign in [
            (sorted_records[0][0], sorted_records[0][1], "top", -1),
            (sorted_records[-1][0], sorted_records[-1][1], "bottom", +1),
        ]:
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
        "OSM place= population distributions (city / town / village)",
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
    out_path = "osm_population_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = fetch_all_populations()

    total = sum(len(v) for v in data.values())
    if total == 0:
        print("No data returned – check your internet connection or Overpass URL.")
    else:
        print(f"\nTotal records: {total:,}")
        plot(data)
