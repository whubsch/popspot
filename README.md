# PopSpot

Population tag analysis for OpenStreetMap place hierarchies.

## Overview

PopSpot analyzes population data tagged on OpenStreetMap `place=*` objects to identify potential classification improvements. It compares each place's population against the statistical distribution of its type and adjacent types, flagging candidates for reclassification.

## Features

- **Upgrade candidates** — Places whose population is more consistent with a higher-tier place type
- **Downgrade candidates** — Places whose population is more consistent with a lower-tier place type
- **Monitor tier** — Places at extreme percentiles within their type, worth revisiting as data improves
- **Multiple editors** — Direct links to edit objects in iD, JOSM, and Level0
- **Population analysis** — Visual distribution plots for each country

## Getting Started

```bash
# Install dependencies
uv sync

# Generate reports for countries
uv run scripts/main.py

# Build HTML pages
uv run scripts/build_pages.py

# Build home page
uv run scripts/build_home.py
```

Output is saved to `output/pages/` and `output/geojson/`.

## Data & Attribution

Data sourced from [OpenStreetMap](https://www.openstreetmap.org) via Overpass API. © OpenStreetMap contributors ([ODbL](https://opendatacommons.org/licenses/odbl/)).

Statistical flags are prompts for investigation, not conclusions. Place hierarchy in OSM reflects infrastructure, administrative status, and regional importance—population alone cannot capture that context.

## License

See LICENSE file for details.
