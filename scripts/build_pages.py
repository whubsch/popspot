import json
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

output_dir = Path("output")
env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("page.html.j2")

# Find all GeoJSON files in the output directory
geojson_files = sorted(output_dir.glob("osm_places_*.geojson"))

for geojson_path in geojson_files:
    with open(geojson_path) as f:
        geojson = json.load(f)

    # Extract country code from filename (e.g., osm_places_CO.geojson -> CO)
    country_code = geojson_path.stem.split("_")[-1]

    html = template.render(
        features=geojson["features"],
        country=country_code,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    # Generate HTML file with country code in the filename
    output_html = output_dir / f"pages/{country_code}.html"
    with open(output_html, "w") as f:
        f.write(html)

    print(f"Generated {output_html} ({len(geojson['features'])} features)")
