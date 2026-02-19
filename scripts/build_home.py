"""Build the home page that lists all available country pages."""

from datetime import datetime, timezone
from pathlib import Path

from countries import countries
from jinja2 import Environment, FileSystemLoader

# Create a mapping of country codes to country names
country_map = {country.code: country.name for country in countries}

# Find all generated HTML pages
output_dir = Path("output")
pages_dir = output_dir / "pages"
html_files = sorted(pages_dir.glob("*.html"))

# Extract country codes from the HTML files and enrich with names
available_countries = []
for html_file in html_files:
    country_code = html_file.stem.upper()
    country_name = country_map.get(country_code, country_code)
    available_countries.append(
        {
            "code": country_code,
            "name": country_name,
            "path": f"pages/{country_code}.html",
        }
    )

# Sort by country name
available_countries.sort(key=lambda x: x["name"])

# Render the template
env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("home.html.j2")

html = template.render(
    countries=available_countries,
    generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    total_countries=len(available_countries),
)

# Write the home page
output_html = output_dir / "index.html"
with open(output_html, "w") as f:
    f.write(html)

print(f"Generated {output_html} ({len(available_countries)} countries)")
