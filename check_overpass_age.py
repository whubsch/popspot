"""
Check the age of data on the three main Overpass API servers.

uv run check_overpass_age.py
"""

import requests
from datetime import datetime, timezone

# The three main Overpass API servers
SERVERS = {
    "Main (overpass-api.de)": "https://overpass-api.de/api/interpreter",
    "VK Maps (maps.mail.ru)": "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "Private.coffee": "https://overpass.private.coffee/api/interpreter",
}

# A minimal query that returns metadata without downloading much data
MINIMAL_QUERY = """
[out:json];
node(1);
out;
"""


def get_server_data_age(server_name: str, endpoint: str):
    """
    Query an Overpass API server and return the age of its data.

    Args:
        server_name: Display name for the server
        endpoint: API endpoint URL

    Returns:
        Tuple of (timestamp_osm_base, timestamp_areas_base, age_string) or None if query fails
    """
    try:
        print(f"Querying {server_name}...", end=" ", flush=True)

        response = requests.post(endpoint, data=MINIMAL_QUERY, timeout=15)
        response.raise_for_status()

        data = response.json()

        # Extract the timestamps
        osm3s = data.get("osm3s", {})
        timestamp_osm = osm3s.get("timestamp_osm_base")
        timestamp_areas = osm3s.get("timestamp_areas_base")

        if not timestamp_osm:
            print("❌ No timestamp found")
            return None

        # Parse the timestamps (format: "2026-02-24T01:17:38Z")
        osm_dt = datetime.fromisoformat(timestamp_osm.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        # Calculate age
        age = now - osm_dt
        days = age.days
        hours = age.seconds // 3600
        minutes = (age.seconds % 3600) // 60

        age_string = f"{days}d {hours}h {minutes}m old"

        print("✓")
        return (timestamp_osm, timestamp_areas, age_string)

    except requests.exceptions.Timeout:
        print("❌ Timeout")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e.response.status_code}")
        return None
    except (ValueError, KeyError) as e:
        print(f"❌ Parse error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Main function to check all servers."""
    print("=" * 70)
    print("Overpass API Server Data Age Check")
    print("=" * 70)
    print()

    results = {}

    for server_name, endpoint in SERVERS.items():
        result = get_server_data_age(server_name, endpoint)
        if result:
            timestamp_osm, timestamp_areas, age = result
            results[server_name] = (timestamp_osm, timestamp_areas, age)

    print()
    print("=" * 70)
    print("Results:")
    print("=" * 70)

    if not results:
        print("❌ Could not retrieve data from any server")
        return

    for server_name, (timestamp_osm, timestamp_areas, age) in results.items():
        print(f"\n{server_name}")
        print(f"  OSM base data:   {timestamp_osm}")
        if timestamp_areas:
            print(f"  Areas base data: {timestamp_areas}")
        print(f"  Age: {age}")

    print()


if __name__ == "__main__":
    main()
