"""Download a small ICON-EU wind grid in a router-friendly JSON format."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--centre-lat", type=float, required=True)
    parser.add_argument("--centre-lon", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--half-lat", type=float, default=0.06)
    parser.add_argument("--half-lon", type=float, default=0.08)
    parser.add_argument("--grid-count", type=int, default=5)
    args = parser.parse_args()

    lats = np.linspace(args.centre_lat - args.half_lat, args.centre_lat + args.half_lat, args.grid_count)
    lons = np.linspace(args.centre_lon - args.half_lon, args.centre_lon + args.half_lon, args.grid_count)
    request_lats = [float(lat) for lat in lats for _ in lons]
    request_lons = [float(lon) for _ in lats for lon in lons]
    params = {
        "latitude": ",".join(f"{value:.6f}" for value in request_lats),
        "longitude": ",".join(f"{value:.6f}" for value in request_lons),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "start_date": args.date,
        "end_date": args.date,
        "models": "icon_eu",
        "cell_selection": "sea",
    }
    url = "https://api.open-meteo.com/v1/dwd-icon?" + urlencode(params)
    with urlopen(url, timeout=60) as response:
        raw = json.load(response)
    locations = raw if isinstance(raw, list) else [raw]
    if len(locations) != len(request_lats):
        raise RuntimeError(f"expected {len(request_lats)} locations, received {len(locations)}")

    times = locations[0]["hourly"]["time"]
    shape = (len(times), len(lats), len(lons))
    speeds = np.empty(shape)
    directions = np.empty(shape)
    source_grid_points: list[dict[str, float]] = []
    for flat_index, location in enumerate(locations):
        y, x = divmod(flat_index, len(lons))
        if location["hourly"]["time"] != times:
            raise RuntimeError("forecast time axes differ between grid points")
        speeds[:, y, x] = location["hourly"]["wind_speed_10m"]
        directions[:, y, x] = location["hourly"]["wind_direction_10m"]
        source_grid_points.append(
            {"latitude": float(location["latitude"]), "longitude": float(location["longitude"])}
        )

    payload = {
        "source": "Open-Meteo",
        "model": "DWD ICON-EU",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_url": url,
        "latitude": lats.tolist(),
        "longitude": lons.tolist(),
        "source_grid_points": source_grid_points,
        "time_utc": times,
        "wind_speed_mps": speeds.tolist(),
        "wind_direction_deg": directions.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Downloaded {len(times)} hours at {len(locations)} requested points to {args.output}")


if __name__ == "__main__":
    main()
