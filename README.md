# J/70 Cascais regatta router

Launch the local development UI from the workspace root:

```powershell
streamlit run app.py
```

Streamlit opens `http://localhost:8501`. The UI starts with synthetic wind, so
course editing, routing, polar inspection, and weather-field inspection remain
available when `GRIBSTREAM_API_KEY` is not configured. Nothing in this command
deploys or publishes the application.

On the **MAP** tab, select the leeward or windward mark and click the Folium map,
then refine either position with the six-decimal coordinate inputs. The weather
selector offers IPMA Portugal AROME-PT2 2.5 km, ICON-EU, and synthetic wind.
**USE CACHED WEATHER** loads the selected provider's most recent local subset
without an API call. Changing the forecast valid hour reuses the loaded dataset.

The **Current** map layer uses the Copernicus Marine IBI hourly surface-current
dataset (`cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m`). Configure your account in
the PowerShell session before starting Streamlit:

```powershell
$env:COPERNICUSMARINE_SERVICE_USERNAME = "your-username"
$env:COPERNICUSMARINE_SERVICE_PASSWORD = "your-password"
streamlit run app.py
```

Alternatively, run `copernicusmarine login` once and use its saved credentials.
Open **WEATHER**, press **LOAD / REFRESH CURRENT**, then select **Current** above
the map. Only `uo` and `vo` over the course plus its margin are retained; subsets
are cached under `data/copernicus/`. When loaded and valid for the race time,
current is added to boat-through-water velocity during every routing step.

This first milestone numerically routes one windward and one downwind leg through
a synthetic wind field. It uses a local Cartesian frame, legal ±45°/±135°
headings, a replaceable boat-speed function, maneuver limits, mark-radius
intersection, and a beam search with spatial deduplication.

Generate the visual milestone from the workspace root:

```powershell
python -m j70_router.demo
```

Run the tests with Python's built-in test runner:

```powershell
python -m unittest discover -s j70_router/tests -v
```

Generate visual checks for uniform wind, western pressure, a cross-course wind
shift, and two start times in a changing wind:

```powershell
python -m j70_router.validation
```

The generated `artifacts/milestone_route.png` contains a course/route panel and
a separate wind-speed explanation panel. Blue segments are port; orange segments
are starboard. Wind arrows show where the air flows, while all sailing angles use
meteorological true-wind direction (where the wind comes from).

The router includes GribStream DWD ICON-EU 0.0625° (about 7 km) and
Météo-France AROME 0.01° subset ingestion, disk caching, space/time U/V
interpolation, the published ORC J/70 best-performance polar, and
heading-uncertainty-aware target angles. GribStream credentials are read only
from `GRIBSTREAM_API_KEY`. ICON-EU is the working Cascais source; GribStream's
AROME output has no populated U/V cells at Cascais.

Generate the polar, VMG, and target-angle diagnostics:

```powershell
python -m j70_router.polar_validation
```

Run a latest-cycle AROME route after setting the token in the process
environment:

```powershell
$env:GRIBSTREAM_API_KEY = "your-token"
python -m j70_router.current_arome_demo
```

The request is limited to a 0.14° × 0.19° box around the race course and only
the 10 m U/V fields. Responses are cached under `data/gribstream/` by forecast
cycle, bounds, variables, and valid-time window.

The dated two-lap forecast experiment can be reproduced after downloading its
small ICON-EU fallback grid:

```powershell
python scripts\download_open_meteo.py --date 2026-09-21 --centre-lat 38.639 --centre-lon -9.497 --output data\forecast\icon_eu_cascais_20260921.json
python -m j70_router.today_race
```
