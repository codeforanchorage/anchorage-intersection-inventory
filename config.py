"""Project configuration: paths, query parameters, scoring rubric."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
RESULTS_DIR = DATA_DIR / "results"

SIGNALS_GEOJSON = DATA_DIR / "signals.geojson"
IMAGERY_MANIFEST = RESULTS_DIR / "imagery_manifest.csv"
INVENTORY_GEOJSON = RESULTS_DIR / "intersection_inventory.geojson"
INVENTORY_CSV = RESULTS_DIR / "intersection_inventory.csv"

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

OVERPASS_QUERY = """
[out:json][timeout:90];
area["name"="Anchorage"]["boundary"="administrative"]["admin_level"="6"]->.a;
node["highway"="traffic_signals"](area.a);
out body;
"""

# Anchorage bounding box fallback if the named-area query returns nothing.
# (south, west, north, east)
ANCHORAGE_BBOX = (61.05, -150.20, 61.40, -149.50)

OVERPASS_QUERY_BBOX = f"""
[out:json][timeout:90];
node["highway"="traffic_signals"]({ANCHORAGE_BBOX[0]},{ANCHORAGE_BBOX[1]},{ANCHORAGE_BBOX[2]},{ANCHORAGE_BBOX[3]});
out body;
"""

# Cluster signal nodes within this radius into a single intersection.
CLUSTER_RADIUS_METERS = 30.0

# Google Street View parameters.
GSV_BASE_URL = "https://maps.googleapis.com/maps/api/streetview"
GSV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
GSV_HEADINGS = [0, 90, 180, 270]
GSV_SIZE = "640x640"
GSV_PITCH = 10
GSV_FOV = 90
GSV_RATE_LIMIT_SEC = 1.0

# Asset condition rubric.
CONDITION_SCORES = {
    "good": 1.0,
    "fair": 0.6,
    "poor": 0.2,
    "not_assessable": None,
}

# Asset types we expect Phase 3 detectors to emit. Used for output flattening.
ASSET_TYPES = [
    "traffic_signal_head",
    "pedestrian_signal",
    "signal_pole",
    "mast_arm",
    "signal_cabinet",
    "crosswalk_marking",
    "curb_ramp",
    "push_button",
    "street_light",
    "road_sign",
    "lane_marking",
    "other",
]
