"""Local tools for CS 839 HW1 (continual learning through agent memory).

Data: us_states_popest2025.geojson
      TIGER/Line 2025 state boundaries (56 rows: 50 states, DC, Puerto Rico, Guam,
      U.S. Virgin Islands, American Samoa, Northern Mariana Islands) joined with the
      Census Bureau's Vintage 2025 population estimates (NST-EST2025-POP).

The agent calls these tools through function calling; it never writes code.
TOOL_SPECS (bottom of file) holds the descriptions given to the task agent and the memory agent.

get_population, get_area, get_density and distance share four optional arguments:
  greater_than / less_than  keep rows whose value is strictly above / below a number
  order                      "asc" or "desc"
  limit                      return at most this many rows (after sorting)
They return {"count", "total", "rows"}; count and total are over the returned rows.

Rules the agent can learn over the task sequence:
  Rule A (scope): "states" means the 50 states. list_regions returns 56 rows.
                  There is deliberately no "states only" filter: choosing the rows is the lesson.
  Rule B (area):  get_area and get_density default to EPSG:3857 (Web Mercator), which inflates area.
                  EPSG:5070 is equal-area. Its exception: for long distances use geodesic.
"""

import ast
import operator
import warnings
from pathlib import Path

import geopandas as gpd
from pyproj import Geod

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "us_states_popest2025.geojson"
POP_FIELD = "POPEST2025"
AREA_CRS = ("EPSG:3857", "EPSG:5070")

_gdf = gpd.read_file(DATA_PATH)
_NAMES = list(_gdf["NAME"])
_NAME_SET = set(_NAMES)
_CODE = dict(zip(_gdf["NAME"], _gdf["STUSPS"]))
_POP = {n: (None if p is None or p != p else int(p)) for n, p in zip(_gdf["NAME"], _gdf[POP_FIELD])}
_AREA = {crs: dict(zip(_gdf["NAME"], _gdf.to_crs(crs).area / 1e6)) for crs in AREA_CRS}
_centroids_ll = _gdf.to_crs("EPSG:5070").geometry.centroid.to_crs("EPSG:4326")
_CENTROID = {n: (p.x, p.y) for n, p in zip(_gdf["NAME"], _centroids_ll)}
_GEOD = Geod(ellps="WGS84")


# ---------- helpers ----------

def _resolve(names):
    names = list(_NAMES) if names == "all" else list(names)
    bad = [n for n in names if n not in _NAME_SET]
    if bad:
        raise ValueError(f"Unknown region name(s): {bad}. Use names exactly as list_regions returns them.")
    return names


def _select(values, key, greater_than=None, less_than=None, order=None, limit=None, ndigits=None):
    """Filter, sort and cut {name: value}. Rows without a value are dropped once a filter or order is used."""
    if order not in (None, "asc", "desc"):
        raise ValueError('order must be "asc" or "desc"')
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    rows = list(values.items())
    if greater_than is not None or less_than is not None or order is not None:
        rows = [(n, v) for n, v in rows if v is not None]
    if greater_than is not None:
        rows = [(n, v) for n, v in rows if v > greater_than]
    if less_than is not None:
        rows = [(n, v) for n, v in rows if v < less_than]
    if order is not None:
        rows.sort(key=lambda x: x[1], reverse=(order == "desc"))
    if limit is not None:
        rows = rows[:limit]
    fmt = (lambda v: v) if ndigits is None else (lambda v: None if v is None else round(v, ndigits))
    present = [v for _, v in rows if v is not None]
    return {
        "count": len(rows),
        "total": fmt(sum(present)) if present else None,
        "rows": [{"name": n, key: fmt(v)} for n, v in rows],
    }


def _check_projection(projection):
    if projection not in AREA_CRS:
        raise ValueError(f"projection must be one of {list(AREA_CRS)}")


# ---------- tools ----------

def list_regions():
    """Every row in the dataset as {"name", "code"}."""
    return [{"name": n, "code": _CODE[n]} for n in _NAMES]


def get_population(names, greater_than=None, less_than=None, order=None, limit=None):
    """Census Bureau population estimate for July 1, 2025. None where the source has no estimate."""
    names = _resolve(names)
    return _select({n: _POP[n] for n in names}, "population", greater_than, less_than, order, limit)


def get_area(names, projection="EPSG:3857", greater_than=None, less_than=None, order=None, limit=None):
    """Area in km² of each region's boundary, measured in the given projection."""
    _check_projection(projection)
    names = _resolve(names)
    return _select({n: _AREA[projection][n] for n in names}, "area_km2",
                   greater_than, less_than, order, limit, ndigits=1)


def get_density(names, projection="EPSG:3857", greater_than=None, less_than=None, order=None, limit=None):
    """Population ÷ area (people per km²), with area measured in the given projection."""
    _check_projection(projection)
    names = _resolve(names)
    dens = {n: (None if _POP[n] is None else _POP[n] / _AREA[projection][n]) for n in names}
    return _select(dens, "people_per_km2", greater_than, less_than, order, limit, ndigits=2)


def get_centroid(name):
    """Centroid of a region's boundary as {"lon", "lat"}."""
    _resolve([name])
    lon, lat = _CENTROID[name]
    return {"lon": round(lon, 4), "lat": round(lat, 4)}


def distance(from_name, to_names, method="planar", projection="EPSG:3857",
             greater_than=None, less_than=None, order=None, limit=None):
    """Centroid-to-centroid distance in km from one region to others."""
    if method not in ("planar", "geodesic"):
        raise ValueError('method must be "planar" or "geodesic"')
    if method == "planar":
        _check_projection(projection)
    _resolve([from_name])
    to_names = [n for n in _resolve(to_names) if n != from_name]  # never the region itself
    lon0, lat0 = _CENTROID[from_name]
    if method == "geodesic":
        d = {n: _GEOD.inv(lon0, lat0, *_CENTROID[n])[2] / 1000 for n in to_names}
    else:
        pts = gpd.GeoSeries.from_xy([lon0] + [_CENTROID[n][0] for n in to_names],
                                    [lat0] + [_CENTROID[n][1] for n in to_names],
                                    crs="EPSG:4326").to_crs(projection)
        d = {n: pts.iloc[0].distance(p) / 1000 for n, p in zip(to_names, pts.iloc[1:])}
    return _select(d, "distance_km", greater_than, less_than, order, limit, ndigits=1)


def rank(values, order="desc", k=None):
    """Sort a {name: number} dict. Entries whose value is None are left out."""
    if order not in ("asc", "desc"):
        raise ValueError('order must be "asc" or "desc"')
    items = sorted(((n, v) for n, v in values.items() if v is not None),
                   key=lambda x: x[1], reverse=(order == "desc"))
    return items if k is None else items[:k]


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculator(expression):
    """Evaluate an arithmetic expression with numbers and + - * / ** ( )."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("Only numbers and + - * / ** ( ) are allowed")
    return ev(ast.parse(expression, mode="eval").body)


TOOLS = {f.__name__: f for f in (list_regions, get_population, get_area, get_density, get_centroid,
                                  distance, rank, calculator)}

# ---------- descriptions shown to the task agent and the memory agent ----------

_NAMES_PARAM = {
    "description": 'List of region names exactly as list_regions returns them, or the string "all" for every row.',
    "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string", "enum": ["all"]}],
}


def _filters(unit):
    return {
        "greater_than": {"type": "number", "description": f"Keep only rows whose value is strictly greater than this ({unit})."},
        "less_than": {"type": "number", "description": f"Keep only rows whose value is strictly less than this ({unit})."},
        "order": {"type": "string", "enum": ["asc", "desc"], "description": "Sort rows by value, ascending or descending."},
        "limit": {"type": "integer", "description": "Return at most this many rows, after sorting."},
    }


_RETURNS = ('Returns {"count", "total", "rows"}: count and total are over the returned rows. '
            "Rows without a value are left out when greater_than, less_than or order is used.")
_PROJECTION = {"type": "string", "enum": list(AREA_CRS), "default": "EPSG:3857",
               "description": "EPSG:3857 (Web Mercator) or EPSG:5070 (Albers Equal Area Conic, USA). Default: EPSG:3857."}

TOOL_SPECS = [
    {"name": "list_regions",
     "description": "List every row in the dataset (U.S. states and other areas). Returns [{name, code}].",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_population",
     "description": ("Census Bureau population estimate for July 1, 2025. "
                     "The value is null for regions the source does not cover. " + _RETURNS),
     "parameters": {"type": "object", "properties": {"names": _NAMES_PARAM, **_filters("people")},
                    "required": ["names"]}},
    {"name": "get_area",
     "description": ("Area in km² of each region's boundary. The boundary includes inland and coastal water "
                     "inside the state line. Area is measured in the given map projection; default EPSG:3857. "
                     + _RETURNS),
     "parameters": {"type": "object", "properties": {"names": _NAMES_PARAM, "projection": _PROJECTION,
                                                     **_filters("km²")},
                    "required": ["names"]}},
    {"name": "get_density",
     "description": ("Population density in people per km²: the population estimate divided by the area of the "
                     "boundary, with area measured in the given map projection; default EPSG:3857. " + _RETURNS),
     "parameters": {"type": "object", "properties": {"names": _NAMES_PARAM, "projection": _PROJECTION,
                                                     **_filters("people per km²")},
                    "required": ["names"]}},
    {"name": "get_centroid",
     "description": "Centroid of a region's boundary as {lon, lat} in degrees.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "distance",
     "description": ("Distance in km from the centroid of one region to the centroids of others (the region itself is skipped). "
                     'method="planar" measures a straight line in the given projection; '
                     'method="geodesic" measures along the earth\'s surface and ignores projection. '
                     "Defaults: method=planar, projection=EPSG:3857. " + _RETURNS),
     "parameters": {"type": "object", "properties": {
         "from_name": {"type": "string"},
         "to_names": _NAMES_PARAM,
         "method": {"type": "string", "enum": ["planar", "geodesic"], "default": "planar"},
         "projection": _PROJECTION,
         **_filters("km")},
         "required": ["from_name", "to_names"]}},
    {"name": "rank",
     "description": "Sort a {name: number} object. Entries with null values are left out. Returns [[name, value], ...].",
     "parameters": {"type": "object", "properties": {
         "values": {"type": "object", "additionalProperties": {"type": ["number", "null"]}},
         "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
         "k": {"type": "integer", "description": "Return only the first k entries."}},
         "required": ["values"]}},
    {"name": "calculator",
     "description": "Evaluate an arithmetic expression with numbers and + - * / ** ( ).",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]}},
]
