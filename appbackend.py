# ---------------------------- Data Location logic ----------------------------
import json as _json
from urllib.parse import quote as _q
import re
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
from geopandas import GeoDataFrame
from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from pydantic import BaseModel
from fastapi import HTTPException
from shapely.geometry import mapping
from shapely.ops import transform as _shp_transform
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None  # We’ll handle gracefully.

# NEW: local reprojection
from pyproj import CRS, Transformer

# --- networking: single session with retries ---------------------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=(429, 502, 503, 504),
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)
_session.mount("http://", HTTPAdapter(max_retries=_retry))
_session.mount("https://", HTTPAdapter(max_retries=_retry))

YAML_DEFAULT_PATH = os.environ.get("EXTRA_DATA_LOCATIONS_YAML", "config/data_locations.yml")

# --- simple print-based "logger" --------------------------------------------
def _print_log(level, *args):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {level.upper()}:", *args)

class PrintLogger:
    def info(self, *args): _print_log("info", *args)
    def debug(self, *args): _print_log("debug", *args)
    def warning(self, *args): _print_log("warn", *args)
    def error(self, *args): _print_log("error", *args)

logger = PrintLogger()

class ClearRequest(BaseModel):
    task_name: str

# --------------------- ARC GIS UPDATE ---------------------

def _json_default(obj):
    # numpy → python scalars
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    # pandas / datetime → ISO8601
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        # Strip tz to plain ISO8601 (ArcGIS is picky); adjust if you need tz-aware
        return obj.isoformat()
    # pandas NaT
    try:
        if obj is pd.NaT:
            return None
    except Exception:
        pass
    return str(obj)

def json_dumps_safe(obj, **kwargs) -> str:
    """Always use this when serializing anything that might contain pandas/numpy/datetime."""
    return _json.dumps(obj, default=_json_default, **kwargs)

def dataframe_records_json_safe(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "[]"
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            try:
                # drop tz if present, then ISO
                if getattr(df2[col].dt, "tz", None) is not None:
                    df2[col] = df2[col].dt.tz_convert(None)
                df2[col] = df2[col].dt.tz_localize(None).dt.isoformat()
            except Exception:
                df2[col] = df2[col].astype(str)
    return json_dumps_safe(df2.to_dict(orient="records"))


def _sanitize_value(v):
    # NaNs/NaT → None
    try:
        import pandas as pd
        if v is pd.NaT:
            return None
    except Exception:
        pass
    try:
        if hasattr(np, "isnat") and isinstance(v, (np.datetime64,)):
            if np.isnat(v):
                return None
    except Exception:
        pass

    # Fast path for common types via existing default
    try:
        return _json.loads(_json.dumps(v, default=_json_default))
    except Exception:
        # Last resort: drop non-serializable
        return None

def _sanitize_attributes(row, allowed_fields=None):
    """
    Build an attribute dict from a pandas/GeoPandas row that is JSON-safe.
    - If allowed_fields is provided, only keep those columns.
    - Timestamp/NaT → ISO 8601 / None
    - Anything still not serializable → None
    """
    cols = (list(allowed_fields) if allowed_fields else list(row.index))
    out = {}
    for c in cols:
        if c in row.index:
            out[c] = _sanitize_value(row[c])
    return out


def _looks_like_url(u: str) -> bool:
    if not isinstance(u, str): return False
    s = u.strip()
    if not s.startswith("http"): return False
    # reject common trailing punctuation that gets copy/pasted accidentally
    if s.endswith((".", ",", ";", ":", ")", "]")): return False
    return True

async def trigger_cleanup(task_name: str):
    logger.info(f"Starting cleanup for task: {task_name}")
    try:
        attrs = get_project_urls(task_name)
        logger.debug(f"Project URLs fetched for {task_name}: {list(attrs.keys())}")

        target_keys = ["CHAT_OUTPUT_POINT", "CHAT_OUTPUT_LINE", "CHAT_OUTPUT_POLYGON"]
        urls = [get_attr(attrs, k) for k in target_keys]
        logger.debug(f"Found specific output URLs: {urls}")
        
        # Fallback: legacy single output
        legacy = get_attr(attrs, "CHAT_OUTPUT")
        if legacy:
            urls.append(legacy)
            logger.debug(f"Included legacy output URL: {legacy}")

        cleaned_any = False
        
        # Filter out None/empty URLs
        valid_urls = [u for u in urls if u]
        if not valid_urls:
            logger.info("No valid CHAT_OUTPUT URLs found for cleanup.")
            return {"status": "success", "message": "Nothing to delete."}

        for target_url in valid_urls:
            logger.info(f"Attempting cleanup for URL: {target_url}")
            try:
                if not _looks_like_url(target_url):
                    logger.error(f"Suspicious or malformed URL: {target_url}")
                    continue

                layer = _sanitise_layer_url(target_url)  # -> .../FeatureServer/<id>
                query_url  = f"{layer}/query"
                delete_url = f"{layer}/deleteFeatures"
                logger.debug(f"Sanitised layer URL: {layer}")

                # Step 1: Query for IDs
                params = {"where": "1=1", "returnIdsOnly": "true", "f": "json"}
                r = _session.get(query_url, params=params, timeout=20)
                r.raise_for_status()
                if "json" not in r.headers.get("Content-Type", ""):
                    logger.error(f"Non-JSON content for ID query: {r.headers.get('Content-Type')} … {r.text[:200]}")
                    continue
                response_json = r.json()
                ids = response_json.get("objectIds", [])
                
                if "error" in response_json:
                    logger.error(f"ArcGIS Error during query for IDs at {query_url}: {response_json['error']}")
                    continue

                if not ids:
                    logger.info(f"No features (ids) found to delete at {layer}.")
                    continue

                logger.info(f"Found {len(ids)} features to delete at {layer}.")

                # Step 2: Delete features
                del_params = {"objectIds": ",".join(map(str, ids)), "f": "json"}
                dr = _session.post(delete_url, data=del_params, timeout=60)
                dr.raise_for_status()
                if "json" not in dr.headers.get("Content-Type", ""):
                    logger.error(f"Non-JSON content for delete: {dr.headers.get('Content-Type')} … {dr.text[:200]}")
                    continue
                delete_response_json = dr.json()
                
                if delete_response_json.get("deleteResults", [{}])[0].get("success", False):
                    logger.info(f"Successfully deleted {len(ids)} features from {layer}.")
                    cleaned_any = True
                else:
                    logger.error(f"ArcGIS Error during delete features at {delete_url}: {delete_response_json}")

            except requests.RequestException as req_e:
                logger.error(f"Network/HTTP Error during cleanup of {target_url}: {req_e}")
            except Exception as e:
                logger.error(f"Error during cleanup of {target_url}: {e}")
                
        logger.info(f"Cleanup finished for task {task_name}.")
        return {"status": "success", "message": "Cleanup completed." if cleaned_any else "Nothing to delete."}
    except Exception as e:
        logger.error(f"Top-level cleanup failure for {task_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

def shapely_to_arcgis_geometry(geom, wkid: int):
    logger.debug(f"Converting geometry type {geom.geom_type} to ArcGIS JSON with WKID {wkid}")
    gt = geom.geom_type
    coords = mapping(geom)["coordinates"]
    sr = {"spatialReference": {"wkid": wkid}}

    if gt == "Point":
        x, y = coords
        return {"x": x, "y": y, **sr}
    if gt == "MultiPoint":
        return {"points": [list(pt) for pt in coords], **sr}
    if gt == "LineString":
        return {"paths": [coords], **sr}
    if gt == "MultiLineString":
        return {"paths": [path for path in coords], **sr}
    if gt == "Polygon":
        return {"rings": coords, **sr}
    if gt == "MultiPolygon":
        rings = [ring for polygon in coords for ring in polygon]
        return {"rings": rings, **sr}

    logger.error(f"Unsupported geometry type encountered: {gt}")
    raise ValueError(f"Unsupported geometry type: {gt}")

def get_project_urls(project_name):
    logger.info(f"Fetching project URLs for '{project_name}'.")
    query_url = "https://services-eu1.arcgis.com/8uHkpVrXUjYCyrO4/arcgis/rest/services/Project_index/FeatureServer/0/query"
    
    fields = [
        "PROJECT_NAME",
        "ORTHOMOSAIC",
        "TREE_CROWNS",
        "TREE_TOPS",
        "PREDICTION",
        "CHAT_OUTPUT",
        "USER_CROWNS",
        "USER_TOPS",
        "SURVEY_DATE",
        "CrownSketch",
        "CrownSketch_Predictions",
        "CHAT_INPUT",
        "CHAT_OUTPUT_POINT ",
        "CHAT_OUTPUT_POLYGON",
        "CHAT_OUTPUT_LINE "
    ]
    
    params = {
        "where": f"PROJECT_NAME = '{project_name}'",
        "outFields": ",".join(fields),
        "f": "json",
    }
    logger.debug(f"Querying project index with WHERE clause: {params['where']}")

    try:
        response = _session.get(query_url, params=params, timeout=20)
        response.raise_for_status()
        if "json" not in response.headers.get("Content-Type", ""):
            logger.error(f"Non-JSON response for project index: {response.headers.get('Content-Type')} … {response.text[:200]}")
            raise ValueError("Non-JSON response from project index.")

        data = response.json()

        if "error" in data:
            logger.error(f"ArcGIS Error fetching project index: {data['error']}")
            raise ValueError(f"ArcGIS Error for project '{project_name}': {data['error']['message']}")
            
        if not data.get("features"):
            logger.warning(f"No project found with the name '{project_name}'.")
            raise ValueError(f"No project found with the name '{project_name}'.")

        attributes = data["features"][0]["attributes"]
        logger.info(f"Successfully fetched attributes for '{project_name}'.")
        return attributes
    except requests.RequestException as e:
        logger.error(f"Network/HTTP Error fetching project index for '{project_name}': {e}")
        raise ValueError(f"Network error fetching project index: {e}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_project_urls for '{project_name}': {e}")
        raise ValueError(f"Unexpected error fetching project data: {e}")

def _sanitise_layer_url(url: str) -> str:
    logger.debug(f"Sanitising URL: {url}")
    if not isinstance(url, str) or not url.strip():
        logger.error("URL provided for sanitisation is empty or not a string.")
        raise ValueError("URL must be a non-empty string")

    if not _looks_like_url(url):
        raise ValueError(f"Suspicious URL (trailing punctuation or malformed): {url}")

    u = url.strip().rstrip("/")
    u = re.sub(r"/query$", "", u, flags=re.IGNORECASE)

    m = re.search(r"(.*?/(?:FeatureServer|MapServer))(?:/(\d+))?$", u, flags=re.IGNORECASE)
    if not m:
        logger.error(f"URL does not appear to be a valid ArcGIS service: {url}")
        raise ValueError(f"Not a valid ArcGIS service URL: {url}")

    base, layer = m.groups()
    if layer is None:
        layer = "0"

    result = f"{base}/{layer}"
    logger.debug(f"Sanitised result: {result}")
    return result

def _get_layer_max_record_count(layer_url: str, timeout: int = 20) -> int:
    logger.debug(f"Attempting to fetch maxRecordCount for {layer_url}")
    default_mrc = 1000
    try:
        r = _session.get(f"{layer_url}?f=json", timeout=timeout)
        r.raise_for_status()
        if "json" not in r.headers.get("Content-Type", ""):
            logger.warning(f"Non-JSON metadata for MRC: {r.headers.get('Content-Type')}. Using default {default_mrc}.")
            return default_mrc
        meta = r.json()
        
        if "error" in meta:
            logger.warning(f"ArcGIS Error getting metadata for MRC at {layer_url}: {meta['error']['message']}. Using default {default_mrc}.")
            return default_mrc
            
        mrc = meta.get("maxRecordCount")
        if isinstance(mrc, int) and mrc > 0:
            logger.debug(f"Found maxRecordCount: {mrc}")
            return mrc
        logger.debug(f"maxRecordCount not found or invalid in metadata. Using default {default_mrc}.")
    except requests.RequestException as e:
        logger.warning(f"Network/HTTP Error fetching MRC for {layer_url}: {e}. Using default {default_mrc}.")
    except Exception as e:
        logger.warning(f"Unexpected error getting MRC for {layer_url}: {e}. Using default {default_mrc}.")
    return default_mrc

def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize a GeoDataFrame to EPSG:4326 (WGS84)."""
    if gdf is None or gdf.empty:
        return gdf
    if not gdf.crs:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def extract_geojson(url: str, where: str = "1=1", out_fields: str = "*", timeout: int = 30) -> gpd.GeoDataFrame | None:
    logger.info(f"Starting GeoJSON extraction from {url} with WHERE: {where}")
    try:
        if not _looks_like_url(url):
            logger.error(f"Suspicious URL for extract_geojson: {url}")
            return None

        layer_url = _sanitise_layer_url(url)
        mrc = _get_layer_max_record_count(layer_url, timeout=timeout)
        page_size = min(mrc if isinstance(mrc, int) and mrc > 0 else 1000, 500)  # cap at 500
        logger.info(f"Layer: {layer_url}, Page Size: {page_size}")

        features = []
        offset = 0

        while True:
            params = {
                "where": where,
                "outFields": out_fields,
                "f": "geojson",
                "outSR": 4326,                 # force WGS84 for GeoJSON pulls
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
            logger.debug(f"Querying with offset={offset}, count={page_size}")

            resp = _session.get(f"{layer_url}/query", params=params, timeout=timeout)
            resp.raise_for_status()

            ctype = resp.headers.get("Content-Type", "")
            if "json" not in ctype:
                logger.error(f"Non-JSON response from service ({ctype}). First 200 chars: {resp.text[:200]}")
                return None

            try:
                payload = resp.json()
            except Exception as je:
                logger.error(f"JSON parse failure (likely truncated). Got {len(resp.content)} bytes. Error: {je}")
                return None

            if "error" in payload:
                err = payload.get("error", {})
                msg = err.get("message") or "Unexpected error in ArcGIS response."
                logger.error(f"ArcGIS error in GeoJSON query for {layer_url}: {msg}")
                return None

            fc_features = payload.get("features", [])
            if not isinstance(fc_features, list):
                logger.error(f"Unexpected response structure: 'features' is not a list. Payload keys: {list(payload.keys())}")
                return None

            features.extend(fc_features)
            logger.debug(f"Fetched {len(fc_features)} features. Total collected: {len(features)}")

            if len(fc_features) < page_size:
                logger.debug("Last page reached. Breaking loop.")
                break

            offset += page_size
            logger.debug(f"Moving to next page, new offset: {offset}")

        if not features:
            logger.info("No features found. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Build GeoDataFrame; normalize to WGS84
        gdf = gpd.GeoDataFrame.from_features(features)
        logger.info(f"Successfully created GeoDataFrame with {len(gdf)} features.")
        gdf = _ensure_wgs84(gdf)
        return gdf

    except requests.HTTPError as he:
        logger.error(f"HTTP error fetching GeoJSON: {he} for URL {url}")
        return None
    except requests.Timeout:
        logger.error(f"GeoJSON fetch error: request timed out for URL {url}")
        return None
    except Exception as e:
        logger.error(f"GeoJSON fetch error for URL {url}: {e}")
        return None

def get_roi_gdf(project_name: str) -> gpd.GeoDataFrame:
    logger.info(f"Fetching ROI GeoDataFrame for project: {project_name}")
    try:
        attrs = get_project_urls(project_name)
        chat_input_url = get_attr(attrs, "CHAT_INPUT")
        if not chat_input_url:
            logger.warning("CHAT_INPUT URL is missing. Returning empty GDF.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            
        logger.info(f"Using CHAT_INPUT URL: {chat_input_url}")
        gdf = extract_geojson(chat_input_url)
        
        if gdf is None:
            logger.warning("extract_geojson returned None. Returning empty GDF.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        logger.info(f"Successfully fetched ROI GDF with {len(gdf)} features, CRS: {gdf.crs}")
        return gdf
    except Exception as e:
        logger.error(f"get_roi_gdf error for {project_name}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def sanitise_add_url(target_url: str) -> str:
    """
    Ensure the ArcGIS FeatureServer URL ends with /<layerId>/addFeatures.
    Compatible with Python 3.8+ (no str.removesuffix).
    """
    logger.debug(f"Sanitising add URL: {target_url}")
    target_url = target_url.rstrip("/")
    if target_url.endswith("/addFeatures"):
        target_url = target_url[: -len("/addFeatures")]

    if target_url.endswith("/FeatureServer"):
        result = f"{target_url}/0/addFeatures"
        logger.debug(f"Sanitised result (default layer 0): {result}")
        return result

    result = f"{target_url}/addFeatures"
    logger.debug(f"Sanitised result (appended addFeatures): {result}")
    return result

def _ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    """Ensure the GeoDataFrame is in the specified EPSG."""
    if gdf is None or gdf.empty:
        return gdf
    if not gdf.crs:
        # Downloads are normalized to WGS84; if missing, assume 4326.
        gdf = gdf.set_crs(epsg=4326)
    if gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg=epsg)
    return gdf

def post_features_to_layer(gdf, target_url, project_name, batch_size=800):
    logger.info(f"Starting feature post to {target_url} for project {project_name} with batch size {batch_size}")
    import math

    # Nested helper functions for clarity
    def _is_finite_number(x):
        return isinstance(x, (int, float)) and math.isfinite(x)

    def _has_only_finite_coords(value):
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                return _is_finite_number(value["x"]) and _is_finite_number(value["y"])
            return all(_has_only_finite_coords(v) for v in value.values())
        elif isinstance(value, (list, tuple)):
            return all(_has_only_finite_coords(v) for v in value)
        else:
            if isinstance(value, (int, float)):
                return math.isfinite(value)
            return True

    def sanitize_arcgis_geometry(arcgis_geom):
        if not arcgis_geom:
            return None
        return arcgis_geom if _has_only_finite_coords(arcgis_geom) else None

    add_url = sanitise_add_url(target_url)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    allowed_fields = {"Health", "Tree_ID", "Species","Health_Level"}

    if gdf is None or gdf.empty:
        logger.warning("Nothing to push: input GeoDataFrame is None/empty.")
        return

    # Detect target layer WKID (use sanitised layer URL for accurate metadata)
    layer_url = _sanitise_layer_url(target_url)
    layer_wkid = fetch_crs(layer_url, default_wkid=3857)
    logger.info(f"Target layer WKID detected: {layer_wkid}")
    logger.debug(f"Input GDF CRS: {gdf.crs}")

    # Reproject entire GDF to layer CRS before batching (ensures coordinates match layer)
    gdf = _ensure_crs(gdf, layer_wkid)

    # Batch post
    for start in range(0, len(gdf), batch_size):
        end = start + batch_size
        batch_gdf = gdf.iloc[start:end]
        features = []
        logger.info(f"Processing batch {start//batch_size + 1} ({start} to {end-1})")

        invalid_geom_count = 0
        for index, row in batch_gdf.iterrows():
            try:
                arcgis_geom = shapely_to_arcgis_geometry(row.geometry, wkid=layer_wkid)
                arcgis_geom_sanitized = sanitize_arcgis_geometry(arcgis_geom)
                
                attributes = _sanitize_attributes(row, allowed_fields)

                if arcgis_geom_sanitized is None:
                    invalid_geom_count += 1
                    logger.debug(f"Row {index} has non-finite/invalid coords; sending geometry=None.")
                
                features.append({
                    "geometry": arcgis_geom_sanitized,
                    "attributes": attributes
                })
                
            except Exception as e:
                logger.error(f"Skipping row {index} due to geometry error: {e}")

        if not features:
            logger.warning(f"Batch {start//batch_size + 1} yielded no valid features. Skipping POST.")
            continue
        
        if invalid_geom_count > 0:
            logger.warning(f"Batch {start//batch_size + 1}: {invalid_geom_count} features sent with geometry=None due to invalid coordinates.")

        payload = {
            "features": _json.dumps(features, default=_json_default),
            "f": "json"
        }

        try:
            response = _session.post(add_url, data=payload, headers=headers, timeout=90)
            response.raise_for_status()
            if "json" not in response.headers.get("Content-Type", ""):
                logger.error(f"Non-JSON addFeatures response: {response.headers.get('Content-Type')} … {response.text[:200]}")
                continue
            response_json = response.json()
            
            if response_json.get("addResults") and all(r.get("success") for r in response_json["addResults"]):
                logger.info(f"Batch {start//batch_size + 1} features added successfully.")
            elif "error" in response_json:
                logger.error(f"ArcGIS Error adding batch {start//batch_size + 1} features: {response_json['error']}")
            else:
                logger.warning(f"Partial success or unexpected response for batch {start//batch_size + 1}: {response_json}")

        except requests.RequestException as e:
            logger.error(f"POST to {add_url} for batch {start//batch_size + 1} failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during POST for batch {start//batch_size + 1}: {e}")

def delete_all_features(target_url):
    logger.info(f"Attempting to delete all features from {target_url}")
    try:
        if not _looks_like_url(target_url):
            logger.error(f"Suspicious URL for delete_all_features: {target_url}")
            return

        layer = _sanitise_layer_url(target_url)    # -> .../FeatureServer/<id>
        query_url  = f"{layer}/query"
        delete_url = f"{layer}/deleteFeatures"
        logger.debug(f"Layer URL for delete: {layer}")

        # Step 1: Get all existing OBJECTIDs
        params = {
            "where": "1=1",
            "returnIdsOnly": "true",
            "f": "json"
        }
        response = _session.get(query_url, params=params, timeout=20)
        response.raise_for_status()
        if "json" not in response.headers.get("Content-Type", ""):
            logger.error(f"Non-JSON response for ID query: {response.headers.get('Content-Type')} … {response.text[:200]}")
            return
        data = response.json()

        if "error" in data:
            logger.error(f"ArcGIS Error querying IDs for deletion: {data['error']}")
            return

        object_ids = data.get("objectIds", [])
        if not object_ids:
            logger.info("No features to delete.")
            return

        logger.info(f"Found {len(object_ids)} existing features to delete.")

        # Step 2: Delete by OBJECTIDs
        delete_params = {
            "objectIds": ",".join(map(str, object_ids)),
            "f": "json"
        }
        delete_response = _session.post(delete_url, data=delete_params, timeout=60)
        delete_response.raise_for_status()
        if "json" not in delete_response.headers.get("Content-Type", ""):
            logger.error(f"Non-JSON delete response: {delete_response.headers.get('Content-Type')} … {delete_response.text[:200]}")
            return
        delete_data = delete_response.json()
        
        if "error" in delete_data:
            logger.error(f"ArcGIS Error during feature deletion: {delete_data['error']}")
        else:
            success_count = sum(1 for r in delete_data.get("deleteResults", []) if r.get("success"))
            logger.info(f"Successfully deleted {success_count} features. Delete response summary: {delete_data}")

    except requests.RequestException as e:
        logger.error(f"Network/HTTP Error during delete_all_features for {target_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in delete_all_features for {target_url}: {e}")


# --- CRS helpers --------------------------------------------------------------

def _wkid_to_epsg(wkid: int | None) -> int | None:
    """Map common ESRI wkids to EPSG codes."""
    if wkid is None:
        return None
    # Web Mercator family
    if wkid in (102100, 102113, 3857):
        return 3857
    # WGS84
    if wkid == 4326:
        return 4326
    # OSGB36 / British National Grid
    if wkid in (27700,):
        return 27700
    # Many services already publish EPSG-aligned wkids; fall back to as-is
    return wkid

def _get_layer_epsg(layer_url: str) -> int:
    """Fetch target layer EPSG from ArcGIS REST."""
    try:
        url = layer_url.rstrip("/") + "?f=json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        sr = None
        # Try extent.spatialReference first, then top-level spatialReference
        if isinstance(js, dict):
            sr = (
                (js.get("extent") or {}).get("spatialReference")
                or js.get("sourceSpatialReference")
                or js.get("spatialReference")
            )
        wkid = None
        if isinstance(sr, dict):
            wkid = sr.get("latestWkid") or sr.get("wkid")
        epsg = _wkid_to_epsg(wkid)
        if not epsg:
            logger.warning(f"Could not determine EPSG for {layer_url}; defaulting to EPSG:4326.")
            return 4326
        return int(epsg)
    except Exception as e:
        logger.warning(f"Failed to read layer CRS for {layer_url}: {e}. Defaulting to EPSG:4326.")
        return 4326

def _drop_zm(geom):
    """Return 2D geometry (strip Z/M) for any Shapely geometry."""
    if geom is None:
        return None
    try:
        # shapely 1.x/2.x compatible transform to 2D
        return _shp_transform(lambda x, y, *args: (x, y), geom)
    except Exception:
        return geom

def _reproject_for_layer(gdf, target_layer_url):
    """Ensure GDF is 2D and in the same EPSG as the target layer."""
    if gdf is None or gdf.empty:
        return gdf
    target_epsg = _get_layer_epsg(target_layer_url)

    # If CRS is missing, assume WGS84 (most common when coming from GeoJSON)
    if gdf.crs is None:
        logger.warning("Input GDF has no CRS; assuming EPSG:4326 before reprojection.")
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    # Reproject if needed
    try:
        current_epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    except Exception:
        current_epsg = None

    if current_epsg != target_epsg:
        logger.info(f"Reprojecting {len(gdf)} feature(s): {current_epsg} → {target_epsg} for {target_layer_url}")
        gdf = gdf.to_crs(epsg=target_epsg)

    # Drop Z/M to avoid issues on 2D layers
    if getattr(gdf.geometry.values[0], "has_z", False):
        logger.info("Stripping Z/M from geometries for compatibility with target layer.")
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.apply(_drop_zm)
    return gdf

# --- Main filter() with CRS normalisation ------------------------------------

def filter(gdf_or_fids, project_name):
    print(f"Starting filter function for project: {project_name}")
    
    try:
        attrs = get_project_urls(project_name)
        tree_crowns_url = get_attr(attrs, "TREE_CROWNS")
        if not tree_crowns_url:
            logger.error("TREE_CROWNS URL missing in Project Index.")
            raise ValueError("TREE_CROWNS URL missing in Project Index.")

        # --- Case 1: Direct GDF provided ---
        gdf_direct = to_gdf(gdf_or_fids)
        if gdf_direct is not None and not gdf_direct.empty:
            logger.info(f"Operating in GDF mode (direct push) with {len(gdf_direct)} features.")
            logger.debug(f"Direct GDF CRS: {gdf_direct.crs}, Geometry types: {gdf_direct.geom_type.unique()}")

            # Split by geometry type
            groups = {
                "point":   gdf_direct[gdf_direct.geom_type.isin(["Point", "MultiPoint"])],
                "line":    gdf_direct[gdf_direct.geom_type.isin(["LineString", "MultiLineString"])],
                "polygon": gdf_direct[gdf_direct.geom_type.isin(["Polygon", "MultiPolygon"])],
            }
            
            for kind, sub in groups.items():
                if sub.empty:
                    logger.debug(f"Skipping empty {kind} group.")
                    continue
                    
                # Determine target URL based on geometry kind
                if kind == "point":
                    chat_output_url = get_attr(attrs, "CHAT_OUTPUT_POINT")
                elif kind == "line":
                    chat_output_url = get_attr(attrs, "CHAT_OUTPUT_LINE")
                else:
                    try:
                        chat_output_url = get_attr(attrs, "CHAT_OUTPUT_POLYGON")
                    except:
                        chat_output_url = get_attr(attrs, "CHAT_OUTPUT")  # Fallback to legacy
                
                if not chat_output_url:
                    logger.error(f"Matching CHAT_OUTPUT URL missing for {kind}.")
                    raise ValueError(f"Matching CHAT_OUTPUT URL missing in Project Index for {kind}.")

                # >>> Ensure CRS matches target layer
                sub_fixed = _reproject_for_layer(sub, chat_output_url)

                logger.info(f"Processing {len(sub_fixed)} {kind} feature(s) for URL: {chat_output_url}")
                delete_all_features(chat_output_url)
                post_features_to_layer(sub_fixed, chat_output_url, project_name)
            return

        # --- Case 2: FIDs provided (masking required) ---
        logger.info("Operating in FID mode (fetch + mask).")
        FIDS = ensure_list(gdf_or_fids)
        if not FIDS:
            logger.warning("FIDs list is empty. Nothing to do.")
            return

        logger.info(f"Fetching source crowns from: {tree_crowns_url}")
        gdf = extract_geojson(tree_crowns_url)
        if gdf is None or gdf.empty:
            logger.warning("Source crowns are empty or unavailable.")
            return

        logger.debug(f"Fetched crowns CRS: {gdf.crs}, Columns: {gdf.columns.tolist()}")

        # Pick ID column
        id_column = None
        for candidate in ["Tree_ID", "OBJECTID", "FID", "Id"]:
            if candidate in gdf.columns:
                id_column = candidate
                break
        if not id_column:
            logger.error("No suitable ID column found in source crowns.")
            raise ValueError("No suitable ID column found (Tree_ID, OBJECTID, FID, Id).")

        logger.info(f"Using ID column: {id_column}")
        
        # Apply mask
        mask = normalize_ids(gdf[id_column], FIDS)
        gdf_to_push = gdf[mask]
        
        if gdf_to_push.empty:
            logger.warning("No matching IDs found after filtering source crowns.")
            return

        logger.info(f"Found {len(gdf_to_push)} matching features to push.")

        # Split & push
        split = {
            "point":   gdf_to_push[gdf_to_push.geom_type.isin(["Point", "MultiPoint"])],
            "line":    gdf_to_push[gdf_to_push.geom_type.isin(["LineString", "MultiLineString"])],
            "polygon": gdf_to_push[gdf_to_push.geom_type.isin(["Polygon", "MultiPolygon"])],
        }

        for kind, sub in split.items():
            if sub.empty:
                logger.debug(f"Skipping empty {kind} group from filtered GDF.")
                continue
                
            # Determine target URL
            chat_output_url = (
                get_attr(attrs, "CHAT_OUTPUT_POINT") if kind == "point"
                else get_attr(attrs, "CHAT_OUTPUT_LINE") if kind == "line"
                else get_attr(attrs, "CHAT_OUTPUT_POLYGON")
            )
            if not chat_output_url:
                logger.error(f"Matching CHAT_OUTPUT URL missing for {kind}.")
                raise ValueError(f"Matching CHAT_OUTPUT URL missing for {kind}.")

            # >>> Ensure CRS matches target layer
            sub_fixed = _reproject_for_layer(sub, chat_output_url)

            logger.info(f"Processing {len(sub_fixed)} {kind} feature(s) from FIDs for URL: {chat_output_url}")
            delete_all_features(chat_output_url)
            post_features_to_layer(sub_fixed, chat_output_url, project_name)

    except Exception as e:
        logger.error(f"Top-level error in filter function: {e}")
        raise


# --- Helpers ---------------------------------------------------------------

def get_attr(attrs: dict, key: str):
    logger.debug(f"Getting attribute '{key}'. Available keys: {list(attrs.keys())}")
    # exact
    if key in attrs:  
        return attrs[key]
    # try with a single trailing space
    if (key + " ") in attrs:
        return attrs[key + " "]
    # try stripped keys map
    stripped = {k.strip(): v for k, v in attrs.items()}
    return stripped.get(key)

def to_gdf(maybe_gdf):
    logger.debug(f"Attempting to convert type {type(maybe_gdf)} to GeoDataFrame.")
    try:
        if isinstance(maybe_gdf, gpd.GeoDataFrame):
            logger.debug("Input is already a GeoDataFrame.")
            return maybe_gdf
            
        # Try dict-like GeoJSON
        if isinstance(maybe_gdf, dict) and "type" in maybe_gdf:
            logger.debug("Input is a dict, checking for GeoJSON features.")
            feats = maybe_gdf.get("features")
            if feats:
                logger.info(f"Created GDF from dict with {len(feats)} features.")
                return gpd.GeoDataFrame.from_features(feats)
                
        # Try stringified GeoJSON
        if isinstance(maybe_gdf, str):
            logger.debug("Input is a string, attempting JSON load.")
            s = maybe_gdf.strip()
            if not s:
                logger.debug("Input string is empty after stripping.")
                return None
            try:
                gj = _json.loads(s)
                feats = gj.get("features")
                if feats:
                    logger.info(f"Created GDF from stringified JSON with {len(feats)} features.")
                    return gpd.GeoDataFrame.from_features(feats)
            except _json.JSONDecodeError:
                logger.debug("String is not valid JSON.")
            except Exception as e:
                logger.debug(f"Error parsing JSON features from string: {e}")
                
    except Exception as e:
        logger.error(f"Unexpected error in to_gdf: {e}")
        pass
        
    logger.debug("Input is not GDF-like or conversion failed. Returning None.")
    return None

def normalize_ids(series, ids):
    logger.debug(f"Normalizing IDs: Series type {series.dtype}, IDs count {len(ensure_list(ids))}")
    try:
        # Try int match first (pd.to_numeric works with errors="raise")
        s_int = pd.to_numeric(series, errors="raise").astype("int64")
        ids_int = [int(x) for x in ensure_list(ids)]
        mask = s_int.isin(ids_int)
        logger.debug("Successful integer ID match.")
        return mask
    except Exception:
        logger.debug("Integer ID match failed, falling back to string match.")
        s_str = series.astype(str)
        ids_str = [str(x) for x in ensure_list(ids)]
        mask = s_str.isin(ids_str)
        logger.debug("Successful string ID match.")
        return mask

def geometry_target_key(geom_types: set):
    logger.debug(f"Mapping geometry types: {geom_types} to CHAT_OUTPUT key.")
    if geom_types & {"Point", "MultiPoint"}:
        return "CHAT_OUTPUT_POINT"
    if geom_types & {"LineString", "MultiLineString"}:
        return "CHAT_OUTPUT_LINE"
    if geom_types & {"Polygon", "MultiPolygon"}:
        return "CHAT_OUTPUT_POLYGON"
    logger.error(f"Unsupported geometry types found: {geom_types}")
    raise ValueError(f"Unsupported geometry types found: {geom_types}")

def ensure_list(obj):
    if obj is None:
        return []
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [obj]
    if isinstance(obj, (np.ndarray, pd.Series, list, tuple, set)):
        return list(obj)
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("[") or s.startswith("{"):
            return [s]
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip() != ""]
        return [s]
    return [obj]

# ---------------- AOI reprojection (local, no REST-side reprojection) ---------

def _transform_ring(ring, transformer: Transformer):
    # ring: [[x,y], [x,y], ...]
    return [list(transformer.transform(x, y)) for x, y in ring]

def _reproject_aoi_to_wkid(aoi: dict, target_wkid: int) -> dict:
    """
    Given an AOI dict like:
      {"geometry": { ... }, "geometryType": "...", "inSR": wkid}
    return a new AOI dict with geometry coordinates transformed to target_wkid.
    """
    src_wkid = int(aoi.get("inSR", 4326))
    tgt_wkid = int(target_wkid)
    if src_wkid == tgt_wkid:
        # Ensure spatialReference updated to target in geometry payload
        out = _json.loads(_json.dumps(aoi, default=_json_default))
        geom = out.get("geometry", {})
        # normalize spatialReference
        if isinstance(geom, dict):
            geom["spatialReference"] = {"wkid": tgt_wkid}
        out["inSR"] = tgt_wkid
        return out

    transformer = Transformer.from_crs(CRS.from_epsg(src_wkid), CRS.from_epsg(tgt_wkid), always_xy=True)
    gtype = aoi.get("geometryType")
    geom = aoi.get("geometry", {})
    if not gtype or not isinstance(geom, dict):
        return aoi  # nothing to do

    if gtype == "esriGeometryPolygon":
        rings = geom.get("rings", [])
        new_rings = [_transform_ring(r, transformer) for r in rings]
        return {
            "geometry": {"rings": new_rings, "spatialReference": {"wkid": tgt_wkid}},
            "geometryType": gtype,
            "inSR": tgt_wkid,
        }

    if gtype == "esriGeometryEnvelope":
        # transform the 4 corners and rebuild bbox
        xmin, ymin = geom.get("xmin", 0), geom.get("ymin", 0)
        xmax, ymax = geom.get("xmax", 0), geom.get("ymax", 0)
        corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
        tx = [transformer.transform(x, y) for x, y in corners]
        xs, ys = zip(*tx)
        out_env = {
            "xmin": min(xs),
            "ymin": min(ys),
            "xmax": max(xs),
            "ymax": max(ys),
            "spatialReference": {"wkid": tgt_wkid},
        }
        return {
            "geometry": out_env,
            "geometryType": gtype,
            "inSR": tgt_wkid,
        }

    # If something else, just set the SR to target (best effort)
    geom["spatialReference"] = {"wkid": tgt_wkid}
    return {"geometry": geom, "geometryType": gtype, "inSR": tgt_wkid}

# --- Geometry helpers ---------------------------------------------------------

def _gdf_from_layer_all(layer_url: str, out_wkid: int = 4326, timeout: int = 30) -> gpd.GeoDataFrame:
    """
    Pulls *all* features from a layer as GeoJSON (handles pagination).
    Returns an empty GDF if there are no features.
    NOTE: Uses the provided out_wkid for outSR; used by AOI builder to keep native CRS.
    """
    logger.info(f"Fetching all features from {layer_url} with target WKID {out_wkid}")
    layer_url = _sanitise_layer_url(layer_url)
    mrc = _get_layer_max_record_count(layer_url, timeout=timeout)
    page_size = min(mrc if isinstance(mrc, int) and mrc > 0 else 1000, 500)  # cap
    features = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "outSR": out_wkid,             # honor requested out_wkid here
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        logger.debug(f"Querying all features with offset={offset}, count={page_size}")
        
        try:
            r = _session.get(f"{layer_url}/query", params=params, timeout=timeout)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "json" not in ctype:
                logger.error(f"Non-JSON response fetching all features ({ctype}). First 200 chars: {r.text[:200]}")
                raise RuntimeError("Non-JSON response when fetching all features.")
            payload = r.json()
        except requests.RequestException as e:
            logger.error(f"Network/HTTP error fetching all features from {layer_url}: {e}")
            raise
        except _json.JSONDecodeError as je:
            logger.error(f"Non-JSON response fetching all features: {je}")
            raise RuntimeError("Non-JSON response when fetching all features.") from je
            
        fc = payload.get("features", [])
        
        if "error" in payload:
            err = payload.get("error", {})
            msg = err.get("message") or "Unexpected error in ArcGIS response."
            logger.error(f"ArcGIS error in _gdf_from_layer_all for {layer_url}: {msg}")
            raise RuntimeError(f"ArcGIS error: {msg}")
            
        if not isinstance(fc, list):
            logger.error("Unexpected response structure: 'features' is not a list.")
            raise RuntimeError("Unexpected response structure from service.")
            
        features.extend(fc)
        logger.debug(f"Fetched {len(fc)} features in batch. Total: {len(features)}")
        
        if len(fc) < page_size:
            logger.debug("Last page reached.")
            break
        offset += page_size
        
    if not features:
        logger.info("No features found in layer. Returning empty GDF.")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{out_wkid}")

    gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{out_wkid}")
    logger.info(f"Successfully created GDF with {len(gdf)} features, CRS EPSG:{out_wkid}.")
    return gdf

def _rings_from_shapely(geom) -> list[list[list[float]]]:
    logger.debug(f"Converting Shapely geometry type {geom.geom_type} to Esri 'rings'.")
    if isinstance(geom, Polygon):
        exterior = list(map(list, geom.exterior.coords)) if geom.exterior else []
        holes = [list(map(list, r.coords)) for r in geom.interiors] if geom.interiors else []
        return [exterior] + holes

    if isinstance(geom, MultiPolygon):
        rings = []
        for poly in geom.geoms:
            rings.extend(_rings_from_shapely(poly))
        return rings

    logger.error(f"Unsupported geometry type for rings: {geom.geom_type}")
    raise ValueError(f"Unsupported geometry type for rings: {geom.geom_type}")

def fetch_crs(base_url, timeout=20, default_wkid=4326):
    """
    Fetches the Coordinate Reference System (CRS) WKID from an ArcGIS REST service endpoint.
    (Read-only metadata; we still do all AOI reprojection locally.)
    """
    logger.debug(f"Fetching CRS for base URL: {base_url}. Default WKID: {default_wkid}")
    try:
        r = _session.get(f"{base_url}?f=json", timeout=timeout)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "json" not in ctype:
            logger.warning(f"Non-JSON metadata for CRS: {ctype}. Returning default {default_wkid}.")
            return default_wkid

        metadata = r.json()
        
        if "error" in metadata:
            logger.warning(f"ArcGIS Error getting CRS metadata for {base_url}: {metadata['error']['message']}. Using default {default_wkid}.")
            return default_wkid
            
        spatial_ref = metadata.get("spatialReference")
        if spatial_ref and "wkid" in spatial_ref:
            logger.debug(f"Found CRS at root: {spatial_ref['wkid']}")
            return spatial_ref["wkid"]
            
        tile_info = metadata.get("tileInfo")
        if tile_info:
            tile_spatial_ref = tile_info.get("spatialReference")
            if tile_spatial_ref and "wkid" in tile_spatial_ref:
                logger.debug(f"Found CRS in tileInfo: {tile_spatial_ref['wkid']}")
                return tile_spatial_ref["wkid"]
                
        layers = metadata.get("layers")
        if layers:
            for layer in layers:
                sr = layer.get("extent", {}).get("spatialReference") or layer.get("spatialReference")
                if sr and "wkid" in sr:
                    logger.debug(f"Found CRS in layer extent/spatialReference: {sr['wkid']}")
                    return sr["wkid"]
                    
        logger.warning(f"WKID not found in metadata for {base_url}. Returning default WKID: {default_wkid}")
        return default_wkid
    except requests.RequestException as e:
        logger.warning(f"Network/HTTP error fetching CRS for {base_url}: {e}. Returning default {default_wkid}.")
        return default_wkid
    except Exception as e:
        logger.warning(f"Unexpected error fetching CRS for {base_url}: {e}. Returning default {default_wkid}.")
        return default_wkid

# --- AOI selection: CHAT_INPUT first, fallback to ORTHOMOSAIC extent -----------

def get_project_aoi_geometry(project_name: str):
    logger.info(f"Determining AOI geometry for project: {project_name}")
    attrs = get_project_urls(project_name)

    chat_input_url = (attrs.get("CHAT_INPUT") or "").strip()
    ortho_url = (attrs.get("ORTHOMOSAIC") or "").strip()

    # 1) Try CHAT_INPUT polygons (FeatureServer layer)
    if chat_input_url:
        logger.info(f"Attempting to use CHAT_INPUT URL: {chat_input_url}")
        try:
            layer_url = _sanitise_layer_url(chat_input_url)
            wkid = fetch_crs(layer_url) or 4326
            gdf = _gdf_from_layer_all(layer_url, out_wkid=wkid)

            # Keep only polygonal geometries
            if not gdf.empty:
                polys = [geom for geom in gdf.geometry if geom is not None and geom.geom_type in ("Polygon", "MultiPolygon")]
                if polys:
                    u = unary_union(polys)
                    logger.debug(f"Union result type: {u.geom_type}")
                    
                    if isinstance(u, GeometryCollection):
                        parts = [g for g in u.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                        u = unary_union(parts) if parts else None
                        logger.debug(f"After GeometryCollection split, union result type: {u.geom_type if u else 'None'}")
                        
                    if u and not u.is_empty:
                        if u.geom_type in ("Polygon", "MultiPolygon"):
                            rings = _rings_from_shapely(u)
                            logger.info(f"AOI determined from CHAT_INPUT polygons (WKID: {wkid}).")
                            return {
                                "geometry": {
                                    "rings": rings,
                                    "spatialReference": {"wkid": wkid}
                                },
                                "geometryType": "esriGeometryPolygon",
                                "inSR": wkid
                            }
                        else:
                            logger.warning(f"Union resulted in unexpected geometry type: {u.geom_type}. Falling back.")
                            
            logger.info("CHAT_INPUT had no valid polygons. Falling back to ORTHOMOSAIC extent.")
        except Exception as e:
            logger.warning(f"CHAT_INPUT AOI fallback due to error: {e}")

    # 2) Fallback: ORTHOMOSAIC extent (ImageServer)
    if not ortho_url:
        logger.error("Both CHAT_INPUT and ORTHOMOSAIC are missing for this project.")
        raise ValueError("Both CHAT_INPUT and ORTHOMOSAIC are missing for this project.")

    logger.info(f"Using ORTHOMOSAIC URL: {ortho_url}")
    try:
        base = ortho_url.rstrip("/")
        r = _session.get(f"{base}?f=json", timeout=30)
        r.raise_for_status()
        if "json" not in r.headers.get("Content-Type", ""):
            logger.error(f"Non-JSON ORTHOMOSAIC metadata: {r.headers.get('Content-Type')} … {r.text[:200]}")
            raise RuntimeError("Non-JSON ORTHOMOSAIC metadata")

        meta = r.json()
        
        if "error" in meta:
             logger.error(f"ArcGIS Error getting metadata from ORTHOMOSAIC: {meta['error']}")
             raise RuntimeError(f"ORTHOMOSAIC service error: {meta['error']['message']}")

        extent = meta.get("extent") or meta.get("fullExtent")
        if not extent:
            logger.error("ORTHOMOSAIC service does not expose an extent.")
            raise RuntimeError("ORTHOMOSAIC service does not expose an extent.")

        sr = extent.get("spatialReference") or meta.get("spatialReference") or {}
        wkid = sr.get("wkid") or fetch_crs(base) or 4326
        
        logger.info(f"AOI determined from ORTHOMOSAIC extent (WKID: {wkid}).")

        env = {
            "xmin": extent["xmin"],
            "ymin": extent["ymin"],
            "xmax": extent["xmax"],
            "ymax": extent["ymax"],
            "spatialReference": {"wkid": wkid}
        }

        return {
            "geometry": env,
            "geometryType": "esriGeometryEnvelope",
            "inSR": wkid
        }
    except Exception as e:
        logger.error(f"Error determining AOI from ORTHOMOSAIC: {e}. Falling back to empty envelope.")
        return {
            "geometry": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0, "spatialReference": {"wkid": 4326}},
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326
        }
        
def _build_spatial_query_url(layer_url: str, aoi: dict, where: str = "1=1", out_fields: str = "*") -> str:
    """
    Build a query URL where the AOI geometry is FIRST transformed locally to the target layer's CRS.
    We do NOT pass inSR/outSR; we send coordinates already in layer CRS so the service can consume directly.
    """
    lyr = _sanitise_layer_url(layer_url)
    target_wkid = fetch_crs(lyr) or 4326  # read-only metadata
    # local reprojection of AOI geometry to the layer's CRS
    aoi_proj = _reproject_aoi_to_wkid(aoi, target_wkid)

    geom_json = _json.dumps(aoi_proj["geometry"], separators=(",", ":"), default=_json_default)
    geom = _q(geom_json)

    geometry_type = aoi_proj.get("geometryType", "esriGeometryPolygon")

    return (
        f"{lyr}/query"
        f"?where={_q(where)}"
        f"&geometry={geom}"
        f"&geometryType={_q(geometry_type)}"
        f"&spatialRel=esriSpatialRelIntersects"
        f"&outFields={_q(out_fields)}"
        f"&f=geojson"
    )

def get_aoi_in_layer_crs(project_name: str, layer_url: str) -> dict:
    """
    Returns AOI geometry dict reprojected locally (no REST reprojection)
    to the WKID of the given layer_url.
    """
    aoi = get_project_aoi_geometry(project_name)
    if not layer_url:
        return aoi  # nothing to align to
    lyr = _sanitise_layer_url(layer_url)
    target_wkid = fetch_crs(lyr) or int(aoi.get("inSR", 4326))
    return _reproject_aoi_to_wkid(aoi, target_wkid)

@lru_cache(maxsize=1)
def _load_extra_locations_yaml(path: str) -> Dict[str, Any]:
    """Load and cache the YAML file. Returns {} if file missing or PyYAML not installed."""
    if yaml is None:
        # PyYAML not available; quietly disable YAML-driven extras
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        # Be defensive: bad YAML should not break the app
        return {}

def _supports_pagination(layer_url: str) -> bool:
    try:
        meta = _session.get(_sanitise_layer_url(layer_url) + "?f=json", timeout=20).json()
        return bool(meta.get("advancedQueryCapabilities", {}).get("supportsPagination", False))
    except Exception:
        return False

def _get_objectid_field(layer_url: str) -> str:
    try:
        meta = _session.get(_sanitise_layer_url(layer_url) + "?f=json", timeout=20).json()
        # prefer objectIdField if present
        if meta.get("objectIdField"):
            return meta["objectIdField"]
        # else scan fields
        for f in meta.get("fields", []):
            if f.get("type") == "esriFieldTypeOID":
                return f.get("name")
    except Exception:
        pass
    return "OBJECTID"

def _count_features_in_aoi(layer_url: str, aoi: dict, where: str = "1=1", timeout: int = 30) -> int:
    """Return AOI-filtered count using returnCountOnly=true (AOI already reprojected to layer CRS)."""
    lyr = _sanitise_layer_url(layer_url)
    aoi_proj = _reproject_aoi_to_wkid(aoi, fetch_crs(lyr) or 4326)
    geometry = _json.dumps(aoi_proj["geometry"], default=_json_default, separators=(",", ":"))
    params = {
        "where": where,
        "geometry": geometry,
        "geometryType": aoi_proj.get("geometryType", "esriGeometryPolygon"),
        "spatialRel": "esriSpatialRelIntersects",
        "returnCountOnly": "true",
        "f": "json",
    }
    try:
        r = _session.post(f"{lyr}/query", data=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        return int(js.get("count", 0))
    except Exception:
        return 0

def _paginate_urls_for_aoi(layer_url: str, aoi: dict, out_fields: str = "*", where: str = "1=1") -> list[str]:
    """
    Build one or many AOI-filtered URLs that together return ALL features.
    Uses resultOffset/resultRecordCount if supported; else falls back to OBJECTID windows.
    """
    lyr = _sanitise_layer_url(layer_url)
    wkid = fetch_crs(lyr) or 4326
    aoi_proj = _reproject_aoi_to_wkid(aoi, wkid)

    # layer caps
    mrc = _get_layer_max_record_count(lyr)
    page_size = max(1, min(mrc if isinstance(mrc, int) and mrc > 0 else 2000, 2000))

    total = _count_features_in_aoi(layer_url, aoi_proj, where=where)
    if total <= page_size:
        # single URL (your current behavior) + stable order
        base = _build_spatial_query_url(layer_url, aoi_proj, where=where, out_fields=out_fields)
        return [f"{base}&orderByFields={_q(_get_objectid_field(layer_url) + ' ASC')}"]

    urls: list[str] = []
    if _supports_pagination(layer_url):
        # resultOffset paging
        oid = _get_objectid_field(layer_url)
        for offset in range(0, total, page_size):
            base = _build_spatial_query_url(layer_url, aoi_proj, where=where, out_fields=out_fields)
            page = f"{base}&orderByFields={_q(oid + ' ASC')}&resultRecordCount={page_size}&resultOffset={offset}"
            urls.append(page)
        return urls

    # Fallback: OBJECTID windowing (for very old services)
    oid = _get_objectid_field(layer_url)
    # get all OIDs (IDs only) and window locally
    try:
        # AOI-filtered OIDs
        geom = _json.dumps(aoi_proj["geometry"], default=_json_default, separators=(",", ":"))
        params = {
            "where": where,
            "geometry": geom,
            "geometryType": aoi_proj.get("geometryType", "esriGeometryPolygon"),
            "spatialRel": "esriSpatialRelIntersects",
            "returnIdsOnly": "true",
            "f": "json",
        }
        r = _session.post(f"{lyr}/query", data=params, timeout=40)
        r.raise_for_status()
        ids = r.json().get("objectIds") or []
        ids = sorted(int(x) for x in ids)
    except Exception:
        ids = []

    if not ids:
        # fallback to single URL anyway
        base = _build_spatial_query_url(layer_url, aoi_proj, where=where, out_fields=out_fields)
        return [base]

    # window by contiguous ranges of size page_size
    for i in range(0, len(ids), page_size):
        lo = ids[i]
        hi = ids[min(i + page_size, len(ids)) - 1]
        w = f"{where} AND {oid} BETWEEN {lo} AND {hi}"
        base = _build_spatial_query_url(layer_url, aoi_proj, where=w, out_fields=out_fields)
        urls.append(base)
    return urls



def _get_layer_columns(layer_url: str, timeout: int = 20) -> list[str]:
    """
    Return a list of column/field names for an ArcGIS Feature/MapServer layer.
    Fast path: read metadata (fields[].name) from <layer_url>?f=json
    Fallback: do a tiny GeoJSON query (1 record) and infer columns using GeoPandas.
    Returns [] if not resolvable (e.g., ImageServer or non-feature endpoint).
    """
    try:
        lu = _sanitise_layer_url(layer_url)
    except Exception:
        # Not a Feature/MapServer layer; no columns to show
        return []

    # --- Fast path: metadata fields[] ---
    try:
        r = _session.get(f"{lu}?f=json", timeout=timeout)
        r.raise_for_status()
        if "json" in (r.headers.get("Content-Type") or "").lower():
            meta = r.json()
            if isinstance(meta, dict) and isinstance(meta.get("fields"), list):
                cols = [f.get("name") for f in meta["fields"] if isinstance(f, dict) and f.get("name")]
                # Keep geometry indicator first if present (OBJECTID/GlobalID), then others sorted
                # But avoid over-manipulation; just return in service order (usually stable & meaningful)
                return cols or []
    except Exception:
        pass

    # --- Fallback: 1-row GeoJSON read via GeoPandas ---
    try:
        # Minimal query that most ArcGIS layers accept
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "resultRecordCount": 1,
            "outSR": 4326,
        }
        qurl = f"{lu}/query"
        # Use requests to get a tiny FeatureCollection
        rr = _session.get(qurl, params=params, timeout=timeout)
        rr.raise_for_status()
        if "json" not in (rr.headers.get("Content-Type") or "").lower():
            return []
        fc = rr.json()
        if isinstance(fc, dict) and "features" in fc:
            gdf = gpd.GeoDataFrame.from_features(fc)
            return list(gdf.columns) if not gdf.empty else list((gdf.columns if hasattr(gdf, "columns") else []))
    except Exception:
        pass

    return []


def _format_columns_inline(cols: list[str], max_len: int = 600) -> str:
    """
    Return a single-line 'columns: a, b, c' string, trimmed if very long.
    """
    if not cols:
        return ""
    s = ", ".join(cols)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return f"  columns: {s}"

def _iter_project_extra_entries(project_name: str, yaml_path: str) -> List[Dict[str, Any]]:
    """
    Merge default + project-specific entries. Later (project) entries come after defaults.
    Returns a clean list of dicts with keys: label, url|use_attr, aoifilter(bool), insert_at(optional).
    """
    data = _load_extra_locations_yaml(yaml_path)
    block = (data.get("additional_data_locations") or {}) if isinstance(data, dict) else {}
    default_list = block.get("default") or []
    proj_list = block.get(project_name) or []

    # Normalize to list[dict]
    def _norm(lst: Any) -> List[Dict[str, Any]]:
        if not isinstance(lst, list):
            return []
        cleaned: List[Dict[str, Any]] = []
        for item in lst:
            if not isinstance(item, dict):
                continue
            # minimal validation: must have either 'url' or 'use_attr'
            if not (item.get("url") or item.get("use_attr")):
                continue
            if not item.get("label"):
                continue
            cleaned.append(item)
        return cleaned

    return _norm(default_list) + _norm(proj_list)

def make_project_data_locations(project_name: str, include_seasons: bool, attrs: dict) -> list[str]:
    logger.info(f"Building data locations for project '{project_name}' (include_seasons={include_seasons}).")
    # Always refresh from the project index (keeps behaviour consistent with your original)
    attrs = get_project_urls(project_name)

    # Core project layer
    tree_crowns_url = get_attr(attrs, "TREE_CROWNS")
    data_locations: list[str] = []

    # Helper to add a layer with correct label + AOI in that layer's CRS
    def _add(layer_url: str, label_prefix: str, insert_at: Optional[int] = None):
        if not layer_url:
            return
        try:
            aoi_layer_crs = get_aoi_in_layer_crs(project_name, layer_url)
            label = _label_for_layer(label_prefix, layer_url)
    
            # Build one or many URLs so ALL rows are retrievable
            page_urls = _paginate_urls_for_aoi(layer_url, aoi_layer_crs, out_fields="*")
    
            # Columns once (no need to fetch for every page)
            _cols = _get_layer_columns(layer_url)
            _cols_str = _format_columns_inline(_cols)
    
            for idx, url in enumerate(page_urls, start=1):
                suffix = f" (page {idx}/{len(page_urls)})" if len(page_urls) > 1 else ""
                entry = f"{label}{suffix}: {url}"
                if _cols_str and idx == 1:  # attach columns to first page line only
                    entry += f"\n{_cols_str}"
                if insert_at is None or insert_at < 0:
                    data_locations.append(entry)
                else:
                    pos = min(insert_at, len(data_locations))
                    data_locations.insert(pos, entry)
                    insert_at += 1  # keep subsequent pages in order
        except Exception as ex:
            logger.error(f"Failed to add data location for {label_prefix}: {ex}")



    # Helper to add a raw, non-AOI URL (e.g., docs, tiles, WMS landing pages)
    def _add_raw(raw_url: str, label: str, insert_at: Optional[int] = None):
        if not raw_url:
            return
        try:
            entry = f"{label}: {raw_url}"
            if insert_at is None or insert_at < 0:
                data_locations.append(entry)
            else:
                idx = min(insert_at, len(data_locations))
                data_locations.insert(idx, entry)
            logger.debug(f"Added RAW data location: {entry}")
        except Exception as ex:
            logger.error(f"Failed to add raw data location for {label}: {ex}")

    # Small helper to normalize categories so we can avoid duplicates later
    def _infer_category(label: str, url: Optional[str]) -> str:
        lbl = (label or "").lower()
        u = (url or "").lower()

        # National flags
        is_national = any(k in lbl for k in ["national"]) or any(k in u for k in [
            "os_openroads", "openmap_local", "open_greenspace", "ordnance", "services.arcgis.com/qhlhlqrcvenxjtpr"
        ])

        # Categories
        if any(k in lbl for k in ["building", "buildings"]) or "openmap_local_buildings" in u:
            base = "buildings"
        elif any(k in lbl for k in ["road", "street"]) or "openroads" in u:
            base = "roads"
        elif any(k in lbl for k in ["green space", "greenspace", "open space", "park"]) or "open_greenspace" in u:
            base = "greenspace"
        elif any(k in lbl for k in ["tree points", "arbotrackpoints"]):
            base = "tree_points"
        elif any(k in lbl for k in ["tree polygons", "arbotrackpolygons"]):
            base = "tree_polygons"
        elif any(k in lbl for k in ["operational property"]):
            base = "operational_property"
        elif any(k in lbl for k in ["non operational property", "non-operational property"]):
            base = "non_operational_property"
        else:
            base = lbl.strip() or "unknown"

        return f"{base}::{'national' if is_national else 'local'}"

    # Track which base categories have already been added (prefer first-come, i.e., local with higher priority)
    def _category_already_present(category_key: str) -> bool:
        # We compare only the base (before ::), so later nationals are skipped if a local exists
        base = category_key.split("::", 1)[0]
        for line in data_locations:
            low = line.lower()
            if base == "buildings" and ("building" in low or "buildings" in low):
                return True
            if base == "roads" and ("road" in low or "street" in low):
                return True
            if base == "greenspace" and any(k in low for k in ["green space", "greenspace", "open space", "park"]):
                return True
            if base == "tree_points" and any(k in low for k in ["points geojson", "tree points"]):
                return True
            if base == "tree_polygons" and any(k in low for k in ["polygons geojson", "tree polygons"]):
                return True
            if base == "operational_property" and "operational property" in low:
                return True
            if base == "non_operational_property" and "non operational property" in low:
                return True
        return False

    # Project tree crowns (if present)
    if tree_crowns_url:
        _add(tree_crowns_url, "Tree crowns")

    # Optional seasonal layers — each uses AOI in its OWN CRS and correct geometry label
    if include_seasons:
        logger.info("Including seasonal crown layers.")
        try:
            attrs_summer = get_project_urls("TT_GCW1_Summer")
            attrs_winter = get_project_urls("TT_GCW1_Winter")
            tree_crown_summer = get_attr(attrs_summer, "TREE_CROWNS")
            tree_crown_winter = get_attr(attrs_winter, "TREE_CROWNS")

            if tree_crown_summer:
                _add(tree_crown_summer, "Before storm tree crowns", insert_at=1 if tree_crowns_url else 0)
            else:
                logger.warning("TT_GCW1_Summer TREE_CROWNS URL missing.")

            if tree_crown_winter:
                insert_idx = 2 if (tree_crowns_url and tree_crown_summer) else (1 if (tree_crowns_url or tree_crown_summer) else 0)
                _add(tree_crown_winter, "After storm tree crowns", insert_at=insert_idx)
            else:
                logger.warning("TT_GCW1_Winter TREE_CROWNS URL missing.")
        except ValueError as ve:
            logger.error(f"Could not fetch seasonal project URLs: {ve}")
        except Exception as e:
            logger.error(f"Error including seasonal data locations: {e}")

    # -----------------------------------------------------------------
    # NEW: Append/insert project-specific extras from YAML with PRIORITY & DE-DUP
    # -----------------------------------------------------------------
    extras = _iter_project_extra_entries(project_name, YAML_DEFAULT_PATH)  # pulls default + project blocks

    # Compute explicit or inferred priority
    def _infer_priority(item: dict) -> int:
        if isinstance(item.get("priority"), int):
            return item["priority"]
        lbl = (item.get("label") or "")
        # heuristic: project-specific/local if label mentions project name or doesn't say 'National'
        if project_name.lower() in lbl.lower() or "national" not in lbl.lower():
            return 10
        return 100

    # Attach computed priorities
    for it in extras:
        it["__priority"] = _infer_priority(it)

    # Sort: lower priority number first (locals before nationals)
    extras.sort(key=lambda d: d.get("__priority", 50))

    # Add extras in sorted order, skipping later items whose category already exists
    for item in extras:
        label: str = item.get("label")
        aoifilter: bool = bool(item.get("aoifilter", True))
        insert_at: Optional[int] = item.get("insert_at", None)

        # Source resolution: use_attr takes precedence if provided, else url
        layer_url: Optional[str] = None
        if item.get("use_attr"):
            try:
                layer_url = get_attr(attrs, str(item["use_attr"]))
                if not layer_url:
                    logger.warning(f"YAML extra '{label}': attribute '{item['use_attr']}' not found or empty; skipping.")
                    continue
            except Exception as ex:
                logger.error(f"YAML extra '{label}': error resolving use_attr '{item['use_attr']}': {ex}")
                continue
        else:
            layer_url = item.get("url")

        if not layer_url:
            logger.warning(f"YAML extra '{label}': no usable 'url' or 'use_attr' resolved; skipping.")
            continue

        cat_key = _infer_category(label, layer_url)
        base_key = cat_key.split("::", 1)[0]
        if _category_already_present(cat_key):
            # If a category is already represented (likely by a local, given sorting),
            # skip adding later (usually national) duplicates.
            logger.info(f"Skipping '{label}' ({base_key}) because a {base_key} layer already exists (local preferred).")
            continue

        try:
            if aoifilter:
                _add(layer_url, label, insert_at=insert_at)
            else:
                _add_raw(layer_url, label, insert_at=insert_at)
        except Exception as ex:
            logger.error(f"Failed to apply YAML extra '{label}': {ex}")

    # -----------------------------------------------------------------

    logger.info(f"Final list of {len(data_locations)} data locations generated.")
    for loc in data_locations:
        logger.debug(f"  - {loc}")

    return data_locations



def get_layer_geometry_type(layer_url: str) -> str:
    """
    Returns one of:
      'esriGeometryPoint' | 'esriGeometryPolyline' | 'esriGeometryPolygon'
    Falls back to '' if not resolvable.
    """
    try:
        meta = requests.get(layer_url.rstrip("/") + "?f=json", timeout=30).json()
        return (meta.get("geometryType") or "").strip()
    except Exception:
        logger.exception(f"Failed to get geometryType for {layer_url}")
        return ""

def _label_for_layer(prefix: str, layer_url: str) -> str:
    gtype = get_layer_geometry_type(layer_url).lower()
    if "point" in gtype:
        kind = "Points"
    elif "polyline" in gtype or "line" in gtype:
        kind = "Lines"
    elif "polygon" in gtype:
        kind = "Polygons"
    else:
        kind = "Features"
    return f"{kind} geoJSON: {prefix}"
