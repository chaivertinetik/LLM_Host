# ---------------------------- Data Location logic ----------------------------
import json as _json
from urllib.parse import quote as _q
import re
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
import json
from geopandas import GeoDataFrame
from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from pydantic import BaseModel
from fastapi import HTTPException
from shapely.geometry import mapping

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
    # numpy → python
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # datetime-like → ISO 8601
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    # pandas Timestamp/NaT
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    # safest fallback
    return str(obj)

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
    allowed_fields = {"Health", "Tree_ID", "Species"}

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
                
                attributes = {k: row.get(k, None) for k in allowed_fields if k in batch_gdf.columns}

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
            "features": json.dumps(features, default=_json_default),
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

def filter(gdf_or_fids, project_name):
    logger.info(f"Starting filter function for project: {project_name}")

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
                "point": gdf_direct[gdf_direct.geom_type.isin(["Point", "MultiPoint"])],
                "line": gdf_direct[gdf_direct.geom_type.isin(["LineString", "MultiLineString"])],
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
                        chat_output_url = get_attr(attrs, "CHAT_OUTPUT") # Fallback to legacy
                
                if not chat_output_url:
                    logger.error(f"Matching CHAT_OUTPUT URL missing for {kind}.")
                    raise ValueError(f"Matching CHAT_OUTPUT URL missing in Project Index for {kind}.")

                logger.info(f"Processing {len(sub)} {kind} feature(s) for URL: {chat_output_url}")
                delete_all_features(chat_output_url)
                post_features_to_layer(sub, chat_output_url, project_name)
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
        for kind, sub in {
            "point": gdf_to_push[gdf_to_push.geom_type.isin(["Point", "MultiPoint"])],
            "line": gdf_to_push[gdf_to_push.geom_type.isin(["LineString", "MultiLineString"])],
            "polygon": gdf_to_push[gdf_to_push.geom_type.isin(["Polygon", "MultiPolygon"])],
        }.items():
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
                
            logger.info(f"Processing {len(sub)} {kind} feature(s) from FIDs for URL: {chat_output_url}")
            delete_all_features(chat_output_url)
            post_features_to_layer(sub, chat_output_url, project_name)

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
                gj = json.loads(s)
                feats = gj.get("features")
                if feats:
                    logger.info(f"Created GDF from stringified JSON with {len(feats)} features.")
                    return gpd.GeoDataFrame.from_features(feats)
            except json.JSONDecodeError:
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
        except json.JSONDecodeError as je:
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
    lyr = _sanitise_layer_url(layer_url)
    in_sr = aoi.get("inSR", 4326)
    out_wkid = 4326  # force WGS84 for GeoJSON

    geom_json = _json.dumps(aoi["geometry"], separators=(",", ":"))
    geom = _q(geom_json)

    return (
        f"{lyr}/query"
        f"?where={_q(where)}"
        f"&geometry={geom}"
        f"&geometryType={_q(aoi['geometryType'])}"
        f"&spatialRel=esriSpatialRelIntersects"
        f"&inSR={in_sr}"
        f"&outFields={_q(out_fields)}"
        f"&outSR={out_wkid}"
        f"&f=geojson"
    )

def make_project_data_locations(project_name: str, include_seasons: bool, attrs: dict) -> list[str]:
    logger.info(f"Building data locations for project '{project_name}' (include_seasons={include_seasons}).")
    aoi = get_project_aoi_geometry(project_name)
    logger.debug(f"AOI geometry type: {aoi['geometryType']}, inSR: {aoi['inSR']}")

    # Core project layers
    tree_crowns_url = get_attr(attrs, "TREE_CROWNS")
    if not tree_crowns_url:
        logger.warning("Main TREE_CROWNS URL missing. Will skip.")

    # National context layers (static)
    os_roads = "https://services.arcgis.com/qHLhLQrcvEnxjtPr/arcgis/rest/services/OS_OpenRoads/FeatureServer/1"
    os_buildings = "https://services.arcgis.com/qHLhLQrcvEnxjtPr/arcgis/rest/services/OS_OpenMap_Local_Buildings/FeatureServer/0"
    os_green = "https://services.arcgis.com/qHLhLQrcvEnxjtPr/arcgis/rest/services/OS_Open_Greenspace/FeatureServer/1"

    data_locations = []
    
    if tree_crowns_url:
        data_locations.append(
            "Tree crown geoJSON shape file: " + _build_spatial_query_url(tree_crowns_url, aoi, out_fields='*')
        )

    data_locations.extend([
        "Roads geoJSON shape file: " + _build_spatial_query_url(os_roads, aoi, out_fields='*'),
        "Buildings geoJSON shape file: " + _build_spatial_query_url(os_buildings, aoi, out_fields='*'),
        "Green spaces geoJSON shape file: " + _build_spatial_query_url(os_green, aoi, out_fields='*'),
    ])

    if include_seasons:
        logger.info("Including seasonal crown layers.")
        try:
            attrs_summer = get_project_urls("TT_GCW1_Summer")
            attrs_winter = get_project_urls("TT_GCW1_Winter")
            tree_crown_summer = get_attr(attrs_summer, "TREE_CROWNS")
            tree_crown_winter = get_attr(attrs_winter, "TREE_CROWNS")
            
            if tree_crown_summer:
                 data_locations.insert(
                    1 if tree_crowns_url else 0,
                    "Before storm tree crown geoJSON: " + _build_spatial_query_url(tree_crown_summer, aoi, out_fields='*')
                )
            else:
                 logger.warning("TT_GCW1_Summer TREE_CROWNS URL missing.")
                 
            if tree_crown_winter:
                data_locations.insert(
                    2 if tree_crowns_url and tree_crown_summer else 1 if tree_crowns_url or tree_crown_summer else 0,
                    "After storm tree crown geoJSON: " + _build_spatial_query_url(tree_crown_winter, aoi, out_fields='*')
                )
            else:
                 logger.warning("TT_GCW1_Winter TREE_CROWNS URL missing.")
                 
        except ValueError as ve:
             logger.error(f"Could not fetch seasonal project URLs: {ve}")
        except Exception as e:
            logger.error(f"Error including seasonal data locations: {e}")
            
    logger.info(f"Final list of {len(data_locations)} data locations generated.")
    for loc in data_locations:
        logger.debug(f"  - {loc}")
        
    return data_locations
