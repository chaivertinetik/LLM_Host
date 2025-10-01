#----------------------------Data Location logic-------------
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
import logging

# Configure basic logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClearRequest(BaseModel):
    task_name: str

# --------------------- ARC GIS UPDATE---------------------

def _json_default(obj):
    # ... (existing function content) ...
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
            return {
                "status": "success",
                "message": "Nothing to delete."
            }

        for target_url in valid_urls:
            logger.info(f"Attempting cleanup for URL: {target_url}")
            try:
                layer = _sanitise_layer_url(target_url)  # -> .../FeatureServer/<id>
                query_url  = f"{layer}/query"
                delete_url = f"{layer}/deleteFeatures"
                logger.debug(f"Sanitised layer URL: {layer}")

                # Step 1: Query for IDs
                params = {"where": "1=1", "returnIdsOnly": "true", "f": "json"}
                r = requests.get(query_url, params=params)
                r.raise_for_status()
                response_json = r.json()
                ids = response_json.get("objectIds", [])
                
                if "error" in response_json:
                    logger.error(f"ArcGIS Error during query for IDs at {query_url}: {response_json['error']}")
                    # Continue to next URL, don't raise
                    continue

                if not ids:
                    logger.info(f"No features (ids) found to delete at {layer}.")
                    continue

                logger.info(f"Found {len(ids)} features to delete at {layer}.")

                # Step 2: Delete features
                del_params = {"objectIds": ",".join(map(str, ids)), "f": "json"}
                dr = requests.post(delete_url, data=del_params)
                dr.raise_for_status()
                delete_response_json = dr.json()
                
                if delete_response_json.get("deleteResults", [{}])[0].get("success", False):
                    logger.info(f"Successfully deleted {len(ids)} features from {layer}.")
                    cleaned_any = True
                else:
                    logger.error(f"ArcGIS Error during delete features at {delete_url}: {delete_response_json}")


            except requests.RequestException as req_e:
                logger.error(f"Network/HTTP Error during cleanup of {target_url}: {req_e}")
            except Exception as e:
                logger.error(f"Error during cleanup of {target_url}: {e}", exc_info=True)
                
        logger.info(f"Cleanup finished for task {task_name}.")
        return {
            "status": "success",
            "message": "Cleanup completed." if cleaned_any else "Nothing to delete."
        }
    except Exception as e:
        logger.error(f"Top-level cleanup failure for {task_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
        

def shapely_to_arcgis_geometry(geom, wkid: int):
    # ... (existing function content) ...
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
    
    # List of all fields you want from the service
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
        response = requests.get(query_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            logger.error(f"ArcGIS Error fetching project index: {data['error']}")
            raise ValueError(f"ArcGIS Error for project '{project_name}': {data['error']['message']}")
            
        if not data.get("features"):
            logger.warning(f"No project found with the name '{project_name}'.")
            raise ValueError(f"No project found with the name '{project_name}'.")

        # Return as a dictionary of all requested attributes
        attributes = data["features"][0]["attributes"]
        logger.info(f"Successfully fetched attributes for '{project_name}'.")
        return attributes
    except requests.RequestException as e:
        logger.error(f"Network/HTTP Error fetching project index for '{project_name}': {e}")
        raise ValueError(f"Network error fetching project index: {e}")
    except ValueError:
        # Re-raise the ValueError about no project found
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_project_urls for '{project_name}': {e}", exc_info=True)
        raise ValueError(f"Unexpected error fetching project data: {e}")

# ... (fetch_crs function is below) ...

def _sanitise_layer_url(url: str) -> str:
    # ... (existing function content) ...
    logger.debug(f"Sanitising URL: {url}")
    if not isinstance(url, str) or not url.strip():
        logger.error("URL provided for sanitisation is empty or not a string.")
        raise ValueError("URL must be a non-empty string")

    u = url.strip().rstrip("/")

    # Strip trailing /query if present
    u = re.sub(r"/query$", "", u, flags=re.IGNORECASE)

    # Match MapServer or FeatureServer, with optional layer id
    m = re.search(r"(.*?/(?:FeatureServer|MapServer))(?:/(\d+))?$", u, flags=re.IGNORECASE)
    if not m:
        logger.error(f"URL does not appear to be a valid ArcGIS service: {url}")
        raise ValueError(f"Not a valid ArcGIS service URL: {url}")

    base, layer = m.groups()
    if layer is None:
        layer = "0"  # default to layer 0 if not provided

    result = f"{base}/{layer}"
    logger.debug(f"Sanitised result: {result}")
    return result

def _get_layer_max_record_count(layer_url: str, timeout: int = 10) -> int:
    """
    Ask the layer for its JSON and read maxRecordCount; fall back to 1000 if absent.
    """
    logger.debug(f"Attempting to fetch maxRecordCount for {layer_url}")
    default_mrc = 1000
    try:
        r = requests.get(f"{layer_url}?f=json", timeout=timeout)
        r.raise_for_status()
        meta = r.json()
        
        if "error" in meta:
            logger.warning(f"ArcGIS Error getting metadata for MRC at {layer_url}: {meta['error']['message']}. Using default {default_mrc}.")
            return default_mrc
            
        # ArcGIS sometimes names it maxRecordCount; default sensibly if missing.
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

def extract_geojson(url: str, where: str = "1=1", out_fields: str = "*", timeout: int = 15) -> gpd.GeoDataFrame | None:
    logger.info(f"Starting GeoJSON extraction from {url} with WHERE: {where}")
    try:
        layer_url = _sanitise_layer_url(url)
        wkid = fetch_crs(layer_url, timeout=timeout)
        page_size = _get_layer_max_record_count(layer_url, timeout=timeout)
        logger.info(f"Layer: {layer_url}, Detected WKID: {wkid}, Page Size: {page_size}")

        features = []
        offset = 0

        while True:
            params = {
                "where": where,
                "outFields": out_fields,
                "f": "geojson",
                "outSR": wkid,
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
            logger.debug(f"Querying with offset={offset}, count={page_size}")

            resp = requests.get(f"{layer_url}/query", params=params, timeout=timeout)
            resp.raise_for_status()

            try:
                payload = resp.json()
            except Exception as je:
                logger.error(f"Non-JSON response from service: {je}. Response text start: {resp.text[:100]}")
                raise RuntimeError(f"Non-JSON response from service: {je}") from je

            fc_features = payload.get("features", [])
            
            if "error" in payload:
                err = payload.get("error", {})
                msg = err.get("message") or "Unexpected error in ArcGIS response."
                logger.error(f"ArcGIS error in GeoJSON query for {layer_url}: {msg}")
                raise RuntimeError(f"ArcGIS error: {msg}")

            if not isinstance(fc_features, list):
                logger.error(f"Unexpected response structure: 'features' is not a list. Payload keys: {payload.keys()}")
                raise RuntimeError(f"Unexpected response structure from service")

            features.extend(fc_features)
            logger.debug(f"Fetched {len(fc_features)} features. Total collected: {len(features)}")

            # Stop if fewer than page_size returned (last page)
            if len(fc_features) < page_size:
                logger.debug(f"Last page reached. Breaking loop.")
                break

            offset += page_size
            logger.debug(f"Moving to next page, new offset: {offset}")


        if not features:
            logger.info("No features found. Returning empty GeoDataFrame.")
            try:
                # Return a valid empty GDF with CRS set
                return gpd.GeoDataFrame.from_features([], crs=f"EPSG:{wkid}")
            except Exception as crs_e:
                logger.warning(f"Failed to set CRS {wkid} on empty GeoDataFrame: {crs_e}")
                return gpd.GeoDataFrame()

        # Build GeoDataFrame; set CRS if possible
        gdf = gpd.GeoDataFrame.from_features(features)
        logger.info(f"Successfully created GeoDataFrame with {len(gdf)} features.")
        
        try:
            # Only set if not already set or if wkid looks valid
            if wkid and (gdf.crs is None or not gdf.crs):
                gdf.set_crs(epsg=wkid, inplace=True)
                logger.debug(f"Set CRS to EPSG:{wkid}")
        except Exception as crs_e:
            logger.warning(f"CRS assignment failed for WKID {wkid}: {crs_e}")
            pass

        return gdf

    except requests.HTTPError as he:
        logger.error(f"HTTP error fetching GeoJSON: {he} for URL {url}")
        return None
    except requests.Timeout:
        logger.error(f"GeoJSON fetch error: request timed out for URL {url}")
        return None
    except Exception as e:
        logger.error(f"GeoJSON fetch error for URL {url}: {e}", exc_info=True)
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
        logger.error(f"get_roi_gdf error for {project_name}: {e}", exc_info=True)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def sanitise_add_url(target_url: str) -> str:
    """
    Ensure the ArcGIS FeatureServer URL ends with /<layerId>/addFeatures.
    """
    logger.debug(f"Sanitising add URL: {target_url}")
    # Remove trailing slashes and 'addFeatures' if they already exist
    target_url = target_url.rstrip("/").removesuffix("/addFeatures")

    # If the URL ends with 'FeatureServer', add '/0' for the default layer.
    if target_url.endswith("/FeatureServer"):
        result = f"{target_url}/0/addFeatures"
        logger.debug(f"Sanitised result (default layer 0): {result}")
        return result

    # Otherwise, assume the layer ID is already present and append 'addFeatures'
    result = f"{target_url}/addFeatures"
    logger.debug(f"Sanitised result (appended addFeatures): {result}")
    return result


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

    # Detect target layer WKID
    layer_wkid = fetch_crs(target_url, default_wkid=3857)
    logger.info(f"Target layer WKID detected: {layer_wkid}")
    logger.debug(f"Input GDF CRS: {gdf.crs}")

    # Batch post
    for start in range(0, len(gdf), batch_size):
        end = start + batch_size
        batch_gdf = gdf.iloc[start:end]
        features = []
        logger.info(f"Processing batch {start//batch_size + 1} ({start} to {end-1})")

        invalid_geom_count = 0
        for index, row in batch_gdf.iterrows():
            try:
                # Reproject to target CRS if necessary (not explicitly done here, relies on caller/ArcGIS)
                # The assumption here is that shapely_to_arcgis_geometry handles the transformation
                # or that the gdf is already in the right CRS (best practice is to reproject explicitly).
                
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
                logger.error(f"Skipping row {index} due to geometry error: {e}", exc_info=True)

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
            response = requests.post(add_url, data=payload, headers=headers, timeout=60)
            response.raise_for_status()
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
            logger.error(f"Unexpected error during POST for batch {start//batch_size + 1}: {e}", exc_info=True)


def delete_all_features(target_url):
    logger.info(f"Attempting to delete all features from {target_url}")
    try:
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
        response = requests.get(query_url, params=params, timeout=10)
        response.raise_for_status()
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
        delete_response = requests.post(delete_url, data=delete_params, timeout=30)
        delete_response.raise_for_status()
        delete_data = delete_response.json()
        
        if "error" in delete_data:
            logger.error(f"ArcGIS Error during feature deletion: {delete_data['error']}")
        else:
            success_count = sum(1 for r in delete_data.get("deleteResults", []) if r.get("success"))
            logger.info(f"Successfully deleted {success_count} features. Delete response summary: {delete_data}")

    except requests.RequestException as e:
        logger.error(f"Network/HTTP Error during delete_all_features for {target_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in delete_all_features for {target_url}: {e}", exc_info=True)


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
                    # Continue to next geometry type if possible, or raise
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
        logger.error(f"Top-level error in filter function: {e}", exc_info=True)
        # Re-raise or handle as appropriate for API
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
        # Try int match first
        s_int = series.astype("int64", errors="raise")
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
    # ... (existing function content) ...
    if geom_types & {"Point", "MultiPoint"}:
        return "CHAT_OUTPUT_POINT"
    if geom_types & {"LineString", "MultiLineString"}:
        return "CHAT_OUTPUT_LINE"
    if geom_types & {"Polygon", "MultiPolygon"}:
        return "CHAT_OUTPUT_POLYGON"
    logger.error(f"Unsupported geometry types found: {geom_types}")
    raise ValueError(f"Unsupported geometry types found: {geom_types}")


def ensure_list(obj):
    # ... (existing function content) ...
    if obj is None:
        return []
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [obj]
    if isinstance(obj, (np.ndarray, pd.Series, list, tuple, set)):
        return list(obj)
    # If it’s a string with commas, split; else wrap
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("[") or s.startswith("{"):  # likely JSON -> leave to to_gdf or json parsing
            return [s]
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip() != ""]
        return [s]
    return [obj]


# --- URL utilities ------------------------------------------------------------

# _sanitise_layer_url and _get_layer_max_record_count are already defined and logged above.

# --- Geometry helpers ---------------------------------------------------------

def _gdf_from_layer_all(layer_url: str, out_wkid: int = 4326, timeout: int = 15) -> gpd.GeoDataFrame:
    """
    Pulls *all* features from a layer as GeoJSON (handles pagination).
    Returns an empty GDF if there are no features.
    """
    logger.info(f"Fetching all features from {layer_url} with target WKID {out_wkid}")
    page_size = _get_layer_max_record_count(layer_url, timeout=timeout)
    features = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "outSR": out_wkid,
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        logger.debug(f"Querying all features with offset={offset}, count={page_size}")
        
        try:
            r = requests.get(f"{layer_url}/query", params=params, timeout=timeout)
            r.raise_for_status()
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
    # ... (existing function content) ...
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

def fetch_crs(base_url, timeout=10, default_wkid=4326):
    """
    Fetches the Coordinate Reference System (CRS) WKID from an ArcGIS REST service endpoint.
    """
    logger.debug(f"Fetching CRS for base URL: {base_url}. Default WKID: {default_wkid}")
    try:
        response = requests.get(f"{base_url}?f=json", timeout=timeout)
        response.raise_for_status()
        metadata = response.json()
        
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
                polys = [geom for geom in gdf.geometry if geom and geom.geom_type in ("Polygon", "MultiPolygon")]
                if polys:
                    u = unary_union(polys)
                    logger.debug(f"Union result type: {u.geom_type}")
                    
                    if isinstance(u, GeometryCollection):
                        parts = [g for g in u.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                        u = unary_union(parts) if parts else None
                        logger.debug(f"After GeometryCollection split, union result type: {u.geom_type if u else 'None'}")
                        
                    if u and not u.is_empty:
                        # Ensure polygon or multipolygon
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
        r = requests.get(f"{base}?f=json", timeout=15)
        r.raise_for_status()
        meta = r.json()
        
        if "error" in meta:
             logger.error(f"ArcGIS Error getting metadata from ORTHOMOSAIC: {meta['error']}")
             raise RuntimeError(f"ORTHOMOSAIC service error: {meta['error']['message']}")

        # Extent could be at 'extent' or 'fullExtent' depending on service type
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
        logger.error(f"Error determining AOI from ORTHOMOSAIC: {e}. Falling back to empty envelope.", exc_info=True)
        # fall back to a harmless empty envelope in EPSG:4326
        return {
            "geometry": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0, "spatialReference": {"wkid": 4326}},
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326
        }
        
def _build_spatial_query_url(layer_url: str, aoi: dict, where: str = "1=1", out_fields: str = "*") -> str:
    logger.debug(f"Building spatial query URL for {layer_url}")
    lyr = _sanitise_layer_url(layer_url)
    
    # Use the layer's native CRS for outSR (best practice for GeoJSON query)
    # Note: ArcGIS sometimes forces outSR=4326 when f=geojson, but we send what we detect.
    out_wkid = fetch_crs(lyr) or 4326
    
    in_sr = aoi.get("inSR", 4326)
    
    # Geometry must be JSON-encoded; keep it compact
    geom_json = _json.dumps(aoi["geometry"], separators=(",", ":"))
    geom = _q(geom_json)

    query_url = (
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
    logger.debug(f"Built spatial query URL (truncated): {query_url[:200]}...")
    return query_url


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
        data_locations.append(f"Tree crown geoJSON shape file : {_build_spatial_query_url(tree_crowns_url, aoi, out_fields='*')}.")
    
    data_locations.extend([
        f"Roads geoJSON shape file: {_build_spatial_query_url(os_roads, aoi, out_fields='*')}.",
        f"Buildings geoJSON shape file: {_build_spatial_query_url(os_buildings, aoi, out_fields='*')}.",
        f"Green spaces geoJSON shape file: {_build_spatial_query_url(os_green, aoi, out_fields='*')}.",
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
                    1 if tree_crowns_url else 0, # put right after the main crowns (if present)
                    f"Before storm tree crown geoJSON: {_build_spatial_query_url(tree_crown_summer, aoi, out_fields='*')}."
                )
            else:
                 logger.warning("TT_GCW1_Summer TREE_CROWNS URL missing.")
                 
            if tree_crown_winter:
                data_locations.insert(
                    2 if tree_crowns_url and tree_crown_summer else 1 if tree_crowns_url or tree_crown_summer else 0,
                    f"After storm tree crown geoJSON: {_build_spatial_query_url(tree_crown_winter, aoi, out_fields='*')}."
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
