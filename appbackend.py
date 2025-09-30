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

class ClearRequest(BaseModel):
    task_name: str

# --------------------- ARC GIS UPDATE---------------------

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

async def trigger_cleanup(task_name: str):
    try:
        attrs = get_project_urls(task_name)

        target_keys = ["CHAT_OUTPUT_POINT", "CHAT_OUTPUT_LINE", "CHAT_OUTPUT_POLYGON"]
        urls = [get_attr(attrs, k) for k in target_keys]
        # Fallback: legacy single output
        legacy = get_attr(attrs, "CHAT_OUTPUT")
        if legacy:
            urls.append(legacy)

        cleaned_any = False
        for target_url in [u for u in urls if u]:
            layer = _sanitise_layer_url(target_url)   # -> .../FeatureServer/<id>
            query_url  = f"{layer}/query"
            delete_url = f"{layer}/deleteFeatures"

            params = {"where": "1=1", "returnIdsOnly": "true", "f": "json"}
            r = requests.get(query_url, params=params)
            ids = r.json().get("objectIds", [])
            if not ids:
                continue

            del_params = {"objectIds": ",".join(map(str, ids)), "f": "json"}
            dr = requests.post(delete_url, data=del_params)
            cleaned_any = True

        return {
            "status": "success",
            "message": "Cleanup completed." if cleaned_any else "Nothing to delete."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
      

def shapely_to_arcgis_geometry(geom, wkid: int):
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

    raise ValueError(f"Unsupported geometry type: {gt}")


def get_project_urls(project_name):
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

    response = requests.get(query_url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if not data.get("features"):
        raise ValueError(f"No project found with the name '{project_name}'.")

    # Return as a dictionary of all requested attributes
    return data["features"][0]["attributes"]

#CRS data is aligned and the same 
# def extract_geojson(service_url):
#     try:
#         # Fetch metadata to determine CRS
#         metadata_url = f"{service_url}/0?f=json"
#         metadata = requests.get(metadata_url, timeout=10).json()

#         wkid = metadata.get("extent", {}).get("spatialReference", {}).get("wkid")
#         if not wkid:
#             raise ValueError("Could not determine CRS from service metadata.")

#         query_url = f"{service_url}/0/query?where=1%3D1&outFields=*&f=geojson&outSR={wkid}"
#         response = requests.get(query_url, timeout=10)
#         if response.status_code == 200:
#             geojson = response.json()
#             gdf = gpd.GeoDataFrame.from_features(geojson["features"])
#             gdf.set_crs(epsg=wkid, inplace=True)
#             return gdf
#         else:
#             print(f"Failed to fetch GeoJSON: {response.status_code}")
#             return None
#     except Exception as e:
#         print(f"Error extracting GeoJSON: {e}")
#         return None

def fetch_crs(base_url, timeout=10, default_wkid=4326):
    """
    Fetches the Coordinate Reference System (CRS) WKID from an ArcGIS REST service endpoint.
    Works for MapServer, ImageServer, and FeatureServer.

    Args:
        base_url (str): The base URL of the ArcGIS service (e.g., ".../MapServer", ".../ImageServer", ".../FeatureServer").
        timeout (int): The maximum number of seconds to wait for a response.
        default_wkid (int): The default WKID to use if 'wkid' is missing from the spatialReference.

    Returns:
        int: The WKID of the spatial reference if found, or default_wkid if not found/error.
    """
    try:
        response = requests.get(f"{base_url}?f=json", timeout=timeout)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        metadata = response.json()

        # Check for spatialReference at the top level (common for FeatureServer and ImageServer)
        spatial_ref = metadata.get("spatialReference")
        if spatial_ref and "wkid" in spatial_ref:
            return spatial_ref["wkid"]

        # If not found at the top level, check for tileInfo (common for MapServer)
        tile_info = metadata.get("tileInfo")
        if tile_info:
            tile_spatial_ref = tile_info.get("spatialReference")
            if tile_spatial_ref and "wkid" in tile_spatial_ref:
                return tile_spatial_ref["wkid"]
            # Sometimes, the spatialReference for MapServer is at the root, or within tileInfo.
            # If not in tileInfo directly, check root again (redundant with initial check, but safer for edge cases)
            elif spatial_ref and "wkid" in spatial_ref:
                 return spatial_ref["wkid"]

        # For feature services with layers, the spatialReference might be within a layer object
        # This assumes you're hitting the base FeatureServer URL, and want the first layer's CRS.
        # If you need a specific layer's CRS, you'd need to append /<layerId> to the base_url
        layers = metadata.get("layers")
        if layers and len(layers) > 0:
            for layer in layers:
                layer_spatial_ref = layer.get("extent", {}).get("spatialReference") # Common for layers
                if layer_spatial_ref and "wkid" in layer_spatial_ref:
                    return layer_spatial_ref["wkid"]
                # Sometimes the spatialReference is directly on the layer object
                layer_spatial_ref_direct = layer.get("spatialReference")
                if layer_spatial_ref_direct and "wkid" in layer_spatial_ref_direct:
                    return layer_spatial_ref_direct["wkid"]


        print(f"WKID not found in metadata for {base_url}. Returning default WKID: {default_wkid}")
        return default_wkid
    except:
        return default_wkid

def _sanitise_layer_url(url: str) -> str:
    """
    Normalise any ArcGIS service URL to a concrete layer URL:
      - Accepts .../FeatureServer, .../FeatureServer/0, .../FeatureServer/0/query
      - Accepts .../MapServer, .../MapServer/3, .../MapServer/3/query
    Returns the canonical layer URL ending with /<layerId> (no /query).
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    u = url.strip().rstrip("/")

    # Strip trailing /query if present
    u = re.sub(r"/query$", "", u, flags=re.IGNORECASE)

    # Match MapServer or FeatureServer, with optional layer id
    m = re.search(r"(.*?/(?:FeatureServer|MapServer))(?:/(\d+))?$", u, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Not a valid ArcGIS service URL: {url}")

    base, layer = m.groups()
    if layer is None:
        layer = "0"  # default to layer 0 if not provided

    return f"{base}/{layer}"

def _get_layer_max_record_count(layer_url: str, timeout: int = 10) -> int:
    """
    Ask the layer for its JSON and read maxRecordCount; fall back to 1000 if absent.
    """
    try:
        r = requests.get(f"{layer_url}?f=json", timeout=timeout)
        r.raise_for_status()
        meta = r.json()
        # ArcGIS sometimes names it maxRecordCount; default sensibly if missing.
        mrc = meta.get("maxRecordCount")
        if isinstance(mrc, int) and mrc > 0:
            return mrc
    except Exception:
        pass
    return 1000

def extract_geojson(url: str, where: str = "1=1", out_fields: str = "*", timeout: int = 15) -> gpd.GeoDataFrame | None:
    """
    Sanitised extractor that:
      - normalises the URL,
      - detects CRS (via your fetch_crs),
      - paginates beyond maxRecordCount,
      - returns a GeoDataFrame with a proper CRS,
      - handles ArcGIS error envelopes gracefully.

    Returns None on hard failure; empty GeoDataFrame on no features.
    """
    try:
        layer_url = _sanitise_layer_url(url)
        wkid = fetch_crs(layer_url, timeout=timeout)  # uses your existing helper
        page_size = _get_layer_max_record_count(layer_url, timeout=timeout)

        features = []
        offset = 0

        while True:
            params = {
                "where": where,
                "outFields": out_fields,
                "f": "geojson",
                # GeoJSON outSR is supported by ArcGIS; align with detected CRS
                "outSR": wkid,
                # Pagination params
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }

            resp = requests.get(f"{layer_url}/query", params=params, timeout=timeout)
            # HTTP-level errors
            resp.raise_for_status()

            # ArcGIS may still return an error JSON with 200 status in non-geojson mode,
            # but in geojson mode it's usually a standard FeatureCollection. We still guard:
            try:
                payload = resp.json()
            except Exception as je:
                raise RuntimeError(f"Non-JSON response from service: {je}") from je

            # Some servers ignore outSR= when f=geojson; that's fine—GeoPandas will carry geometry in lon/lat if given.
            fc_features = payload.get("features", [])
            if not isinstance(fc_features, list):
                # If the service emitted an error envelope, surface it
                err = payload.get("error", {})
                msg = err.get("message") or "Unexpected response structure from service"
                raise RuntimeError(f"ArcGIS error: {msg}")

            features.extend(fc_features)

            # Stop if fewer than page_size returned (last page)
            if len(fc_features) < page_size:
                break

            offset += page_size

        if not features:
            # Return a valid empty GDF with CRS set (helps caller logic)
            try:
                return gpd.GeoDataFrame.from_features([], crs=f"EPSG:{wkid}")
            except Exception:
                return gpd.GeoDataFrame()

        # Build GeoDataFrame; set CRS if possible
        gdf = gpd.GeoDataFrame.from_features(features)
        try:
            # Only set if not already set or if wkid looks valid
            if wkid and (gdf.crs is None):
                gdf.set_crs(epsg=wkid, inplace=True)
        except Exception:
            # If CRS assignment fails, still return data
            pass

        return gdf

    except requests.HTTPError as he:
        print(f"HTTP error fetching GeoJSON: {he}")
        return None
    except requests.Timeout:
        print("GeoJSON fetch error: request timed out")
        return None
    except Exception as e:
        print(f"GeoJSON fetch error: {e}")
        return None

def get_roi_gdf(project_name: str) -> gpd.GeoDataFrame:
    """Return CHAT_INPUT features as GDF in EPSG:4326 (empty if none)."""
    try:
        attrs = get_project_urls(project_name)
        chat_input_url = get_attr(attrs, "CHAT_INPUT")
        if not chat_input_url:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        gdf = extract_geojson(chat_input_url)  # tagged 4326 above
        if gdf is None:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return gdf
    except Exception as e:
        print(f"get_roi_gdf error: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def sanitise_add_url(target_url: str) -> str:
    """
    Ensure the ArcGIS FeatureServer URL ends with /<layerId>/addFeatures.
    """
    # Remove trailing slashes and 'addFeatures' if they already exist
    target_url = target_url.rstrip("/").removesuffix("/addFeatures")

    # If the URL ends with 'FeatureServer', add '/0' for the default layer.
    if target_url.endswith("/FeatureServer"):
        return f"{target_url}/0/addFeatures"

    # Otherwise, assume the layer ID is already present and append 'addFeatures'
    return f"{target_url}/addFeatures"


def post_features_to_layer(gdf, target_url,project_name, batch_size=800):
    """
    Legacy-simple poster (no ROI, no clipping, no reprojection).
    - Uses only a small whitelist of attributes (Health, Tree_ID, Species)
    - Converts geometries to ArcGIS JSON and posts in batches
    - Guards against non-finite coords by sending geometry=None instead of dropping rows
    """
    import math

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
        print("Nothing to push: input is None/empty.")
        return

    # Detect target layer WKID (pass through to geometry builder if your helper supports it)
    layer_wkid = fetch_crs(target_url, default_wkid=3857)
    print(f"Target layer WKID: {layer_wkid}")

    # Batch post
    for start in range(0, len(gdf), batch_size):
        batch_gdf = gdf.iloc[start:start + batch_size]
        features = []

        for _, row in batch_gdf.iterrows():
            try:
                # Support either signature: with wkid (preferred) or without
                try:
                    arcgis_geom = shapely_to_arcgis_geometry(row.geometry, wkid=layer_wkid)
                except TypeError:
                    arcgis_geom = shapely_to_arcgis_geometry(row.geometry)

                arcgis_geom = sanitize_arcgis_geometry(arcgis_geom)
                if arcgis_geom is None:
                    print("Non-finite/invalid coords; sending geometry=None for one row.")

                attributes = {k: row.get(k, None) for k in allowed_fields if k in batch_gdf.columns}

                features.append({
                    "geometry": arcgis_geom,
                    "attributes": attributes
                })
            except Exception as e:
                # Keep parity with legacy behavior: skip the row on hard geometry error
                print(f"Skipping row due to geometry error: {e}")

        if not features:
            continue

        payload = {
            "features": json.dumps(features, default=_json_default),
            "f": "json"
        }

        try:
            response = requests.post(add_url, data=payload, headers=headers, timeout=60)
        except Exception as e:
            print(f"POST to {add_url} failed: {e}")
            continue

        if response.status_code == 200:
            print("Features added successfully:", response.json())
        else:
            print("Failed to add features:", response.text)


def delete_all_features(target_url):
    layer = _sanitise_layer_url(target_url)   # -> .../FeatureServer/<id>
    query_url  = f"{layer}/query"
    delete_url = f"{layer}/deleteFeatures"

    # Step 1: Get all existing OBJECTIDs
    params = {
        "where": "1=1",
        "returnIdsOnly": "true",
        "f": "json"
    }
    response = requests.get(query_url, params=params)
    data = response.json()

    object_ids = data.get("objectIds", [])
    if not object_ids:
        print("No features to delete.")
        return

    print(f"Deleting {len(object_ids)} existing features...")

    # Step 2: Delete by OBJECTIDs
    delete_params = {
        "objectIds": ",".join(map(str, object_ids)),
        "f": "json"
    }
    delete_response = requests.post(delete_url, data=delete_params)
    print("Delete response:", delete_response.json())


def filter(gdf_or_fids, project_name):
    """
    If gdf_or_fids is a GeoDataFrame (or GeoJSON) -> push those features directly (LLM already filtered).
    If gdf_or_fids is a list/iterable of IDs -> fetch source crowns, filter by ID, then push.
    """
    print("Made it to the filter function")

    attrs = get_project_urls(project_name)
    tree_crowns_url = get_attr(attrs, "TREE_CROWNS")
    if not tree_crowns_url:
        raise ValueError("TREE_CROWNS URL missing in Project Index.")

    # --- Case 1: Direct GDF provided ---
    gdf_direct = to_gdf(gdf_or_fids)
    if gdf_direct is not None and not gdf_direct.empty:
        print("Operating in GDF mode (direct push, no masking).")
        print(f"Direct GDF CRS: {gdf_direct.crs}")

        # Split by geometry type
        groups = {
            "point": gdf_direct[gdf_direct.geom_type.isin(["Point", "MultiPoint"])],
            "line": gdf_direct[gdf_direct.geom_type.isin(["LineString", "MultiLineString"])],
            "polygon": gdf_direct[gdf_direct.geom_type.isin(["Polygon", "MultiPolygon"])],
        }
        for kind, sub in groups.items():
            if sub.empty:
                continue
            if kind == "point":
                chat_output_url = get_attr(attrs, "CHAT_OUTPUT_POINT")
            elif kind == "line":
                chat_output_url = get_attr(attrs, "CHAT_OUTPUT_LINE")
            else:
                try:
                    chat_output_url = get_attr(attrs, "CHAT_OUTPUT_POLYGON")
                except:
                    chat_output_url = get_attr(attrs, "CHAT_OUTPUT")

            if not chat_output_url:
                raise ValueError(f"Matching CHAT_OUTPUT URL missing in Project Index for {kind}.")

            delete_all_features(chat_output_url)
            print(f"Pushing {len(sub)} {kind} feature(s) in CRS {sub.crs} → {chat_output_url}")
            post_features_to_layer(sub, chat_output_url, project_name)
        return

    # --- Case 2: FIDs provided (masking required) ---
    print("Operating in FID mode (fetch + mask).")
    FIDS = ensure_list(gdf_or_fids)

    gdf = extract_geojson(tree_crowns_url)
    if gdf is None or gdf.empty:
        print("Source crowns are empty or unavailable.")
        return

    print(f"Fetched crowns CRS: {gdf.crs}")
    print("Columns in GDF:", gdf.columns.tolist())
    print("Geometry types in GDF:", gdf.geom_type.unique())

    # Pick ID column
    id_column = None
    for candidate in ["Tree_ID", "OBJECTID", "FID", "Id"]:
        if candidate in gdf.columns:
            id_column = candidate
            break
    if not id_column:
        raise ValueError("No suitable ID column found (Tree_ID, OBJECTID, FID, Id).")

    print(f"Using ID column: {id_column}")
    mask = normalize_ids(gdf[id_column], FIDS)
    gdf_to_push = gdf[mask]
    if gdf_to_push.empty:
        print("No matching IDs found.")
        return

    # Split & push
    for kind, sub in {
        "point": gdf_to_push[gdf_to_push.geom_type.isin(["Point", "MultiPoint"])],
        "line": gdf_to_push[gdf_to_push.geom_type.isin(["LineString", "MultiLineString"])],
        "polygon": gdf_to_push[gdf_to_push.geom_type.isin(["Polygon", "MultiPolygon"])],
    }.items():
        if sub.empty:
            continue
        chat_output_url = (
            get_attr(attrs, "CHAT_OUTPUT_POINT") if kind == "point"
            else get_attr(attrs, "CHAT_OUTPUT_LINE") if kind == "line"
            else get_attr(attrs, "CHAT_OUTPUT_POLYGON")
        )
        if not chat_output_url:
            raise ValueError(f"Matching CHAT_OUTPUT URL missing for {kind}.")
        delete_all_features(chat_output_url)
        print(f"Pushing {len(sub)} {kind} feature(s) in CRS {sub.crs} → {chat_output_url}")
        post_features_to_layer(sub, chat_output_url, project_name)



# --- Helpers ---------------------------------------------------------------

def get_attr(attrs: dict, key: str):
    """Robust attribute getter that tolerates trailing spaces in field names."""
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
    """Accept a GeoDataFrame, a GeoJSON string/dict, or return None if not GDF-like."""
    try:
        
        if isinstance(maybe_gdf, gpd.GeoDataFrame):
            return maybe_gdf
        # Try dict-like GeoJSON
        if isinstance(maybe_gdf, dict) and "type" in maybe_gdf:
            feats = maybe_gdf.get("features")
            if feats:
                return gpd.GeoDataFrame.from_features(feats)
        # Try stringified GeoJSON
        if isinstance(maybe_gdf, str):
            try:
                gj = json.loads(maybe_gdf)
                feats = gj.get("features")
                if feats:
                    return gpd.GeoDataFrame.from_features(feats)
            except Exception:
                pass
    except Exception:
        pass
    return None

def normalize_ids(series, ids):
    """Try int match first; if that fails, fall back to string match."""
    try:
        s_int = series.astype("int64", errors="raise")
        ids_int = [int(x) for x in ensure_list(ids)]
        return s_int.isin(ids_int)
    except Exception:
        s_str = series.astype(str)
        ids_str = [str(x) for x in ensure_list(ids)]
        return s_str.isin(ids_str)

def geometry_target_key(geom_types: set):
    """Map geometry types to the correct CHAT_OUTPUT_* key name (no trailing space)."""
    if geom_types & {"Point", "MultiPoint"}:
        return "CHAT_OUTPUT_POINT"
    if geom_types & {"LineString", "MultiLineString"}:
        return "CHAT_OUTPUT_LINE"
    if geom_types & {"Polygon", "MultiPolygon"}:
        return "CHAT_OUTPUT_POLYGON"
    raise ValueError(f"Unsupported geometry types found: {geom_types}")


def ensure_list(obj):
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

def _sanitise_layer_url(url: str) -> str:
    """
    Normalise any ArcGIS service URL to a concrete layer URL:
      accepts .../FeatureServer, .../FeatureServer/0, .../FeatureServer/0/query
      and .../MapServer, .../MapServer/3, .../MapServer/3/query
    Returns canonical '.../(FeatureServer|MapServer)/<layerId>'.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    u = url.strip().rstrip("/")
    u = re.sub(r"/query$", "", u, flags=re.IGNORECASE)

    m = re.search(r"(.*?/(?:FeatureServer|MapServer))(?:/(\d+))?$", u, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Not a valid ArcGIS service URL: {url}")

    base, layer = m.groups()
    layer = layer or "0"
    return f"{base}/{layer}"

def _get_layer_max_record_count(layer_url: str, timeout: int = 10) -> int:
    try:
        r = requests.get(f"{layer_url}?f=json", timeout=timeout)
        r.raise_for_status()
        meta = r.json()
        mrc = meta.get("maxRecordCount")
        if isinstance(mrc, int) and mrc > 0:
            return mrc
    except Exception:
        pass
    return 1000

# --- Geometry helpers ---------------------------------------------------------

def _gdf_from_layer_all(layer_url: str, out_wkid: int = 4326, timeout: int = 15) -> gpd.GeoDataFrame:
    """
    Pull *all* features from a layer as GeoJSON (handles pagination).
    Returns an empty GDF if there are no features.
    """
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
        r = requests.get(f"{layer_url}/query", params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        fc = payload.get("features", [])
        if not isinstance(fc, list):
            err = payload.get("error", {})
            msg = err.get("message") or "Unexpected response structure from service"
            raise RuntimeError(f"ArcGIS error: {msg}")
        features.extend(fc)
        if len(fc) < page_size:
            break
        offset += page_size

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{out_wkid}")

    gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{out_wkid}")
    return gdf

def _rings_from_shapely(geom) -> list[list[list[float]]]:
    """
    Convert Polygon/MultiPolygon into Esri 'rings' (list of linear rings).
    Includes holes where present. Coordinates assumed in desired CRS already.
    """
    if isinstance(geom, Polygon):
        exterior = list(map(list, geom.exterior.coords)) if geom.exterior else []
        holes = [list(map(list, r.coords)) for r in geom.interiors] if geom.interiors else []
        return [exterior] + holes

    if isinstance(geom, MultiPolygon):
        rings = []
        for poly in geom.geoms:
            rings.extend(_rings_from_shapely(poly))
        return rings

    raise ValueError(f"Unsupported geometry type for rings: {geom.geom_type}")

def fetch_crs(base_url, timeout=10, default_wkid=4326):
    """
    Your existing helper in the previous message can be reused.
    Kept here for completeness—drop this if you already defined it elsewhere.
    """
    try:
        response = requests.get(f"{base_url}?f=json", timeout=timeout)
        response.raise_for_status()
        metadata = response.json()
        spatial_ref = metadata.get("spatialReference")
        if spatial_ref and "wkid" in spatial_ref:
            return spatial_ref["wkid"]
        tile_info = metadata.get("tileInfo")
        if tile_info:
            tile_spatial_ref = tile_info.get("spatialReference")
            if tile_spatial_ref and "wkid" in tile_spatial_ref:
                return tile_spatial_ref["wkid"]
        layers = metadata.get("layers")
        if layers:
            for layer in layers:
                sr = layer.get("extent", {}).get("spatialReference") or layer.get("spatialReference")
                if sr and "wkid" in sr:
                    return sr["wkid"]
        return default_wkid
    except Exception:
        return default_wkid

# --- AOI selection: CHAT_INPUT first, fallback to ORTHOMOSAIC extent -----------

def get_project_aoi_geometry(project_name: str):
    """
    Build an AOI geometry for querying:
      - If CHAT_INPUT has polygons: union them and return Esri polygon JSON.
      - Else: use ORTHOMOSAIC service extent and return Esri envelope JSON.

    Returns a dict:
      {
        "geometry": <Esri JSON geometry>,
        "geometryType": "esriGeometryPolygon" | "esriGeometryEnvelope",
        "inSR": <wkid>
      }
    """
    attrs = get_project_urls(project_name)

    chat_input_url = (attrs.get("CHAT_INPUT") or "").strip()
    ortho_url = (attrs.get("ORTHOMOSAIC") or "").strip()

    # 1) Try CHAT_INPUT polygons (FeatureServer layer)
    if chat_input_url:
        try:
            layer_url = _sanitise_layer_url(chat_input_url)
            wkid = fetch_crs(layer_url) or 4326
            gdf = _gdf_from_layer_all(layer_url, out_wkid=wkid)

            # Keep only polygonal geometries
            if not gdf.empty:
                polys = [geom for geom in gdf.geometry if geom and geom.geom_type in ("Polygon", "MultiPolygon")]
                if polys:
                    u = unary_union(polys)
                    # Handle GeometryCollection by extracting polygonal parts
                    if isinstance(u, GeometryCollection):
                        parts = [g for g in u.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                        u = unary_union(parts) if parts else None

                    if u and not u.is_empty:
                        # Ensure polygon or multipolygon
                        if u.geom_type == "Polygon":
                            rings = _rings_from_shapely(u)
                        elif u.geom_type == "MultiPolygon":
                            rings = _rings_from_shapely(u)
                        else:
                            raise ValueError(f"Unexpected union type: {u.geom_type}")

                        return {
                            "geometry": {
                                "rings": rings,
                                "spatialReference": {"wkid": wkid}
                            },
                            "geometryType": "esriGeometryPolygon",
                            "inSR": wkid
                        }
        except Exception as e:
            print(f"CHAT_INPUT AOI fallback due to: {e}")

    # 2) Fallback: ORTHOMOSAIC extent (ImageServer)
    if not ortho_url:
        raise ValueError("Both CHAT_INPUT and ORTHOMOSAIC are missing for this project.")

    try:
        base = ortho_url.rstrip("/")
        r = requests.get(f"{base}?f=json", timeout=15)
        r.raise_for_status()
        meta = r.json()

        # Extent could be at 'extent' or 'fullExtent' depending on service type
        extent = meta.get("extent") or meta.get("fullExtent")
        if not extent:
            raise RuntimeError("ORTHOMOSAIC service does not expose an extent.")

        sr = extent.get("spatialReference") or meta.get("spatialReference") or {}
        wkid = sr.get("wkid") or fetch_crs(base) or 4326

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
        # fall back to a harmless empty envelope in EPSG:4326
        return {
            "geometry": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0, "spatialReference": {"wkid": 4326}},
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326
        }
        
def _build_spatial_query_url(layer_url: str, aoi: dict, where: str = "1=1", out_fields: str = "*") -> str:
    """
    Returns a fully-formed ArcGIS /query URL that uses the project AOI.
    - layer_url: .../FeatureServer[/<id>] OR .../MapServer[/<id>] (any form is fine)
    - aoi: result of get_project_aoi_geometry(project_name)
    Notes:
      - f=geojson requires outSR=4326. We therefore force outSR to 4326.
    """
    lyr = _sanitise_layer_url(layer_url)
    print(lyr)

    # Force WGS84 when requesting GeoJSON
    out_wkid = fetch_crs(lyr) or 4326
    print(f"crs is {out_wkid}")

    # inSR comes from AOI; fall back to 4326 if missing
    in_sr = aoi.get("inSR", 4326)

    # Geometry must be JSON-encoded; keep it compact
    geom = _q(_json.dumps(aoi["geometry"], separators=(",", ":")))

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
    """
    Builds your data_locations array using the project AOI for spatial filtering.
    - include_seasons: True for TT_GCW1_Summer/Winter branch; False otherwise
    - attrs: result from get_project_urls(project_name)
    """
    aoi = get_project_aoi_geometry(project_name)

    # Core project layers
    tree_crowns_url = get_attr(attrs, "TREE_CROWNS")

    # National context layers (static)
    os_roads = "https://services.arcgis.com/qHLhLQrcvEnxjtPr/arcgis/rest/services/OS_OpenRoads/FeatureServer/1"
    os_buildings = "https://services.arcgis.com/qHLhLQrcvEnxjtPr/arcgis/rest/services/OS_OpenMap_Local_Buildings/FeatureServer/0"
    os_green = "https://services.arcgis.com/qHLhLQrcvEnxjtPr/arcgis/rest/services/OS_Open_Greenspace/FeatureServer/1"

    data_locations = [
        f"Tree crown geoJSON shape file : {_build_spatial_query_url(tree_crowns_url, aoi, out_fields='*')}.",
        f"Roads geoJSON shape file: {_build_spatial_query_url(os_roads, aoi, out_fields='*')}.",
        f"Buildings geoJSON shape file: {_build_spatial_query_url(os_buildings, aoi, out_fields='*')}.",
        f"Green spaces geoJSON shape file: {_build_spatial_query_url(os_green, aoi, out_fields='*')}.",
    ]

    if include_seasons:
        attrs_summer = get_project_urls("TT_GCW1_Summer")
        attrs_winter = get_project_urls("TT_GCW1_Winter")
        tree_crown_summer = get_attr(attrs_summer, "TREE_CROWNS")
        tree_crown_winter = get_attr(attrs_winter, "TREE_CROWNS")

        data_locations.insert(
            1,  # put seasonal layers right after the main crowns
            f"Before storm tree crown geoJSON: {_build_spatial_query_url(tree_crown_summer, aoi, out_fields='*')}."
        )
        data_locations.insert(
            2,
            f"After storm tree crown geoJSON: {_build_spatial_query_url(tree_crown_winter, aoi, out_fields='*')}."
        )
    print("Assigned data locations:")
    for loc in data_locations:
        print("  -", loc)
    return data_locations