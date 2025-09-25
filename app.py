# --------------------- IMPORTS and helper modules ---------------------
import os
import json
import networkx as nx
import vertexai
import ee 
import datetime
import helper
import time
import LLM_Geo_Constants as constants
import requests
import uuid
import threading
import requests
import geopandas as gpd
import json
import re
import textwrap
import black  
import autopep8
import numpy as np
import collections.abc
import pandas as pd
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel
from google.oauth2 import service_account
from LLM_Heroku_Kernel import Solution
from flask_cors import CORS
from google.api_core.exceptions import ResourceExhausted
from pyvis.network import Network
from google.oauth2.service_account import Credentials
from shapely.geometry import mapping
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import firestore 
from shapely.ops import unary_union
from sentence_transformers import SentenceTransformer
import rtree
# --------------------- Setup FASTAPI app ---------------------
# Initialize FastAPI app
app = FastAPI()

# Enable CORS (same as Flask-CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rchahel-vertinetik.github.io"],  # Adjust based on security needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class RequestData(BaseModel):
    task: str = "No task provided."
    task_name: str = "default_task"
   

# --------------------- SETUP and INIT---------------------

google_creds = os.environ.get("GOOGLE_CREDENTIALS")
if not google_creds:
    raise EnvironmentError("GOOGLE_CREDENTIALS env var is missing")

credentials_data = json.loads(google_creds)
credentials = service_account.Credentials.from_service_account_info(credentials_data)
# service_account_email = credentials_data.get("client_email")
# print(service_account_email)

# === Init Vertex AI ===
vertexai.init(
    project="disco-parsec-444415-c4",
    location="us-central1",
    credentials=credentials
)
#testing earth engine service 
SERVICE_ACCOUNT= 'earthengine@disco-parsec-444415-c4.iam.gserviceaccount.com'
key_path = '/tmp/earthengine-key.json'
with open(key_path, 'w') as f:
    f.write(os.environ['EARTH_CREDENTIALS'])
earth_credentials= ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_path)
ee.Initialize(earth_credentials, project='disco-parsec-444415-c4')
db = firestore.Client(project="disco-parsec-444415-c4", credentials=credentials)
parser = JsonOutputParser()
rag_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)
hf_token = os.environ.get("HF_TOKEN")
emd_model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=hf_token)

# --------------------- GIS CODE AGENT WRAPPER ---------------------

class GeminiLLM(LLM):
    model: GenerativeModel

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "code-gemini"
    
# === Create Gemini model ===
model = GenerativeModel("gemini-2.0-flash-001")
llm = GeminiLLM(model=model)


# Global dictionary to track job statuses
job_status: Dict[str, Dict[str, str]] = {}

# Load credentials from environment variable or file
# def get_credentials():
#     credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
#     return service_account.Credentials.from_service_account_info(credentials_info)



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


#Temporary : need to revert to the collections-> with documents version and fix the datetime serialization error 
def load_history(session_id:str, max_turns=5):
        doc= db.collection("chat_histories").document(session_id).get()
        history= doc.to_dict().get("history", []) if doc.exists else []
        return history[ -2* max_turns:]
    
def save_history(session_id: str, history: list):
    # Load existing history
    doc = db.collection("chat_histories").document(session_id).get()
    existing_history = doc.to_dict().get("history", []) if doc.exists else []

    # Append new history entries
    combined_history = existing_history + history

    # Save combined history back
    db.collection("chat_histories").document(session_id).set({"history": combined_history})
        
def build_conversation_prompt(new_user_prompt: str,
                              history: list | None = None,
                              max_turns: int = 5) -> str:
    history = history or []
    recent = history[-2 * max_turns:]
    lines = []
    for entry in recent:
        prefix = "User: " if entry.get('role') == 'user' else "Assistant: "
        lines.append(f"{prefix}{entry.get('content', '')}")
    lines.append(f"User: {new_user_prompt}")
    lines.append("Assistant:")
    return "\n".join(lines)



# --------------------- ARC GIS UPDATE---------------------
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
      
class ClearRequest(BaseModel):
    task_name: str

@app.post("/clear")
async def clear_state(req: ClearRequest):
    return await trigger_cleanup(req.task_name)
       
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
        import geopandas as gpd
        from geopandas import GeoDataFrame
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


# --------------------- ERDO LLM main functions ---------------------

def wants_map_output_keyword(prompt: str) -> bool:
    keywords = ["show", "display", "highlight", "visualize"]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in keywords)

def wants_map_output_genai(prompt: str) -> bool:
    credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
    credentials = service_account.Credentials.from_service_account_info(credentials_data)
    # Initialize Vertex AI
    # Adjust project and location as needed
    
    vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)

    model = GenerativeModel("gemini-2.0-flash-001")
    system_prompt = (
        "Decide if the user's input is asking for a map, geodataframe, or visual display of spatial features. "
        "Return only 'yes' for dispalying on the map or 'no' for things that can't be mapped. Examples:\n"
        "- 'Show all healthy trees' -> yes\n"
        "- 'Map the lost trees' -> yes\n"
        "- 'Can you count the ash trees' -> no\n"
        "- 'How many ash trees are there' -> no\n"
        "- 'Show trees with crown size over 5m' -> yes\n"
        "- 'List trees with crown size over 5m' -> no\n"
        "- 'What is the distance between trees' -> no\n"
        "- 'Visualize all ash trees' -> yes\n"
        "- 'How many ash trees are there' -> no\n"
        "- 'Count the number of oak trees' -> no\n"
        "- 'Which trees are missing?' -> yes\n"
        "- 'How much volume was lost?' -> no\n"
        "- 'What is the total number of trees?' -> no\n"
        "- 'Summarize changes between surveys' -> no"
    )
    full_prompt = f"{system_prompt}\n\nUser input: {prompt}\nAnswer:"
    response = model.generate_content(
        full_prompt,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 5
        }
    )
    answer = response.text.strip().lower()
    return answer.startswith("yes")

def wants_map_output(prompt: str) -> bool:
    # First try keyword matching
    if wants_map_output_keyword(prompt):
        return True
    # Fallback to GenAI classification
    return wants_map_output_genai(prompt)

def is_geospatial_task(prompt: str) -> bool:
    """Vertex AI does intent classification to determine if the task is geo spatial related"""
    from vertexai.language_models import TextGenerationModel

    credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
    credentials = service_account.Credentials.from_service_account_info(credentials_data)
    vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)
    # gemini-1.5-flash-002
    model = GenerativeModel("gemini-2.0-flash-001")
    system_prompt = (
        "Decide if the user's input is related to geospatial analysis or geospatial data. "
        "This includes queries about map features, tree health, species, spatial attributes, survey date, spatial selections, overlays, or analysis."
        "Return only 'yes' or 'no'. Examples:\n"
        "- 'Find all ash trees' -> yes\n"
        "- 'What's my mother’s name?' -> no\n"
        "- 'Show healthy trees' -> yes\n"
        "- 'List all trees with a crown size over 5m' -> yes\n"
        "- 'Show areas with high NDVI in a satellite image' -> yes\n"
        "- 'What is the capital of France?' -> no"
    )
    
    full_prompt = f"{system_prompt}\n\nUser input: {prompt}\nAnswer:"
    # response = model.predict(full_prompt, temperature=0.0, max_output_tokens=5)
    response = model.generate_content(
       full_prompt,
       generation_config={
          "temperature": 0.0,
          "max_output_tokens": 5}
    )
    answer = response.text.strip().lower()
    return answer.startswith("yes")

def clean_indentation(code):
     # Split the code into lines
    lines = code.split('\n')
     # Remove leading spaces/tabs on each line, and replace tabs with 4 spaces
    cleaned_lines = []
    for line in lines:
         # Strip unwanted leading spaces/tabs and then add consistent 4 spaces for each level
        cleaned_lines.append(line.lstrip())
     
     # Join the cleaned lines back into a single string with proper indentation
    return '\n'.join(cleaned_lines)
# job_id: str, 

def wants_additional_info_keyword(prompt: str) -> bool:
    keywords = [
        "advice", "explain", "reason", "why", "weather", "soil", "context",
        "impact", "effect", "should I do", "recommend", "suggest",
        "interpret", "analysis", "information", "based on", "because",
        "caused by", "influence", "due to", "assessment"
    ]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in keywords)

def wants_additional_info_genai(prompt: str) -> bool:
    import os, json
    from vertexai.generative_models import GenerativeModel
    import vertexai

    credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
    credentials = service_account.Credentials.from_service_account_info(credentials_data)
    vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)

    model = GenerativeModel("gemini-2.0-flash-001")
    system_prompt = (
        "Decide if the user's input is asking for additional geospatial explanation or advice, "
        "beyond simply showing or listing features. This includes queries about reasons, causes, impact, recommendations, "
        "interpretations, soil, weather, context, or what should be done. "
        "Return only 'yes' or 'no'. Examples:\n"
        "- 'Show all healthy trees' -> no\n"
        "- 'Which trees are unhealthy?' -> no\n"
        "- 'Map the largest crown' -> no\n"
        "- 'Why are many trees unhealthy?' -> yes\n"
        "- 'Give me advice based on temperature' -> yes\n"
        "- 'Should I plant here given the soil?' -> yes\n"
        "- 'What was the likely cause of tree loss?' -> yes\n"
        "- 'Explain the difference between two areas' -> yes"
    )
    full_prompt = f"{system_prompt}\n\nUser input: {prompt}\nAnswer:"
    response = model.generate_content(
        full_prompt,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 5
        }
    )
    answer = response.text.strip().lower()
    return answer.startswith("yes")

def wants_additional_info(prompt: str) -> bool:
    # Try keyword first
    if wants_additional_info_keyword(prompt):
        return True
    # Backstop with vertex AI LLM classification if keyword not found
    return wants_additional_info_genai(prompt)

def wants_gis_task_keyword(prompt: str) -> bool:
    keywords = [
        "show", "display", "map", "highlight", "visualize", "which trees", 
        "what trees", "list", "extract", "buffer", "join", "select", "clip", 
        "overlay", "spatial", "geopandas", "geospatial", "coordinates", 
        "location", "find", "query", "identify"
    ]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in keywords)

def wants_gis_task_genai(prompt: str) -> bool:
    import os, json
    from vertexai.generative_models import GenerativeModel
    import vertexai
    from google.oauth2 import service_account

    credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
    credentials = service_account.Credentials.from_service_account_info(credentials_data)
    vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)

    model = GenerativeModel("gemini-2.0-flash-001")

    system_prompt = (
        "Decide if the user's input is asking for a geospatial operation involving spatial data processing or analysis. "
        "This includes tasks like mapping, buffering, spatial querying, extraction of features, overlays, joins, or any "
        "operation needing geospatial calculations or data manipulation. Return only 'yes' or 'no'. Examples:\n"
        "- 'Show all healthy trees' -> yes\n"
        "- 'Find trees within 10 meters of the river' -> yes\n"
        "- 'Display soil quality around trees' -> yes\n"
        "- 'List species of trees in an area' -> yes\n"
        "- 'Explain why trees are unhealthy' -> no\n"
        "- 'What is the weather today?' -> no\n"
        "- 'Give me advice on planting trees' -> no\n"
        "- 'Map the areas with high NDVI' -> yes\n"
        "- 'Visualize crown sizes of oak trees' -> yes\n"
        "- 'Summarize changes in tree health' -> no"
    )
    full_prompt = f"{system_prompt}\n\nUser input: {prompt}\nAnswer:"
    response = model.generate_content(
        full_prompt,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 5
        }
    )
    answer = response.text.strip().lower()
    return answer.startswith("yes")

def want_gis_task(prompt: str) -> bool:
    # Try keyword matching first for speed
    if wants_gis_task_keyword(prompt):
        return True
    # Fallback to GenAI classifier for ambiguous queries
    return wants_gis_task_genai(prompt)

#-------- The debug agent ---------------

def try_llm_fix(code, error_message=None, max_attempts=2):
    fixed_code = code
    exec_globals = {}
    for attempt in range(max_attempts):
        try:
            if error_message:
                prompt = (
                    f"The following Python code produced the error: \n"
                    f"{error_message}\n"
                    f"Please fix the code and output only the corrected Python code:\n{fixed_code}\n"
                )
            else:
                prompt = f"Fix the following Python code and output only the corrected code:\n{fixed_code}\n"
            response = model.generate_content(prompt)
            fixed_code = helper.extract_code(response.text)
            exec(fixed_code, exec_globals)
            return True, fixed_code
        except Exception as e:
            print(f"Error during LLM fix attempt {attempt + 1}: {e}")
            error_message = str(e)
    return False, error_message

#---- The geospatial code llm pipeline -----------

def long_running_task(user_task: str, task_name: str, data_locations: list):
    try:
        # job_status[job_id] = {"status": "running", "message": "Task is in progress"}
        # Set up task and directories
        # print(f"Received user_task (should be single prompt): {user_task}")
        save_dir = os.path.join(os.getcwd(), task_name)
        os.makedirs(save_dir, exist_ok=True)
        # Initialize Vertex AI done at the start. 

        # credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
        # credentials = service_account.Credentials.from_service_account_info(credentials_data)
        # vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)

        # user_task = r"""1) To use a geoJSON file and return all the "Tree_ID" that are ash species ('Predicted Tree Species':'Ash').
        # """
        # task_name ='Tree_crown_quality'
        #Create Solution object        
        solution = Solution(
            task=user_task,
            task_name=task_name,
            save_dir=save_dir,
            data_locations=data_locations
    
        )

        # Generate solution graph
        response_for_graph = solution.get_LLM_response_for_graph()
        solution.graph_response = response_for_graph
        solution.save_solution()

        #  file_path = "debug_tree_id.py"

        # # Read the file content
        #  with open(file_path, "r") as file:
        #      debugged_code = file.read()
        
        # Store the code into solution.code_for_graph
        #solution.code_for_graph = debugged_code
        #print("The code is:",solution.code_for_graph)
        solution.code_for_graph = clean_indentation(solution.code_for_graph)
        exec(solution.code_for_graph)
        # Load graph file
        solution_graph = solution.load_graph_file()
        G = nx.read_graphml(solution.graph_file) 
        nt = helper.show_graph(G)
        html_name = os.path.join(os.getcwd(), solution.task_name + '.html') 

        # Generate operations
        operations = solution.get_LLM_responses_for_operations(review=False)
        solution.save_solution()
        all_operation_code_str = '\n'.join([operation['operation_code'] for operation in operations])

        # Generate assembly code
        assembly_LLM_response = solution.get_LLM_assembly_response(review=False)
        # solution.assembly_LLM_response = assembly_LLM_response
        # solution.save_solution()
        
        # Run the generated code
        #gemini-1.5-flash-002
        model = GenerativeModel("gemini-2.0-flash-001")
        for attempt in range(10):
           try: 
              response = model.generate_content(solution.assembly_prompt)
              break
           except ResourceExhausted: 
              if attempt<10:
                 time.sleep(10)
              else:
                 raise
        # response = model.generate_content(solution.assembly_prompt)
        code_for_assembly = helper.extract_code(response.text)

        # Combine all code
        
        # print("The combined code is: ", code_for_assembly)
        
            
        
        print("Starting execution...")
        # code_for_assembly = textwrap.dedent(code_for_assembly).strip()
        # code_for_assembly = autopep8.fix_code(code_for_assembly)
        # code_for_assembly = black.format_str(code_for_assembly, mode=black.FileMode())
        exec_globals = {}
        # Execute the code directly 
        try:
            exec(code_for_assembly, globals())
        except IndentationError as e:
            print("Entered exception zone")
            for attempt in range(10):
                try:
                    prompt = f"Fix Indentation in the following Python code:\n{code_for_assembly}\n"
                    response = model.generate_content(prompt)
                    break
                except ResourceExhausted: 
                    if attempt<10:
                        time.sleep(10)
                    else:
                        raise
            code_for_assembly = helper.extract_code(response.text)
            exec(code_for_assembly, globals())
        except Exception as e:
            print(f"Caught Exception: {e}, attempting LLM fix...")
            success, fixed_code_or_error = try_llm_fix(code_for_assembly, error_message=str(e))
            if success:
                try:
                    exec(fixed_code_or_error, globals())
                except Exception as e2:
                    print(f"Execution after LLM fix failed: {e2}")
                    return {
                        "status": "completed",
                        "message": f"Try being more specific with your prompt."
                    }
            else:
                print(f"LLM fix failed: {fixed_code_or_error}")
                return {
                        "status": "completed",
                        "message": "The server seems to be down or what you're asking for isn't in the database."
                       }
        result = globals().get('result', None)
        print("result type:", type(result))
        print("Final result:", result)
       
        if wants_map_output(user_task):
            
            print("Execution completed.")
            if hasattr(result, "to_json") and "GeoDataFrame" in str(type(result)):
                geojson = result.to_json()
                #need to update this to go to the write place in arcgis if its a geodf
                filter(geojson,task_name)
            elif isinstance(result, list): 
                filter(result,task_name)
            message = f"Task '{task_name}' executed successfully."
            if isinstance(result, str):
                message = result
            return {
                "status": "completed",
                "message": message,
                "tree_ids": result if isinstance(result, list) else (result.to_json() if hasattr(result, "to_json") and "GeoDataFrame" in str(type(result)) else None)
            }
        # job_status[job_id] = {"status": "completed", "message": f"Task '{task_name}' executed successfully, adding it to the map shortly."}
        else: 
                return{
                    "status": "completed",
                    "message": str(result)
                }
        
     
    except Exception as e:
        print(f"Error during execution: {e}")
        #job_status[job_id] = {"status": "failed", "message": str(e)}
        # return f"Error during execution: {str(e)}"
        return f"Error during execution: The server seems to be down."


# === Simulated tools ===
def get_geospatial_context_tool(coords: str) -> str:
    #dynamically get based on map 
    
    lat, lon = map(float, coords.split(","))
    context = get_geospatial_context(lat, lon)  # Your GEE function
    return json.dumps(context)
    
def get_zoning_info(coords: str = "40.7128,-74.0060") -> str:
    # Since zoning isn't directly in Earth Engine data, we use land cover and forest loss as proxy
    context_json = get_geospatial_context_tool(coords)
    context = json.loads(context_json)

    land_cover = context.get("Land Cover Class (ESA)", "Unknown")
    forest_loss_year = context.get("Forest Loss Year (avg)", "N/A")
    
    zoning_msg = f"Land cover class: {land_cover}."
    if forest_loss_year != 'N/A':
        zoning_msg += f" Recent forest loss observed, average year: {forest_loss_year}."
    zoning_msg += " Tree planting recommended in reforestation or conservation zones."

def get_climate_info(coords: str = "40.7128,-74.0060") -> str:
    
    context_json = get_geospatial_context_tool(coords)
    context = json.loads(context_json)
    
    precipitation = context.get("Precipitation (mm)", 0)
    temperature = context.get("Temperature (°C)", 0)
    ndvi = context.get("NDVI (mean)", 0)

    flood_risk = "High" if precipitation > 1000 else "Moderate" if precipitation > 500 else "Low"
    sea_level_rise_estimate_m = 1.2  # Placeholder: for real, integrate NOAA data externally

    climate_msg = (f"Climate summary at {coords}:\n"
                   f"Precipitation: {precipitation} mm (Flood Risk: {flood_risk})\n"
                   f"Mean Temperature: {temperature} °C\n"
                   f"Vegetation Health (NDVI): {ndvi}\n"
                   f"Estimated sea-level rise: {sea_level_rise_estimate_m} m over next decades")
                   
    return climate_msg

    
def check_tree_health(coords: str = "40.7128,-74.0060")  -> dict:
    ee_result = get_geospatial_context_tool(coords)
    context = json.loads(ee_result)
    health_comment = "Healthy canopy" if context["NDVI (mean)"] > 0.5 else "Canopy thinning or stress"
    drought_comment = "Low drought stress" if context["Soil Moisture (m3/m3)"] > 0.25 else "Signs of drought stress"
    return {
        "Location": coords,
        "Canopy NDVI": context["NDVI (mean)"],
        "Soil Moisture": context["Soil Moisture (m3/m3)"],
        "Health Assessment": f"{health_comment}; {drought_comment}",
        "Forestry Recommendation": (
            "Monitor for canopy decline; consider supplemental watering and replace non-native stressed species."
        )
    }

def assess_tree_benefit(coords: str = "40.7128,-74.0060") -> dict:
    # Example: Logic grounded in context
    geo = json.loads(get_geospatial_context_tool(coords))
    benefit = "Excellent for carbon capture" if geo["NDVI (mean)"] > 0.7 and geo["Precipitation (mm)"] > 600 else "Moderate"
    cooling = "Substantial cooling from mature canopy" if geo["Land Cover Class (ESA)"] == "Forest" else "Potential cooling with reforestation"
    return {
        "Location": coords,
        "Carbon Capture Potential": benefit,
        "Shade/Cooling Impact": cooling,
        "Reference Data": geo
    }

def check_soil_suitability(coords: str) -> str:
    context_json = get_geospatial_context_tool(coords)
    context = json.loads(context_json)
    
    # Use soil moisture, elevation or land cover info as proxy for soil suitability
    soil_moisture = context.get("Soil Moisture (m3/m3)", None)
    elevation = context.get("Elevation (m)", None)
    land_cover = context.get("Land Cover Class (ESA)", "Unknown")

    # Simplified interpretation rules (expand or replace with richer logic)
    if soil_moisture is not None and 0.2 <= soil_moisture <= 0.4:
        moisture_msg = "Suitable soil moisture for native tree species growth."
    else:
        moisture_msg = "Soil moisture outside ideal range; irrigation or species choice recommended."

    return (f"Soil suitability at {coords}:\n"
            f"{moisture_msg}\n"
            f"Elevation: {elevation} m\n"
            f"Land Cover Type: {land_cover}")

def get_geospatial_context(lat=40.7128, lon=-74.0060):
    point = ee.Geometry.Point([lon, lat])
    year = datetime.date.today().year
    today = datetime.date.today()

    # Try using current year
    try_start = ee.Date.fromYMD(year, 1, 1)
    try_end = ee.Date.fromYMD(year, today.month, today.day)

    # Fallback default year
    fallback_start = ee.Date('2023-01-01')
    fallback_end = ee.Date('2023-12-31')
    
    def fetch(collection_id, selector, start, end, scale):
        try:
            coll = ee.ImageCollection(collection_id) \
                .filterDate(start, end) \
                .filterBounds(point) \
                .select(selector)
            return coll.mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=scale
            ).getInfo()
        except:
            return {}

    # Fetch NDVI (MODIS)
    ndvi = fetch('MODIS/006/MOD13Q1', 'NDVI', try_start, try_end, 250) or \
           fetch('MODIS/006/MOD13Q1', 'NDVI', fallback_start, fallback_end, 250)

    # Fetch Precipitation (CHIRPS)
    precip = fetch('UCSB-CHG/CHIRPS/DAILY', 'precipitation', try_start, try_end, 5000) or \
             fetch('UCSB-CHG/CHIRPS/DAILY', 'precipitation', fallback_start, fallback_end, 5000)

    # Fetch Temperature (ERA5-Land)
    temp = fetch('ECMWF/ERA5_LAND/DAILY_AGGR', 'temperature_2m', try_start, try_end, 1000) or \
           fetch('ECMWF/ERA5_LAND/DAILY_AGGR', 'temperature_2m', fallback_start, fallback_end, 1000)

    # Land use from ESA (static - 2020)
    landcover = ee.Image('ESA/WorldCover/v100/2020').sample(point, 10).first().getInfo()

    # Soil Moisture from SMAP (daily 10km)
    soil = fetch('NASA_USDA/HSL/SMAP10KM_soil_moisture', 'ssm', try_start, try_end, 10000) or \
           fetch('NASA_USDA/HSL/SMAP10KM_soil_moisture', 'ssm', fallback_start, fallback_end, 10000)

    # Forest loss (Hansen 2000–2022)
    forest = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
    forest_loss = forest.select('lossyear').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=30
    ).getInfo()

    # Elevation (SRTM, static)
    elevation = ee.Image('USGS/SRTMGL1_003').sample(point, 30).first().getInfo()

    # Assemble response
    return {
        "Latitude": lat,
        "Longitude": lon,
        "NDVI (mean)": round(ndvi.get('NDVI', 0) / 10000.0, 3),
        "Precipitation (mm)": round(precip.get('precipitation', 0), 2),
        "Temperature (°C)": round(temp.get('temperature_2m', 273.15) - 273.15, 2),
        "Soil Moisture (m3/m3)": round(soil.get('ssm', 0), 3),
        "Forest Loss Year (avg)": forest_loss.get('lossyear', 'N/A'),
        "Land Cover Class (ESA)": landcover.get('map', 'N/A'),
        "Elevation (m)": elevation.get('elevation', 'N/A')
    }

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_rag_chunks(collection_name, query, top_k=5):
    """
    Retrieve top K most semantically similar chunks from the specified
    subcollection under knowledge_chunks/root document in Firestore.
    """
    root_ref = db.collection("knowledge_chunks").document("root")
    chunks_ref = root_ref.collection(collection_name).stream()

    query_emb = emd_model.encode([query])[0]

    scored_chunks = []
    for doc in chunks_ref:
        chunk = doc.to_dict()
        emb = chunk.get("embedding", None)
        if emb is not None:
            # Convert embedding list to numpy array
            emb_np = np.array(emb)
            sim = cosine_similarity(query_emb, emb_np)
            scored_chunks.append((sim, chunk))

    # Sort chunks by descending similarity
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Return only the content field of top_k documents
    top_contents = [chunk["content"] for _, chunk in scored_chunks[:top_k]]

    return top_contents

prompt_template = PromptTemplate(
    input_variables=["query", "context", "format_instructions"],
    template=(
        "Use the following forestry data extracted from documents:\n"
        "{context}\n\n"
        "Answer the query with geospatial reasoning:\n"
        "{query}\n\n"
        "{format_instructions}\n"
        "Return only valid JSON."
    )
)

def rag_tree_grants_tool(query: str) -> str:
    chunks = retrieve_rag_chunks("tree_grants", query)
    if not chunks:
        return json.dumps({"result": [], "message": "No relevant tree grants data found."})
    context_text = "\n".join(chunks)
    prompt = prompt_template.format(
        query=query,
        context=context_text,
        format_instructions=parser.get_format_instructions()
    )
    response = rag_llm.invoke(prompt)
    parsed = parser.parse(response.content)
    return json.dumps(parsed)

def rag_tree_info_tool(query: str) -> str:
    chunks = retrieve_rag_chunks("tree_info", query)
    if not chunks:
        return json.dumps({"result": [], "message": "No relevant tree info data found."})
    context_text = "\n".join(chunks)
    prompt = prompt_template.format(
        query=query,
        context=context_text,
        format_instructions=parser.get_format_instructions()
    )
    response = rag_llm.invoke(prompt)
    parsed = parser.parse(response.content)
    return json.dumps(parsed)



#Can wrap the entire long process into this tool. so LLM orchestrator can handle. 
# def gis_solution_tool(query: str) -> str:
#     """
#     Invokes your existing long_running_task with params parsed or defaulted from query.
#     You may want to improve parsing logic depending on query format.
#     """
#     user_task = query
#     task_name = "GIS_LongRunningTask"
#     data_locations = []  # Fill as appropriate, could parse from query or configure by task_name

#     result = long_running_task(user_task, task_name, data_locations)

#     if isinstance(result, dict):
#         message = result.get("message", str(result))
#         if "tree_ids" in result:
#             message += f"\nTree IDs found: {result['tree_ids']}"
#         return message
#     return str(result)


# gis_batch_tool = Tool(
#     name="GISBatchProcessor",
#     func=gis_solution_tool,
#     description="Executes advanced GIS batch processing tasks using the Solution pipeline."
# )

tools = [
    Tool(name="ZoningLookup", func=get_zoning_info, description="Provides zoning-related land cover and forest loss info as proxy to guide tree planting recommendations."),
    Tool(name="ClimateLookUp", func=get_climate_info, description="Returns precipitation, temperature, vegetation health (NDVI), flood risk, and sea level rise estimates for forestry planning."),
    Tool(name="CheckTreeHealth", func=check_tree_health, description="Assess how healthy the trees are using the canopy cover and soil."),
    Tool(name="SoilSuitabilityCheck",func=check_soil_suitability,description="Analyzes soil moisture, elevation, and land cover to evaluate suitability for native tree species planting."), 
    Tool(name="TreeBenefitAssessment", func=assess_tree_benefit, description="Estimates carbon capture potential and cooling benefits based on NDVI, precipitation, and land cover data."),
    Tool(
        name="RAGTreeGrants",
        func=rag_tree_grants_tool,
        description="Retrieves recent tree grant and licensing information based on the users query."
    ),

    Tool(
        name="RAGTreeInfo",
        func=rag_tree_info_tool,
        description="Retrieves additional forestry and tree information from based on UK forestry records and rules."
    )
    # gis_batch_tool
]

#----------------------------Data Location logic-------------
import json as _json
from urllib.parse import quote as _q
import re
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

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

    return data_locations




# --------------------- Initialize agent with tools and LangChain LLM ---------------------

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# --------------------- Handle App process---------------------
  
@app.post("/process")
async def process_request(request_data: RequestData):
    user_task = request_data.task.strip().lower()
    task_name = request_data.task_name
    session_id = request_data.task_name
    if re.search(r"\b(clear|reset|cleanup|clean|wipe)\b", user_task):
        return await trigger_cleanup(task_name)
       
    
    history = load_history(session_id, max_turns=10)
    full_context = build_conversation_prompt(user_task, history) 
    if not is_geospatial_task(full_context):
        history.append({'role': 'user', 'content': user_task})
        history.append({'role': 'assistant', 'content': "Not programmed to do that."})
        save_history(session_id, history)
        return {
            "status": "completed",
            "message": "I haven't been programmed to do that"
        }
    # Generate a unique job ID
    # job_id = str(uuid.uuid4())
    # job_status[job_id] = {"status": "queued", "message": "Task is queued for processing"}
    
    do_gis_op= want_gis_task(user_task)
    do_info = wants_additional_info(user_task)
    
    if do_gis_op and do_info:
        try:
            attrs = get_project_urls(task_name)
            tree_crowns_url = get_attr(attrs, "TREE_CROWNS")
            roi_url = get_attr(attrs, "CHAT_INPUT")
            if task_name in ["TT_GCW1_Summer", "TT_GCW1_Winter"]:
                data_locations = make_project_data_locations(task_name, include_seasons=True, attrs=attrs)
            else:
                data_locations = make_project_data_locations(task_name, include_seasons=False, attrs=attrs)
            # background_tasks.add_task(long_running_task, job_id, user_task, task_name, data_locations)
            result = long_running_task(user_task, task_name, data_locations)
            message = result.get("message") if isinstance(result, dict) else str(result)
            reasoning_prompt = (
                    f"User asked: {user_task}\n"
                    f"Batch task results summary: {message}\n"
                    "Use the GIS tools (soil, climate, tree health, etc.) to answer the user's question."
            )
            full_context = build_conversation_prompt(reasoning_prompt, history) 
            
            try:    
                reasoning_response = agent.run(full_context)
                
                combined_message = f"{message}\n\nAdditional Analysis:\n{reasoning_response}"
            except Exception as e: 
                combined_message= message
            history.append({'role': 'user', 'content': user_task})
            history.append({'role': 'assistant', 'content': combined_message})
            save_history(session_id, history)
            
            return {
                "status": "completed",
                "message": combined_message,
                "response": {
                    "role": "assistant",
                    "content": result.get("tree_ids") if isinstance(result, dict) and "tree_ids" in result else message
                }
            }
        
        # return {"status": "success", "job_id": job_id, "message": "Processing started..."}
        except Exception as e:
            return {"status": "completed", "message": "Request not understood as a task requiring GIS operations or information."}
    elif do_gis_op: 
        
        try:
            attrs = get_project_urls(task_name)
            tree_crowns_url = get_attr(attrs, "TREE_CROWNS")
            roi_url = get_attr(attrs, "CHAT_INPUT")
            if task_name in ["TT_GCW1_Summer", "TT_GCW1_Winter"]:
                data_locations = make_project_data_locations(task_name, include_seasons=True, attrs=attrs)
            else:
                data_locations = make_project_data_locations(task_name, include_seasons=False, attrs=attrs)
            # background_tasks.add_task(long_running_task, job_id, user_task, task_name, data_locations)
            result = long_running_task(user_task, task_name, data_locations)
            message = result.get("message") if isinstance(result, dict) else str(result)
            
            history.append({'role': 'assistant', 'content': user_task})
            history.append({'role': 'assistant', 'content': message})
            save_history(session_id, history)
            
            return { 
                "status": "completed",
                "message": message,
                "response": {
                    "role": "assistant",
                    "content": result.get("tree_ids") if isinstance(result, dict) and "tree_ids" in result else message
                }
            }
        except Exception as e:
            return {"status": "completed", "message": "Request not understood as a GIS task."}

    elif do_info: 
        response = agent.run(full_context)
        history.append({'role': 'assistant', 'content': user_task})
        history.append({'role': 'assistant', 'content': response})
        save_history(session_id, history)
        return {"status": "completed", "response": response}
    
    return {"status": "completed", "message": "Request not understood as a task requiring geospatial data."}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Fetch the status of a background job using its job ID"""
    return job_status.get(job_id, {"status": "unknown", "message": "Job ID not found"})


# --------------------- run the app ---------------------
# Run the FastAPI app using Uvicorn (for Cloud Run)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
