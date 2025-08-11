# --------------------- IMPORTS and helper modules ---------------------
import os
import json
import networkx as nx
import vertexai
import ee 
from datetime import datetime, timezone, date
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
from langchain_core.language_models import LLM


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





# --------------------- ARC GIS UPDATE---------------------
async def trigger_cleanup(task_name):
    project_name = task_name
    attrs = get_project_urls(task_name)
    target_url = attrs.get("CHAT_OUTPUT")
    target_url = _norm_layer(target_url) 
    try:
        query_url = f"{target_url}/0/query"
        delete_url = f"{target_url}/0/deleteFeatures"
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
           return {
              "status": "success",
              "message": "No features to delete.",
              "response": response.text
           }
        # Step 2: Delete by OBJECTIDs
        delete_params = {
        "objectIds": ",".join(map(str, object_ids)),
        "f": "json"
        }
        delete_response = requests.post(delete_url, data=delete_params)
        
        return {
            "status": "success",
            "message": "Cleanup triggered successfully.",
            "response": delete_response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
       
@app.post("/clear")
async def clear_state(req: RequestData):
    return await trigger_cleanup(req.task_name)
       
def shapely_to_arcgis_geometry(geom):
    if geom.geom_type == "Polygon":
        return {
            "rings": mapping(geom)["coordinates"],
            "spatialReference": {"wkid": 4326}
        }
    elif geom.geom_type == "MultiPolygon":
        return {
            "rings": [ring for polygon in mapping(geom)["coordinates"] for ring in polygon],
            "spatialReference": {"wkid": 4326}
        }
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


FIELDS = [
    "PROJECT_NAME","ORTHOMOSAIC","TREE_CROWNS","TREE_TOPS","ObjectId","PREDICTION",
    "CHAT_OUTPUT","USER_CROWNS","USER_TOPS","SURVEY_DATE","CrownSketch",
    "CrownSketch_Predictions","CHAT_INPUT"
]



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
        "CHAT_INPUT"
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

def relation_key(project_name: str) -> str:
    """
    Return 'DE_BOLSTONE' from inputs like:
      - 'DE_BOLSTONE_2024'
      - 'de_bolstone_2025_q2'
      - 'DE__BOLSTONE__2024'
    Rules: split on underscores/spaces, take first TWO non-empty parts, uppercased.
    """
    tokens = [t for t in re.split(r'[_\s]+', project_name.strip()) if t]
    if len(tokens) < 2:
        # If only one token, treat that as the key (edge case)
        return tokens[0].upper() if tokens else ""
    return f"{tokens[0].upper()}_{tokens[1].upper()}"

def _escape_like_literal(s: str) -> str:
    """
    ArcGIS/SQL LIKE escapes: '_' matches one char, '%' matches many.
    We want literal underscores from our key, so escape '_' and '%' and '\' itself.
    """
    s = s.replace("\\", "\\\\").replace("_", r"\_").replace("%", r"\%")
    return s

def _norm_layer(u: str) -> str:
    if not u: return u
    u = u.rstrip('/')
    # already has a layer id
    if re.search(r'/FeatureServer/\d+$', u):
        return u
    # has FeatureServer root only
    if u.endswith('/FeatureServer'):
        return u + '/0'
    return u  # leave as-is



def get_related_projects(project_name: str) -> list[dict]:
    """
    Fetch ALL Project_index rows whose PROJECT_NAME starts with the relation key.
    e.g., key 'DE_BOLSTONE' -> PROJECT_NAME LIKE 'DE\_BOLSTONE%'
    Sorted by SURVEY_DATE desc (fallback ObjectId).
    """
    key = relation_key(project_name)
    if not key:
        return []

    like_prefix = _escape_like_literal(key) + "%"  # e.g., 'DE\_BOLSTONE%'
    where = f"PROJECT_NAME LIKE '{like_prefix}' ESCAPE '\\'"

    url = "https://services-eu1.arcgis.com/8uHkpVrXUjYCyrO4/arcgis/rest/services/Project_index/FeatureServer/0/query"
    params = {
        "where": where,
        "outFields": ",".join(FIELDS),
        "returnGeometry": "false",
        "f": "json",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    records = [f["attributes"] for f in data.get("features", [])]

    # newest first by SURVEY_DATE (fallback ObjectId)
    records.sort(key=lambda a: (a.get("SURVEY_DATE") or 0, a.get("ObjectId") or 0), reverse=True)
    return records


def are_related(a: str, b: str) -> bool:
    """Quick check if two project names share the same relation key."""
    return relation_key(a) == relation_key(b)


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

def extract_geojson(url):
    try:
        response = requests.get(f"{url}/0/query?where=1%3D1&outFields=*&f=geojson", timeout=10)
        if response.status_code == 200:
            geojson = response.json()
            gdf = gpd.GeoDataFrame.from_features(geojson["features"])
            if gdf is None or gdf.empty:
                print("No crowns returned; nothing to post.")
                return
            return gdf
        else:
            print(f"Failed to fetch GeoJSON: {response.status_code}")
            return None
    except Exception as e:
        print(f"GeoJSON fetch error: {e}")
        return None
        
#change the batch_size based on the upper cap for Foxholes 
#def post_features_to_layer(gdf, target_url):
def post_features_to_layer(gdf, target_url, batch_size=800):
    add_url = f"{target_url}/0/addFeatures"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    allowed_fields = {"Health", "Tree_ID", "Species"}
    # uncomment this for the batch implementation
    # for start in range(0, len(gdf), batch_size):
    #     batch_gdf=gdf.iloc[start:start+batch_size]
    #     features=[]
    # features = []
    # for _, row in gdf.iterrows():
    for start in range(0, len(gdf), batch_size):
        batch_gdf=gdf.iloc[start:start+batch_size]
        features=[]
        for _, row in batch_gdf.iterrows():
            try:
                arcgis_geom = shapely_to_arcgis_geometry(row.geometry)
                attributes = {k: v for k, v in row.items() if k in allowed_fields}
                features.append({
                    "geometry": arcgis_geom,
                    "attributes": attributes
                })
            except Exception as e:
                print(f"Skipping row due to geometry error: {e}")

        payload = {
            "features": json.dumps(features),
            "f": "json"
        }

        response = requests.post(add_url, data=payload, headers=headers)
        if response.status_code == 200:
            # print(f"Batch {start//batch_size +1}: Features added successfully:", response.json())
            print("Features added successfully:", response.json())
        else:
            # print(f"Batch  {start//batch_size + 1} : Failed to add features:", response.text) 
            print("Failed to add features:", response.text)

def delete_all_features(target_url):
    query_url = f"{target_url}/0/query"
    delete_url = f"{target_url}/0/deleteFeatures"

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

def ensure_list(obj):
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [obj]
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        return list(obj)
    elif obj is None:
        return []
    else:
        return [obj]


def filter(FIDS, project_name):
    print("Made it to the filter function") 
    FIDS = ensure_list(FIDS)

    attrs = get_project_urls(project_name)
    tree_crowns_url = attrs.get("TREE_CROWNS")
    chat_output_url = attrs.get("CHAT_OUTPUT")

    if not tree_crowns_url or not chat_output_url:
        raise ValueError("Required URLs missing in Project Index.")

    gdf = extract_geojson(tree_crowns_url)

    print("Columns in GDF:", gdf.columns.tolist())

    if "Tree_ID" in gdf.columns:
        # Ensure TREE_ID is treated as integers for matching
        gdf["Tree_ID"] = gdf["Tree_ID"].astype(int)
        ash_gdf = gdf[gdf["Tree_ID"].isin(FIDS)]

        # Delete all features before posting new ones
        delete_all_features(chat_output_url)

        # Then post
        post_features_to_layer(ash_gdf, chat_output_url)
    else:
        print("Column 'Species' not found in GeoDataFrame.")   

# --------------------- ERDO LLM main functions ---------------------

def wants_map_output_keyword(prompt: str) -> bool:
    keywords = ["show", "display", "highlight", "visualize", "which trees", "what trees"]
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
        "Decide if the user's input is asking for a map, list, or visual display of spatial features. "
        "Return only 'yes' or 'no'. Examples:\n"
        "- 'Show all healthy trees' -> yes\n"
        "- 'Map the lost trees' -> yes\n"
        "- 'List trees with crown size over 5m' -> yes\n"
        "- 'What is the distance between trees' -> no\n"
        "- 'Visualize all ash trees' -> yes\n"
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


def long_running_task(user_task: str, task_name: str, data_locations: list):
    try:
        # job_status[job_id] = {"status": "running", "message": "Task is in progress"}
        # Set up task and directories
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
            data_locations=data_locations,
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
        solution.assembly_LLM_response = assembly_LLM_response
        solution.save_solution()

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
            return {
                "status": "completed",
                "message": "The server seems to be down or what you're asking for isn't in the database."
            }
        result = globals().get('result', None)
        print("result type:", type(result))
        print("Final result:", result)
       
        if wants_map_output(user_task):
            filter(result,task_name)
            print("Execution completed.")
            return {
                "status": "completed",
                "message": f"Task '{task_name}' executed successfully.",
                "tree_ids": result if isinstance(result, list) else None
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
    lat, lon = map(float, coords.split(","))
    context = get_geospatial_context(lat, lon)  # Your GEE function
    return json.dumps(context)
    
def get_zoning_info(coords: str) -> str:
    return f"Zoning: Residential permitted, max height 50ft at {coords}"

def get_climate_info(coords: str) -> str:
    return f"Climate: High flood risk zone, sea-level rise of 1.2m expected at {coords}"

def get_population_info(coords: str) -> str:
    return f"Population: 11,000 people/km² at {coords}"
    
def check_tree_health(coords: str) -> str:
    return f"Tree Health: Moderate tree cover, signs of drought stress at {coords}"

def assess_tree_benefit(coords: str) -> str:
    return f"Tree Benefits: High potential for carbon capture and shade cooling at {coords}"

def check_soil_suitability(coords: str) -> str:
    return f"Soil: Slightly compacted clay, pH 6.5 – suitable for native tree species at {coords}"

def get_geospatial_context(lat=40.7128, lon=-74.0060):
    point = ee.Geometry.Point([lon, lat])
    year = date.today().year
    today = date.today()

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
    Tool(name="ZoningLookup", func=get_zoning_info, description="Returns zoning rules..."),
    Tool(
        name="EarthEngineContext",
        func=get_geospatial_context_tool,
        description="Returns NDVI, precipitation, temperature, soil moisture, land cover, and elevation for given coordinates"
    ),
    Tool(name="ClimateData", func=get_climate_info, description="Returns climate risk..."),
    Tool(name="PopulationStats", func=get_population_info, description="Returns population density..."),
    Tool(name="TreeHealthCheck", func=check_tree_health, description="Assesses existing tree health at given coordinates"),
    Tool(name="TreeBenefitAssessment", func=assess_tree_benefit, description="Estimates environmental impact of planting trees"),
    Tool(name="SoilSuitability", func=check_soil_suitability, description="Checks soil type, pH, and suitability for tree planting")
    # gis_batch_tool
]



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

    if re.search(r"\b(clear|reset|cleanup|clean|wipe)\b", user_task):
        return await trigger_cleanup(task_name)
       
    if not is_geospatial_task(user_task):
        return {"status": "completed", "message": "I haven't been programmed to do that"}

    do_gis_op = want_gis_task(user_task)
    do_info   = wants_additional_info(user_task)

    # --- inline helpers (kept minimal) ---
    def _fmt_date(ms):
        if not ms:
            return "unknown-date"
        return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d")

    FIELDS = [
        "PROJECT_NAME","ORTHOMOSAIC","TREE_CROWNS","TREE_TOPS","ObjectId","PREDICTION",
        "CHAT_OUTPUT","USER_CROWNS","USER_TOPS","SURVEY_DATE","CrownSketch",
        "CrownSketch_Predictions","CHAT_INPUT"
    ]

    # Build relation key from first two underscore-separated tokens
    # e.g., "DE_BOLSTONE_2024" -> base_key "DE_BOLSTONE"
    tokens = [t for t in re.split(r'[_\s]+', (task_name or "").strip()) if t]
    if len(tokens) >= 2:
        base_key = f"{tokens[0].upper()}_{tokens[1].upper()}"
    elif tokens:
        base_key = tokens[0].upper()
    else:
        base_key = ""

    # Escape for LIKE (treat '_' literally) and make prefix
    like_prefix = base_key.replace("\\", "\\\\").replace("_", r"\_").replace("%", r"\%") + "%"
    where_clause = f"PROJECT_NAME LIKE '{like_prefix}' ESCAPE '\\'"

    query_url = "https://services-eu1.arcgis.com/8uHkpVrXUjYCyrO4/arcgis/rest/services/Project_index/FeatureServer/0/query"
    # ----------------------------------------------

    if do_gis_op and do_info:
        try:
            params = {
                "where": where_clause,
                "outFields": ",".join(FIELDS),
                "returnGeometry": "false",
                "f": "json",
            }
            r = requests.get(query_url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            records = [f["attributes"] for f in data.get("features", [])]
            if not records:
                raise ValueError(f"No related projects found for '{task_name}'.")

            # newest first by SURVEY_DATE (fallback ObjectId)
            records.sort(key=lambda a: (a.get("SURVEY_DATE") or 0, a.get("ObjectId") or 0), reverse=True)

            # Build data_locations for MULTI-DATE crowns + point input (prefer USER_TOPS)
            data_locations = []
            for attrs in records:
                date_label = _fmt_date(attrs.get("SURVEY_DATE"))
                crowns_url = attrs.get("TREE_CROWNS")
                points_url = attrs.get("USER_TOPS") or attrs.get("TREE_TOPS")
                crowns = _norm_layer(crowns_url)
                points = _norm_layer(points_url)
                if crowns:
                    data_locations.append(
                        f"Tree crown GeoJSON ({date_label}): {crowns}/query?where=1%3D1&outFields=*&f=geojson."
                    )
                if points:
                    data_locations.append(
                        f"Point input GeoJSON ({date_label}): {points}/query?where=1%3D1&outFields=*&f=geojson."
                    )

            result = long_running_task(user_task, task_name, data_locations)
            message = result.get("message") if isinstance(result, dict) else str(result)

            reasoning_prompt = (
                f"User asked: {user_task}\n"
                f"Batch task results summary: {message}\n"
                "Use the GIS tools (soil, climate, tree health, etc.) to answer the user's question."
            )
            try:
                reasoning_response = agent.run(reasoning_prompt)
                combined_message = f"{message}\n\nAdditional Analysis:\n{reasoning_response}"
            except Exception:
                combined_message = message

            return {
                "status": "completed",
                "message": combined_message,
                "response": {
                    "role": "assistant",
                    "content": result.get("tree_ids") if isinstance(result, dict) and "tree_ids" in result else message
                }
            }

        except Exception:
            return {"status": "completed", "message": "Request not understood as a task requiring GIS operations or information."}

    elif do_gis_op:
        try:
            params = {
                "where": where_clause,
                "outFields": ",".join(FIELDS),
                "returnGeometry": "false",
                "f": "json",
            }
            r = requests.get(query_url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            records = [f["attributes"] for f in data.get("features", [])]
            if not records:
                raise ValueError(f"No related projects found for '{task_name}'.")

            records.sort(key=lambda a: (a.get("SURVEY_DATE") or 0, a.get("ObjectId") or 0), reverse=True)

            data_locations = []
            for attrs in records:
                date_label = _fmt_date(attrs.get("SURVEY_DATE"))
                crowns_url = attrs.get("TREE_CROWNS")
                points_url = attrs.get("USER_TOPS") or attrs.get("TREE_TOPS")
                crowns = _norm_layer(crowns_url)
                points = _norm_layer(points_url)
                if crowns:
                    data_locations.append(
                        f"Tree crown GeoJSON ({date_label}): {crowns}/query?where=1%3D1&outFields=*&f=geojson."
                    )
                if points:
                    data_locations.append(
                        f"Point input GeoJSON ({date_label}): {points}/query?where=1%3D1&outFields=*&f=geojson."
                    )

            result = long_running_task(user_task, task_name, data_locations)
            message = result.get("message") if isinstance(result, dict) else str(result)

            return {
                "status": "completed",
                "message": message,
                "response": {
                    "role": "assistant",
                    "content": result.get("tree_ids") if isinstance(result, dict) and "tree_ids" in result else message
                }
            }
        except Exception:
            return {"status": "completed", "message": "Request not understood as a GIS task."}

    elif do_info:
        response = agent.run(user_task)
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
