import os
import json
import networkx as nx
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel
from google.oauth2 import service_account
from LLM_Heroku_Kernel import Solution
import helper
from flask_cors import CORS
import time
from google.api_core.exceptions import ResourceExhausted
import LLM_Geo_Constants as constants
from pyvis.network import Network
import requests
from google.oauth2.service_account import Credentials
import uuid
import threading
import requests
import geopandas as gpd
import json
from shapely.geometry import mapping
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from pydantic import BaseModel
import re
import textwrap
import black  
import autopep8
import numpy as np
import collections.abc
import pandas as pd
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import ee 
import datetime

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

# Global dictionary to track job statuses
# job_status: Dict[str, Dict[str, str]] = {}

# Load credentials from environment variable or file
# def get_credentials():
#     credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
#     return service_account.Credentials.from_service_account_info(credentials_info)

class RequestData(BaseModel):
    task: str = "No task provided."
    task_name: str = "default_task"

async def trigger_cleanup(task_name):
    project_name = task_name
    tree_crowns_url, delete_url = get_project_urls(project_name)
    try:
        target_url = delete_url
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
async def clear_state():
    return await trigger_cleanup()
   
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


def get_project_urls(project_name):
    query_url = "https://services-eu1.arcgis.com/8uHkpVrXUjYCyrO4/arcgis/rest/services/Project_index/FeatureServer/0/query"
    params = {
        "where": f"PROJECT_NAME = '{project_name}'",
        "outFields": "TREE_CROWNS,CHAT_OUTPUT",
        "f": "json",
    }

    response = requests.get(query_url, params=params, timeout=10)
    data = response.json()

    if not data.get("features"):
        raise ValueError(f"No project found with the name '{project_name}'.")

    attributes = data["features"][0]["attributes"]
    return attributes.get("TREE_CROWNS"), attributes.get("CHAT_OUTPUT")

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
    # Handle numpy scalars and standard scalars
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [obj]
    # Handle numpy arrays and pandas Series
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    # Handle other iterable types (list, tuple, set)
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        return list(obj)
    else:
        pass

def filter(FIDS, project_name):
    print("Made it to the filter function") 
    FIDS = ensure_list(FIDS)
    tree_crowns_url, chat_output_url = get_project_urls(project_name)

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
def long_running_task(user_task: str, task_name: str, data_locations: list):
    try:
        # job_status[job_id] = {"status": "running", "message": "Task is in progress"}
        # Set up task and directories
        save_dir = os.path.join(os.getcwd(), task_name)
        os.makedirs(save_dir, exist_ok=True)
        # Initialize Vertex AI
        credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
        credentials = service_account.Credentials.from_service_account_info(credentials_data)
        vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)
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

#, background_tasks: BackgroundTasks
@app.post("/process")
async def process_request(request_data: RequestData):
    user_task = request_data.task.strip().lower()
    task_name = request_data.task_name
    if re.search(r"\b(clear|reset|cleanup|clean|wipe)\b", user_task):
        return await trigger_cleanup(task_name)
       
    if not is_geospatial_task(user_task):
        return {
            "status": "completed",
            "message": "I haven't been programmed to do that"
        }
    # Generate a unique job ID
    # job_id = str(uuid.uuid4())
    # job_status[job_id] = {"status": "queued", "message": "Task is queued for processing"}
    try:
        tree_crowns_url, chat_output_url = get_project_urls(task_name)
        if task_name in ["TT_GCW1_Summer", "TT_GCW1_Winter"]:
            tree_crown_summer, _ = get_project_urls("TT_GCW1_Summer")
            tree_crown_winter, _ = get_project_urls("TT_GCW1_Winter")
        
            # Fetch CRS for each relevant URL
            # crs_current = fetch_crs(tree_crowns_url)
            # crs_summer = fetch_crs(tree_crown_summer)
            # crs_winter = fetch_crs(tree_crown_winter)
            # (CRS: EPSG:{crs_current})
            # (CRS: EPSG:{crs_summer})
            #( CRS: EPSG:{crs_winter})
            data_locations = [
                f"Tree crown geoJSON shape file: {tree_crowns_url}/0/query?where=1%3D1&outFields=*&f=geojson.",
                f"Before storm tree crown geoJSON: {tree_crown_summer}/0/query?where=1%3D1&outFields=*&f=geojson.",
                f"After storm tree crown geoJSON: {tree_crown_winter}/0/query?where=1%3D1&outFields=*&f=geojson."
            ]
        else:
            # Fetch CRS for the single URL
            # crs_current = fetch_crs(tree_crowns_url)
            # (CRS: EPSG:{crs_current})
            data_locations = [
                f"Tree crown geoJSON shape file: {tree_crowns_url}/0/query?where=1%3D1&outFields=*&f=geojson."
            ]
        # background_tasks.add_task(long_running_task, job_id, user_task, task_name, data_locations)
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
    # return {"status": "success", "job_id": job_id, "message": "Processing started..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Fetch the status of a background job using its job ID"""
    return job_status.get(job_id, {"status": "unknown", "message": "Job ID not found"})

# Run the FastAPI app using Uvicorn (for Cloud Run)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
