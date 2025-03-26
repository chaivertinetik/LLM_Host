from flask import Flask, request, jsonify
import os
import json
import networkx as nx
import vertexai
from vertexai.generative_models import GenerativeModel
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


   
# Initialize Flask app

app = Flask(__name__)
CORS(app)

# Global dictionary to track job statuses
job_status = {}

# Load credentials from environment variable or file
# def get_credentials():
#     credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
#     return service_account.Credentials.from_service_account_info(credentials_info)
def shapely_to_arcgis_geometry(geom):
    if geom.geom_type == "Polygon":
        return {
            "rings": mapping(geom)["coordinates"],
            "spatialReference": {"wkid": 27700}
        }
    elif geom.geom_type == "MultiPolygon":
        # Flatten the list of polygons
        rings = []
        for polygon in mapping(geom)["coordinates"]:
            rings.extend(polygon)
        return {
            "rings": rings,
            "spatialReference": {"wkid": 27700}
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


def extract_geojson(url):
    try:
        response = requests.get(f"{url}/0/query?where=1%3D1&outFields=*&f=geojson", timeout=10)
        if response.status_code == 200:
            geojson = response.json()
            gdf = gpd.GeoDataFrame.from_features(geojson["features"])
            gdf.set_crs("EPSG:4326", inplace=True)
            return gdf.to_crs("EPSG:27700")
        else:
            print(f"Failed to fetch GeoJSON: {response.status_code}")
            return None
    except Exception as e:
        print(f"GeoJSON fetch error: {e}")
        return None


def post_features_to_layer(gdf, target_url):
    add_url = f"{target_url}/0/addFeatures"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    allowed_fields = {"Health", "Tree_ID", "Species"}

    features = []
    for _, row in gdf.iterrows():
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
        print("Features added successfully:", response.json())
    else:
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

def filter(FIDS):
    project_name = "FOXHOLES"
    tree_crowns_url, chat_output_url = get_project_urls(project_name)

    if not tree_crowns_url or not chat_output_url:
        raise ValueError("Required URLs missing in Project Index.")

    gdf = extract_geojson(tree_crowns_url)

    print("Columns in GDF:", gdf.columns.tolist())

    if "Tree_ID" in gdf.columns:
        # Ensure TREE_ID is treated as integers for matching
        gdf["Tree_ID"] = gdf["Tree_ID"].astype(int)
        ash_gdf = gdf[gdf["Tree_ID"].isin(FIDS)]

        # ❌ Delete all features before posting new ones
        delete_all_features(chat_output_url)

        # ✅ Then post
        post_features_to_layer(ash_gdf, chat_output_url)
    else:
        print("Column 'Species' not found in GeoDataFrame.")    

def long_running_task(job_id, user_task, task_name, data_locations):
    try:
        job_status[job_id] = {"status": "running", "message": "Task is in progress"}
        # Set up task and directories
        save_dir = os.path.join(os.getcwd(), task_name)
        os.makedirs(save_dir, exist_ok=True)
        # Initialize Vertex AI
        credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
        credentials = service_account.Credentials.from_service_account_info(credentials_data)
        vertexai.init(project="llmgis", location="us-central1", credentials=credentials)
        user_task = r"""1) To use a geoJSON file and return all the "Tree_ID" that are ash species ('Predicted Tree Species':'Ash').
        """
        task_name ='Tree_crown_quality'
        # Create Solution object
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

        file_path = "debug_tree_id.py"

        # Read the file content
        with open(file_path, "r") as file:
            debugged_code = file.read()

        # Store the code into solution.code_for_graph
        solution.code_for_graph = debugged_code

        exec(solution.code_for_graph)
        # Load graph file
        # solution_graph = solution.load_graph_file()
        # G = nx.read_graphml(solution.graph_file) 
        # nt = helper.show_graph(G)
        # html_name = os.path.join(os.getcwd(), solution.task_name + '.html') 

        # Generate operations
        operations = solution.get_LLM_responses_for_operations(review=False)
        solution.save_solution()
        all_operation_code_str = '\n'.join([operation['operation_code'] for operation in operations])

        # Generate assembly code
        assembly_LLM_response = solution.get_LLM_assembly_response(review=False)
        solution.assembly_LLM_response = assembly_LLM_response
        solution.save_solution()

        # Run the generated code
        model = GenerativeModel("gemini-1.5-flash-002")
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
        all_code = all_operation_code_str + '\n' + code_for_assembly + '\n' + 'assembely_solution()'
        with open('all_code_id.py', 'r') as file:
            all_code = file.read()
            
        
        print("Starting execution...")
        exec_globals = {}
        # Execute the code directly - this is the simplest approach
        exec(all_code, globals())
        result = globals().get('result', None)
        print("Final result:", result)
        filter(result)
        print("Execution completed.")
        job_status[job_id] = {"status": "completed", "message": f"Task '{task_name}' executed successfully, adding it to the map shortly."}
        
    except Exception as e: 
        job_status[job_id] = {"status": "failed", "message": str(e)}
        
@app.route('/process', methods=['POST'])
def process_request():
    try:
        # Parse request data
        request_data = request.get_json()
        user_task = request_data.get('task', "No task provided.")
        task_name = request_data.get('task_name', "default_task")

        
        data_locations = ["Tree crown geoJSON shape file: https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/TreeCrowns_Foxholes_21032025.geojson."]
        
        # Generate a unique job ID
        
        job_id = str(uuid.uuid4())
        job_status[job_id] = {"status": "queued", "message": "Task is queued for processing"}    
        
        
        # Start the task in a separate thread
        thread = threading.Thread(target=long_running_task, args=(job_id, user_task, task_name, data_locations))
        thread.start()
        
        # Change this to return the output to return the values generated. 
        return jsonify({"status": "success", "job_id": job_id, "message": "Updating ERDO..."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
        
@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """
    Endpoint to fetch the status of a background job using its job ID.
    """
    status = job_status.get(job_id, {"status": "unknown", "message": "Job ID not found"})
    return jsonify(status)
# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
