from credentials import db, emd_model, parser, rag_llm
import torch
import vertexai
import json 
import os
import helper
import networkx as nx
import datetime
import time
import ee 
import numpy as np 
from google.oauth2 import service_account
from sentence_transformers import util
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LLM
from vertexai.generative_models import GenerativeModel
from google.api_core.exceptions import ResourceExhausted

from LLM_Heroku_Kernel import Solution



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
#------------------------------------------- Firestore to store and retrieve old prompts ------------------------------------------------------

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

def prompt_suggetions(task_name:str, user_prompt:str) -> list[str]: 
    prompt_list = [
        "Show the tallest ash tree", 
        "Show the trees within a 30m range of the tallest ash tree",
        "Show the trees within a 30m range of the shortest ash tree",
        "Show the diseased ash trees", 
        "Show the unhealthy trees", 
        "Show all the trees approx 40m to green spaces", 
        "Show all the trees approx 40m to religious buildings", 
        "What is the average height of the oak trees suveryed in 2024", 
        "Sumamarize the health status of trees by species", 
        "What is the average height of the ash trees", 
        "Find the tree with the largest shape area"
    ]
    chat_doc = db.collection("chat_histories").document(task_name).get()
    old_prompts= []
    
    if chat_doc.exists: 
        data = chat_doc.to_dict()
        history = data.get("history", [])

        for i in range (len(history) -1):
            if history[i].get('role') == 'user' and history[i+1].get('role') == 'assistant':
                if "successfully" in history[i+1].get('content', '').lower(): 
                    old_prompts.append(history[i].get('content'))
                    
    combined_prompts = list(dict.fromkeys(prompt_list + old_prompts))
    user_embd= emd_model.encode(user_prompt, convert_to_tensor = True) 
    prompt_embeddings = emd_model.encode(combined_prompts, convert_to_tensor = True) 
    similarity_scores = util.pytorch_cos_sim(user_embd, prompt_embeddings)[0]
    top_results = torch.topk(similarity_scores, k=min(4, len(combined_prompts)))
    return [combined_prompts[idx] for idx in top_results.indices]


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
    message = None
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
            message = f"The task has been executed successfully, and the results should be on your screen."
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

# --------------------- Initialize agent with tools and LangChain LLM ---------------------

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)



