# --------------------- IMPORTS and helper modules ---------------------
import numpy as np 
import pandas as pd 
import re
import uvicorn
import os 
from pydantic import BaseModel
from typing import Dict
from credentials import db, parser, rag_llm, emd_model
from appagents import agent, llm, load_history, save_history, build_conversation_prompt, wants_map_output_keyword, wants_map_output_genai, wants_map_output, is_geospatial_task, clean_indentation, wants_additional_info_keyword, wants_additional_info_genai, wants_additional_info, wants_gis_task_genai, want_gis_task, prompt_suggetions, try_llm_fix, long_running_task, get_geospatial_context_tool, get_zoning_info, get_climate_info, check_tree_health, assess_tree_benefit, check_soil_suitability, get_geospatial_context, cosine_similarity, retrieve_rag_chunks, rag_tree_grants_tool, rag_tree_info_tool
from appbackend import trigger_cleanup, ClearRequest, get_project_urls, get_attr, make_project_data_locations
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware




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
   
# Global dictionary to track job statuses
job_status: Dict[str, Dict[str, str]] = {}

# Load credentials from environment variable or file
# def get_credentials():
#     credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
#     return service_account.Credentials.from_service_account_info(credentials_info)

@app.post("/clear")
async def clear_state(req: ClearRequest):
    return await trigger_cleanup(req.task_name)
       
# --------------------- Handle App process---------------------
  
@app.post("/process")
async def process_request(request_data: RequestData):
    message = ""
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
        prompt_options = prompt_suggetions(task_name, message) 
        print(prompt_options)
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
            prompt_options = prompt_suggetions(task_name, message) 
            print(prompt_options)
            
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
            prompt_options = prompt_suggetions(task_name, message) 
            print(prompt_options) 
            return { 
                "status": "completed",
                "message": message,
                "response": {
                    "role": "assistant",
                    "content": result.get("tree_ids") if isinstance(result, dict) and "tree_ids" in result else message
                }
            }
        except Exception as e:
            save_history(session_id, history)
            prompt_options = prompt_suggetions(task_name, message) 
            return {"status": "completed", "message": "Request not understood as a GIS task."}

    elif do_info: 
        response = agent.run(full_context)
        history.append({'role': 'assistant', 'content': user_task})
        history.append({'role': 'assistant', 'content': response.get("response")})
        save_history(session_id, history)
        prompt_options = prompt_suggetions(task_name, response.get("response")) 
        print(prompt_options)
        
        return {"status": "completed", "response": response.get("response")}
    
    return {"status": "completed", "message": "Request not understood as a task requiring geospatial data."}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Fetch the status of a background job using its job ID"""
    return job_status.get(job_id, {"status": "unknown", "message": "Job ID not found"})


# --------------------- run the app ---------------------
# Run the FastAPI app using Uvicorn (for Cloud Run)
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
