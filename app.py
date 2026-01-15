# --------------------- IMPORTS and helper modules ---------------------
import numpy as np
import pandas as pd
import re
import uvicorn
import os
import json
import time
import uuid
from typing import Dict, Optional, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from credentials import db, parser, emd_model
from appagents import (
    llm,
    load_history, save_history, build_conversation_prompt,
    wants_map_output_keyword, wants_map_output_genai, wants_map_output,
    is_geospatial_task,
    wants_additional_info_keyword, wants_additional_info_genai, wants_additional_info,
    wants_gis_task_genai, want_gis_task,
    prompt_suggetions,
    try_llm_fix,
    long_running_task,
    get_geospatial_context_tool, get_zoning_info, get_climate_info,
    check_tree_health, assess_tree_benefit, check_soil_suitability,
    get_geospatial_context, cosine_similarity, retrieve_rag_chunks,
    geospatial_helper,
    get_query_hash, check_firestore_for_cached_answer, store_answer_in_firestore, cache_load_helper, get_forestry_agent
)
from appbackend import trigger_cleanup, ClearRequest, get_project_urls, get_attr, make_project_data_locations, get_project_coords
from appbackend import filter as push_to_map

# Cloud Tasks (queueing) - optional
try:
    from google.cloud import tasks_v2
    from google.protobuf import timestamp_pb2
    import datetime
    _TASKS_IMPORT_OK = True
except Exception as _e:
    tasks_v2 = None
    timestamp_pb2 = None
    datetime = None
    _TASKS_IMPORT_OK = False
    _TASKS_IMPORT_ERR = str(_e)


# --------------------- Setup FASTAPI app ---------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rchahel-vertinetik.github.io"],  # Adjust for your UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestData(BaseModel):
    task: str = "No task provided."
    task_name: str = "default_task"


# --------------------- Cloud Tasks config ---------------------
# NOTE:
# - GOOGLE_CLOUD_PROJECT on Cloud Run is the PROJECT ID (e.g. "disco-parsec-444415-c4")
#   not the project number. Don’t hardcode a number.
#
# Optional env vars:
# - TASKS_LOCATION e.g. "us-central1"
# - TASKS_QUEUE e.g. "llm-dev-jobs"
# - WORKER_URL e.g. "https://<service>.run.app/run_job"
# - TASKS_INVOKER_SA e.g. "<sa>@<project>.iam.gserviceaccount.com" (only if /run_job is protected)
# - USE_QUEUE: "1" => try Cloud Tasks; "0" => always run inline (DEFAULT: 0)
#
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "disco-parsec-444415-c4").strip()
TASKS_LOCATION = os.getenv("TASKS_LOCATION", "us-central1").strip()
TASKS_QUEUE = os.getenv("TASKS_QUEUE", "llm-dev-jobs").strip()
WORKER_URL = os.getenv("WORKER_URL", "https://llmgeo-dev-1042524106019.us-central1.run.app/run_job").strip()
TASKS_INVOKER_SA = os.getenv("TASKS_INVOKER_SA", "").strip()

# IMPORTANT: default to inline until IAM is sorted
USE_QUEUE = os.getenv("USE_QUEUE", "0").strip() in ("1", "true", "True", "yes", "YES")

if not PROJECT_ID:
    print("WARN: GOOGLE_CLOUD_PROJECT is not set (Cloud Run usually sets this automatically).")
if not WORKER_URL and USE_QUEUE:
    print("WARN: WORKER_URL is not set. Queueing will fail until it's configured.")
if USE_QUEUE and not _TASKS_IMPORT_OK:
    print(f"WARN: google-cloud-tasks not available; queueing disabled. Import error: {_TASKS_IMPORT_ERR}")


# --------------------- Firestore job tracking ---------------------
def _job_ref(job_id: str):
    return db.collection("jobs").document(job_id)

def set_job(job_id: str, payload: dict, status: str, message: str = "", result: Any = None, error: str = ""):
    doc = {
        "job_id": job_id,
        "status": status,   # queued|running|completed|failed
        "message": message,
        "payload": payload,
        "error": error,
        "updated_at": (datetime.datetime.utcnow().isoformat() + "Z") if datetime else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if result is not None:
        doc["result"] = result
    _job_ref(job_id).set(doc, merge=True)

def get_job(job_id: str) -> dict:
    snap = _job_ref(job_id).get()
    if not snap.exists:
        return {"status": "unknown", "message": "Job ID not found", "job_id": job_id}
    return snap.to_dict() or {"status": "unknown", "message": "Job empty", "job_id": job_id}


# --------------------- Cloud Tasks enqueue helper ---------------------
def enqueue_cloud_task(payload: dict, job_id: Optional[str] = None, delay_seconds: int = 0) -> str:
    """
    Enqueue a Cloud Task that POSTs JSON to WORKER_URL.
    If this fails (e.g. IAM 403 cloudtasks.tasks.create), caller should fall back to inline.
    """
    if not _TASKS_IMPORT_OK or tasks_v2 is None:
        raise RuntimeError(f"Cloud Tasks not available: {_TASKS_IMPORT_ERR}")
    if not WORKER_URL:
        raise RuntimeError("WORKER_URL is not configured")

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(PROJECT_ID, TASKS_LOCATION, TASKS_QUEUE)

    job_id = job_id or str(uuid.uuid4())
    body = json.dumps({"job_id": job_id, **payload}).encode("utf-8")

    http_request: dict = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": WORKER_URL,
        "headers": {"Content-Type": "application/json"},
        "body": body,
    }

    # If your /run_job endpoint requires auth, Cloud Tasks can attach an OIDC token
    if TASKS_INVOKER_SA:
        http_request["oidc_token"] = {"service_account_email": TASKS_INVOKER_SA}

    task: dict = {"http_request": http_request}

    if delay_seconds and delay_seconds > 0:
        schedule_time = timestamp_pb2.Timestamp()
        schedule_time.FromDatetime(datetime.datetime.utcnow() + datetime.timedelta(seconds=delay_seconds))
        task["schedule_time"] = schedule_time

    client.create_task(parent=parent, task=task)
    return job_id


# --------------------- Core worker logic (shared by /run_job and inline fallback) ---------------------
def _do_heavy_work(user_task: str, task_name: str) -> dict:
    """
    Runs the heavy GIS pipeline and returns a dict with at least {"message": "..."}.
    """
    attrs = get_project_urls(task_name)
    if task_name in ["TT_GCW1_Summer", "TT_GCW1_Winter"]:
        data_locations = make_project_data_locations(task_name, include_seasons=True, attrs=attrs)
    else:
        data_locations = make_project_data_locations(task_name, include_seasons=False, attrs=attrs)

    result = long_running_task(user_task, task_name, data_locations)

    if isinstance(result, dict):
        return result
    return {"message": str(result)}


# --------------------- Clear endpoint ---------------------
@app.post("/clear")
async def clear_state(req: ClearRequest):
    return await trigger_cleanup(req.task_name)


# --------------------- Worker endpoint (Cloud Tasks calls this) ---------------------
@app.post("/run_job")
async def run_job(request: Request):
    """
    Called by Cloud Tasks (or manually).
    Runs heavy work and writes status/result to Firestore.
    DOES NOT stream logs to the user.
    """
    data = await request.json()
    job_id = data.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")

    payload = {k: v for k, v in data.items() if k != "job_id"}
    set_job(job_id, payload, status="running", message="Job is running")

    try:
        user_task = payload["task"]
        task_name = payload["task_name"]
        session_id = task_name

        result = _do_heavy_work(user_task, task_name)
        message = result.get("message", str(result))

        # Save history
        history = load_history(session_id, max_turns=10)
        history.append({'role': 'user', 'content': user_task})
        history.append({'role': 'assistant', 'content': message})
        save_history(session_id, history)

        # Persist result (keep small)
        result_payload = {
            "message": message,
            "tree_ids": result.get("tree_ids") if isinstance(result, dict) else None,
        }
        set_job(job_id, payload, status="completed", message=message, result=result_payload)
        return {"ok": True, "job_id": job_id, "status": "completed"}

    except Exception as e:
        set_job(job_id, payload, status="failed", message="Job failed", error=str(e))
        raise


# --------------------- Main API endpoint ---------------------
@app.post("/process")
async def process_request(request_data: RequestData):
    """
    UI calls this endpoint.

    Behavior (NO log streaming):
      - Non-geo or info-only tasks: run synchronously (no queue)
      - GIS operation:
          * If USE_QUEUE=1: try Cloud Tasks; if enqueue fails -> run inline and return completed
          * If USE_QUEUE=0 (default): always run inline and return completed
    """
    user_task = (request_data.task or "").strip()
    task_name = request_data.task_name
    session_id = request_data.task_name
    bbox = get_project_coords(task_name)

    if not user_task:
        raise HTTPException(status_code=400, detail="Empty task")

    # cleanup shortcuts stay synchronous
    if re.search(r"\b(clear|reset|cleanup|clean|wipe)\b", user_task.lower()):
        return await trigger_cleanup(task_name)

    history = load_history(session_id, max_turns=10)
    full_context = build_conversation_prompt(user_task, history)

    # Non-geospatial: answer immediately (no queue)
    if not is_geospatial_task(full_context):
        history.append({'role': 'user', 'content': user_task})
        history.append({'role': 'assistant', 'content': "Not programmed to do that."})
        save_history(session_id, history)

        prompt_options = prompt_suggetions(task_name, "")
        return {
            "status": "completed",
            "message": "I haven't been programmed to do that",
            "prompt_options": prompt_options,
        }

    # Determine intent
    do_gis_op = want_gis_task(user_task)
    do_info = wants_additional_info(user_task)

    # INFO-ONLY: answer immediately (no queue)
    if (not do_gis_op) and do_info:
        try: 
            content = get_forestry_agent(user_task, bbox, task_name, llm)
            # content = geospatial_helper(user_task)
            history.append({'role': 'user', 'content': user_task})
            history.append({'role': 'assistant', 'content': content})
            save_history(session_id, history)

            prompt_options = prompt_suggetions(task_name, content)
            return {
                "status": "completed",
                "response": content,
                "prompt_options": prompt_options,
            }
        except Exception as e: 
            print(f"Agent Reasoning Error: {e}")
            # Fallback to the old simple helper if the agent fails
            content = geospatial_helper(str(user_task))
    

    # GIS operation required -> queue (optional) with inline fallback
    if do_gis_op:
        payload = {
            "task": user_task,
            "task_name": task_name,
            "do_info": bool(do_info),
        }

        # If queue is enabled, attempt enqueue; otherwise inline
        if USE_QUEUE:
            job_id = str(uuid.uuid4())
            set_job(job_id, payload, status="queued", message="Job queued")

            try:
                enqueue_cloud_task(payload=payload, job_id=job_id, delay_seconds=0)
                return {
                    "status": "queued",
                    "job_id": job_id,
                    "message": "Your request has been queued.",
                    "poll": f"/status/{job_id}",
                    "prompt_options": prompt_suggetions(task_name, user_task),
                }
            except Exception as e:
                # Queue failed -> run inline (no "queued" response)
                set_job(job_id, payload, status="failed", message="Failed to enqueue job", error=str(e))

        # ---- Inline processing path ----
        try:
            # Optional: write a job record even for inline runs (helps debugging / future UI polling)
            job_id = str(uuid.uuid4())
            set_job(job_id, payload, status="running", message="Running inline")

            result = _do_heavy_work(user_task, task_name)
            message = result.get("message", str(result))

            # Save history
            history.append({'role': 'user', 'content': user_task})
            history.append({'role': 'assistant', 'content': message})
            save_history(session_id, history)

            result_payload = {
                "message": message,
                "tree_ids": result.get("tree_ids") if isinstance(result, dict) else None,
                "ran_inline": True,
            }
            set_job(job_id, payload, status="completed", message=message, result=result_payload)

            return {
                "status": "completed",
                "message": message,
                "job_id": job_id,
                "ran_inline": True,
                "prompt_options": prompt_suggetions(task_name, user_task),
            }

        except Exception as e:
            set_job(job_id, payload, status="failed", message="Inline job failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Fallback
    return {
        "status": "completed",
        "message": "Request not understood as a task requiring geospatial data.",
        "prompt_options": [],
    }


# --------------------- Status endpoint (reads from Firestore) ---------------------
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    UI polls this endpoint (only relevant if you later enable USE_QUEUE=1).
    Returns queued/running/completed/failed + message + result (if completed).
    """
    return get_job(job_id)


# --------------------- run the app ---------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
