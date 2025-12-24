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
    agent, llm,
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
    get_query_hash, check_firestore_for_cached_answer, store_answer_in_firestore, cache_load_helper
)
from appbackend import trigger_cleanup, ClearRequest, get_project_urls, get_attr, make_project_data_locations, get_project_coords
from appbackend import filter as push_to_map

# Cloud Tasks (queueing)
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
import datetime


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
# Required env vars for Cloud Run:
# - GOOGLE_CLOUD_PROJECT (usually set automatically)
# - TASKS_LOCATION e.g. "europe-west1"
# - TASKS_QUEUE e.g. "gis-job-queue"
# - WORKER_URL e.g. "https://<service>.run.app/run_job"
#
# Optional (recommended): make /run_job authenticated
# - TASKS_INVOKER_SA e.g. "<cloud-run-service-account>@<project>.iam.gserviceaccount.com"
#
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "1042524106019").strip()
TASKS_LOCATION = os.getenv("TASKS_LOCATION", "us-central1").strip()
TASKS_QUEUE = os.getenv("TASKS_QUEUE", "llm-dev-jobs").strip()
WORKER_URL = os.getenv("WORKER_URL", "https://llmgeo-dev-1042524106019.us-central1.run.app/run_job").strip()
TASKS_INVOKER_SA = os.getenv("TASKS_INVOKER_SA", "service-1042524106019@gcp-sa-cloudtasks.iam.gserviceaccount.com").strip()

if not PROJECT_ID:
    print("WARN: GOOGLE_CLOUD_PROJECT is not set (Cloud Run usually sets this automatically).")
if not WORKER_URL:
    print("WARN: WORKER_URL is not set. Queueing will fail until it's configured.")


# --------------------- Firestore job tracking (recommended on Cloud Run) ---------------------
# In-memory dicts won't survive instance restarts; use Firestore for status.
# Collection: jobs/{job_id}
def _job_ref(job_id: str):
    return db.collection("jobs").document(job_id)

def set_job(job_id: str, payload: dict, status: str, message: str = "", result: Any = None, error: str = ""):
    doc = {
        "job_id": job_id,
        "status": status,   # queued|running|completed|failed
        "message": message,
        "payload": payload,
        "error": error,
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
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
    Uses OIDC token if TASKS_INVOKER_SA is set (recommended).
    """
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

    # If your /run_job endpoint requires auth, Cloud Tasks should attach an OIDC token
    # for the provided service account.
    if TASKS_INVOKER_SA:
        http_request["oidc_token"] = {"service_account_email": TASKS_INVOKER_SA}

    task: dict = {"http_request": http_request}

    # Optional: schedule with a delay
    if delay_seconds and delay_seconds > 0:
        schedule_time = timestamp_pb2.Timestamp()
        schedule_time.FromDatetime(datetime.datetime.utcnow() + datetime.timedelta(seconds=delay_seconds))
        task["schedule_time"] = schedule_time

    client.create_task(parent=parent, task=task)
    return job_id


# --------------------- Clear endpoint ---------------------
@app.post("/clear")
async def clear_state(req: ClearRequest):
    return await trigger_cleanup(req.task_name)


# --------------------- Worker endpoint (Cloud Tasks calls this) ---------------------
@app.post("/run_job")
async def run_job(request: Request):
    """
    This endpoint is called by Cloud Tasks.
    It runs the heavy work and writes status/result to Firestore.

    IMPORTANT:
    - If this raises an exception, Cloud Tasks will retry (good).
    - If it returns 2xx, Cloud Tasks considers it done.
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

        # Build data locations (same logic you already have)
        attrs = get_project_urls(task_name)
        if task_name in ["TT_GCW1_Summer", "TT_GCW1_Winter"]:
            data_locations = make_project_data_locations(task_name, include_seasons=True, attrs=attrs)
        else:
            data_locations = make_project_data_locations(task_name, include_seasons=False, attrs=attrs)

        # Run heavy GIS pipeline
        result = long_running_task(user_task, task_name, data_locations)

        message = result.get("message") if isinstance(result, dict) else str(result)

        # Save history (optional)
        history = load_history(session_id, max_turns=10)
        history.append({'role': 'user', 'content': user_task})
        history.append({'role': 'assistant', 'content': message})
        save_history(session_id, history)

        # Persist result
        # Keep result payload reasonably sized; for big GeoJSON you already push_to_map separately.
        result_payload = {
            "message": message,
            "tree_ids": result.get("tree_ids") if isinstance(result, dict) else None,
        }
        set_job(job_id, payload, status="completed", message=message, result=result_payload)
        return {"ok": True, "job_id": job_id, "status": "completed"}

    except Exception as e:
        set_job(job_id, payload, status="failed", message="Job failed", error=str(e))
        # Raise so Cloud Tasks retries (unless you prefer not to retry)
        raise


# --------------------- Main API endpoint (enqueue work instead of running inline) ---------------------
@app.post("/process")
async def process_request(request_data: RequestData):
    """
    UI calls this endpoint.
    - Fast, non-blocking: it ENQUEUES a job for heavy tasks
    - Immediate responses for non-geo tasks or simple info tasks
    """
    message = ""
    user_task = (request_data.task or "").strip()
    task_name = request_data.task_name
    session_id = request_data.task_name

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

        prompt_options = prompt_suggetions(task_name, message)
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
        content = geospatial_helper(user_task)
        history.append({'role': 'user', 'content': user_task})
        history.append({'role': 'assistant', 'content': content})
        save_history(session_id, history)

        prompt_options = prompt_suggetions(task_name, content)
        return {
            "status": "completed",
            "response": content,
            "prompt_options": prompt_options,
        }

    # GIS operation required (GIS only OR GIS+info) -> queue it
    if do_gis_op:
        # Create job payload
        payload = {
            "task": user_task,
            "task_name": task_name,
            "do_info": bool(do_info),
        }

        # Create job_id and store "queued" status
        job_id = str(uuid.uuid4())
        set_job(job_id, payload, status="queued", message="Job queued")

        # Enqueue Cloud Task
        try:
            enqueue_cloud_task(payload=payload, job_id=job_id, delay_seconds=0)
        except Exception as e:
            set_job(job_id, payload, status="failed", message="Failed to enqueue job", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {e}")

        # Return immediately for UI to poll /status/{job_id}
        return {
            "status": "queued",
            "job_id": job_id,
            "message": "Your request has been queued.",
            "poll": f"/status/{job_id}",
            "prompt_options": prompt_suggetions(task_name, user_task),
        }

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
    UI polls this endpoint.
    Returns queued/running/completed/failed + message + result (if completed).
    """
    return get_job(job_id)


# --------------------- run the app ---------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
