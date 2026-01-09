from credentials import db, emd_model, parser, client
# rag_llm
import torch
import vertexai
import json
import os
import helper
import networkx as nx
import datetime
import time
import ast
import textwrap
import ee
import numpy as np
from google.oauth2 import service_account
from sentence_transformers import util
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import LLM
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    # Fallback for older/unusual package structures
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
# from langchain.tools import StructuredTool
from typing import List, Optional, Any, Tuple
from pydantic import PrivateAttr, BaseModel, Field 
from vertexai.generative_models import GenerativeModel
from google.api_core.exceptions import ResourceExhausted
from appbackend import filter as push_to_map
from appbackend import get_project_coords
from LLM_Heroku_Kernel import Solution
from google import genai
from google.genai import types
from functools import wraps
from langchain_core.tools import StructuredTool
import hashlib
import io
import geopandas as gpd


# ============================================================
#                 SAFE LINT + FIX + COMPILE GATE
# ============================================================


def _strip_markdown_fences(code: str) -> str:
   code = (code or "").strip()
   if code.startswith("```"):
       # Remove leading fence line e.g. ```python
       code = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", code)
       # Remove trailing fence
       code = re.sub(r"\n```$", "", code.strip())
   return code.strip()


def _basic_normalize(code: str) -> str:
   # Preserve indentation structure (do NOT lstrip lines).
   code = _strip_markdown_fences(code)
   code = code.replace("\t", "    ")
   code = textwrap.dedent(code).strip() + "\n"
   return code


def _try_black(code: str) -> Optional[str]:
   try:
       import black
       return black.format_str(code, mode=black.FileMode())
   except Exception:
       return None


def _try_autopep8(code: str) -> Optional[str]:
   try:
       import autopep8
       return autopep8.fix_code(code, options={"aggressive": 1})
   except Exception:
       return None


def _try_ruff_fix(code: str) -> Tuple[Optional[str], List[str]]:
   """
   Best-effort: uses ruff via subprocess if installed.
   Returns (fixed_code_or_none, notes)
   """
   notes: List[str] = []
   try:
       import subprocess, tempfile, os as _os


       with tempfile.TemporaryDirectory() as td:
           path = _os.path.join(td, "snippet.py")
           with open(path, "w", encoding="utf-8") as f:
               f.write(code)


           # Autofix lint issues (non-fatal)
           subprocess.run(
               ["ruff", "check", "--fix", "--exit-zero", path],
               capture_output=True,
               text=True,
               check=False,
           )
           # Format (ruff formatter)
           subprocess.run(
               ["ruff", "format", "--exit-zero", path],
               capture_output=True,
               text=True,
               check=False,
           )


           with open(path, "r", encoding="utf-8") as f:
               fixed = f.read()


       return fixed, notes
   except Exception as e:
       notes.append(f"ruff not available or failed: {e}")
       return None, notes


def safe_lint_fix_compile(code: str) -> Tuple[bool, str, str]:
   """
   Returns (ok, code_out, message)


   Gate:
     - normalize (tabs->spaces, dedent)
     - optional ruff autofix + format
     - optional black format
     - optional autopep8 fallback
     - ast.parse + compile (catches indentation + syntax)
   """
   original = code
   code = _basic_normalize(code)


   # Try Ruff first (fix + format)
   ruff_fixed, _ = _try_ruff_fix(code)
   if ruff_fixed:
       code = ruff_fixed


   # Then Black (deterministic formatting)
   black_fixed = _try_black(code)
   if black_fixed:
       code = black_fixed


   # Fallback to autopep8 if still not parsable
   try:
       ast.parse(code)
   except SyntaxError:
       ap8 = _try_autopep8(code)
       if ap8:
           code = ap8


   # Final parse/compile gate
   try:
       ast.parse(code)
       compile(code, "<generated>", "exec")
       return True, code, "Lint/format/compile gate passed."
   except IndentationError as e:
       return False, original, f"IndentationError after fixes: {e}"
   except SyntaxError as e:
       return False, original, f"SyntaxError after fixes: {e}"
   except Exception as e:
       return False, original, f"Unexpected compile failure: {e}"


# NOTE: this is required by _strip_markdown_fences
import re


# ============================================================
#                 GIS CODE AGENT WRAPPER
# ============================================================


# class GeminiLLMWrapper(LLM):
#    _gemini_llm: Any = PrivateAttr()
#    _tools: Optional[List[Any]] = PrivateAttr(default=None)


#    def __init__(self, gemini_llm, **kwargs):
#        super().__init__(**kwargs)
#        self._gemini_llm = gemini_llm


#    def _call(self, prompt: str, stop=None, **kwargs) -> str:
#        raw_response = self._gemini_llm.generate_content(prompt)
#        print("DEBUG type:", type(raw_response.text), "value:", raw_response.text)
#        return raw_response.text if isinstance(raw_response.text, str) else str(raw_response.text)


#    def bind_tools(self, tools: List[Any]) -> "GeminiLLMWrapper":
#        self._tools = tools
#        return self


#    @property
#    def _llm_type(self) -> str:
#        return "gemini"


#    @property
#    def _identifying_params(self) -> dict:
#        return {"model": "gemini"}


# === Create Gemini model ===
model = GenerativeModel("gemini-2.5-flash")
smart_model = GenerativeModel("gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Use 1.5-flash or your preferred version
    temperature=0,
)
# llm = GeminiLLMWrapper(gemini_llm=model)


# ============================================================
#     Firestore to store and retrieve old prompts and fetch data
# ============================================================


# Temporary : need to revert to the collections-> with documents version and fix the datetime serialization error
def load_history(session_id: str, max_turns=1):
   doc = db.collection("chat_histories").document(session_id).get()
   history = doc.to_dict().get("history", []) if doc.exists else []
   return history[-2 * max_turns :]


def save_history(session_id: str, history: list):
   doc = db.collection("chat_histories").document(session_id).get()
   existing_history = doc.to_dict().get("history", []) if doc.exists else []
   combined_history = existing_history + history
   db.collection("chat_histories").document(session_id).set({"history": combined_history})


def build_conversation_prompt(new_user_prompt: str, history: list | None = None, max_turns: int = 1) -> str:
   history = history or []
   recent = history[-2 * max_turns :]
   lines = []
   for entry in recent:
       prefix = "User: " if entry.get("role") == "user" else "Assistant: "
       lines.append(f"{prefix}{entry.get('content', '')}")
   lines.append(f"User: {new_user_prompt}")
   lines.append("Assistant:")
   return "\n".join(lines)


def get_query_hash(prompt):
   return hashlib.md5(prompt.encode()).hexdigest()


def check_firestore_for_cached_answer(prompt):
   query_hash = get_query_hash(prompt)
   doc_ref = db.collection("map_history").document(query_hash)
   doc = doc_ref.get()
   if doc.exists:
       geojson_str = doc.to_dict().get("geojson_data")
       if geojson_str:
           return gpd.read_file(io.BytesIO(geojson_str.encode()))
   return None


def store_answer_in_firestore(prompt, gdf):
   query_hash = get_query_hash(prompt)
   geojson_str = gdf.to_json()
   doc_ref = db.collection("map_history").document(query_hash)
   doc_ref.set({"geojson_data": geojson_str})


# ============================================================
#                 ERDO LLM main functions
# ============================================================


def cache_load_helper(prompt: str):
   cache_prompt = (
       f"The user is asking about geospatial or forestry information: {prompt} and you were able to fetch the result "
       f"from their history, so reply telling this to the user (two or three lines max) as a GIS expert in a simple friendly way."
   )
   response = model.generate_content(cache_prompt).text.strip()
   return str(response)


def geospatial_helper(prompt: str):
   geospatial_prompt = (
       f"The user is asking about geospatial or forestry information: {prompt}. "
       f"Answer their query in simple terms (two or three lines max) as a GIS expert in a simple friendly way. "
       f"They may ask for assistance for things like how to remove ash trees safely or other diseases and pest infestations. "
       f"Pull from trusted geospatial resources and respond within these constraints as a geospatial expert in a friendly way."
   )
   # response = smart_model.generate_content(geospatial_prompt).text.strip()
   response = smart_model.generate_content(geospatial_prompt).text.strip()
   return str(response)


def long_debug(prompt: str, error: str):
   geospatial_prompt = (
       f"The user is asking about geospatial or forestry information: {prompt}. "
       f"But encountered the following error: {error}. "
       f"As a geospatial helper in simple terms (two or three lines max, dont overexplain) can you explain to the user "
       f"what the error is (in a non technical way, don't refer to the code) and how they should requery the system to not get an error (don't mention technical info about the code)"
       f"to prevent this from happening."
   )
   # response = smart_model.generate_content(geospatial_prompt).text.strip()
   response = smart_model.generate_content(geospatial_prompt).text.strip()
     
  
   return str(response)


def wants_map_output_keyword(prompt: str) -> bool:
   keywords = ["show", "display", "highlight", "visualize"]
   prompt_lower = prompt.lower()
   return any(kw in prompt_lower for kw in keywords)


def wants_map_output_genai(prompt: str) -> bool:
   # credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
   # credentials = service_account.Credentials.from_service_account_info(credentials_data)
   # vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)


   model_local = GenerativeModel("gemini-2.0-flash-001")
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
   response = model_local.generate_content(
       full_prompt,
       generation_config={"temperature": 0.0, "max_output_tokens": 20},
   )
   answer = response.text.strip().lower()
   return answer.startswith("yes")


def wants_map_output(prompt: str) -> bool:
   if wants_map_output_keyword(prompt):
       return True
   return wants_map_output_genai(prompt)


def is_geospatial_task(prompt: str) -> bool:
   """Vertex AI intent classification to determine if the task is geo-spatial related"""
   # credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
   # credentials = service_account.Credentials.from_service_account_info(credentials_data)
   # vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)


   model_local = GenerativeModel("gemini-2.0-flash-001")
   system_prompt = (
       "Decide if the user's input is related to geospatial analysis or geospatial data. "
       "This includes queries about map features, tree health, species, spatial attributes, survey date, "
       "spatial selections, overlays, or analysis."
       "Return only 'yes' or 'no'. Examples:\n"
       "- 'Find all ash trees' -> yes\n"
       "- 'What's my mother’s name?' -> no\n"
       "- 'Show healthy trees' -> yes\n"
       "- 'List all trees with a crown size over 5m' -> yes\n"
       "- 'Show areas with high NDVI in a satellite image' -> yes\n"
       "- 'What is the capital of France?' -> no"
   )


   full_prompt = f"{system_prompt}\n\nUser input: {prompt}\nAnswer:"
   response = model_local.generate_content(
       full_prompt,
       generation_config={"temperature": 0.0, "max_output_tokens": 20},
   )
   answer = response.text.strip().lower()
   return answer.startswith("yes")


# NOTE: removed unsafe clean_indentation() that lstrips every line.


def wants_additional_info_keyword(prompt: str) -> bool:
   keywords = [
       "advice",
       "explain",
       "reason",
       "why",
       "weather",
       "soil",
       "context",
       "impact",
       "effect",
       "should I do",
       "recommend",
       "suggest",
       "interpret",
       "analysis",
       "information",
       "based on",
       "because",
       "caused by",
       "influence",
       "due to",
       "assessment",
   ]
   prompt_lower = prompt.lower()
   return any(kw in prompt_lower for kw in keywords)


def wants_additional_info_genai(prompt: str) -> bool:
   # credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
   # credentials = service_account.Credentials.from_service_account_info(credentials_data)
   # vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)


   model_local = GenerativeModel("gemini-2.0-flash-001")
   system_prompt = (
       "Decide if the user's input is asking for additional geospatial explanation or advice, "
       "beyond simply showing or listing features. This includes queries about reasons, causes, impact, "
       "recommendations, interpretations, soil, weather, context, or what should be done. Return no if the task "
       "requires analyzing site data for specific features (like number of ash trees/healthy trees on a site..). "
       "Return only 'yes' or 'no'. Examples:\n"
       "- 'Show all healthy trees' -> no\n"
       "- 'How many trees are on the site' -> no\n"
       "- 'Which trees are unhealthy?' -> no\n"
       "- 'How many trees are there?' -> no\n"
       "- 'Map the largest crown' -> no\n"
       "- 'Why are many trees unhealthy?' -> yes\n"
       "- 'Give me advice based on temperature' -> yes\n"
       "- 'Should I plant here given the soil?' -> yes\n"
       "- 'What was the likely cause of tree loss?' -> yes\n"
       "- 'Explain the difference between two areas' -> yes"
   )
   full_prompt = f"{system_prompt}\n\nUser input: {prompt}\nAnswer:"
   response = model_local.generate_content(
       full_prompt,
       generation_config={"temperature": 0.0, "max_output_tokens": 20},
   )
   answer = response.text.strip().lower()
   return answer.startswith("yes")


def wants_additional_info(prompt: str) -> bool:
   if wants_additional_info_keyword(prompt):
       return True
   return wants_additional_info_genai(prompt)


def wants_gis_task_keyword(prompt: str) -> bool:
   keywords = [
       "show",
       "display",
       "map",
       "highlight",
       "visualize",
       "which trees",
       "what trees",
       "list",
       "extract",
       "buffer",
       "join",
       "select",
       "clip",
       "overlay",
       "spatial",
       "geopandas",
       "geospatial",
       "coordinates",
       "location",
       "find",
       "query",
       "identify",
   ]
   prompt_lower = prompt.lower()
   return any(kw in prompt_lower for kw in keywords)


def wants_gis_task_genai(prompt: str) -> bool:
   # credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
   # credentials = service_account.Credentials.from_service_account_info(credentials_data)
   # vertexai.init(project="disco-parsec-444415-c4", location="us-central1", credentials=credentials)


   model_local = GenerativeModel("gemini-2.0-flash-001")
   system_prompt = (
       "Decide if the user's input is asking for a geospatial operation involving spatial data processing or analysis. "
       "This includes tasks like mapping, buffering, spatial querying, extraction of features, overlays, joins, or any "
       "operation needing geospatial calculations or data manipulation. Return only 'yes' or 'no'. Examples:\n"
       "- 'Show all healthy trees' -> yes\n"
       "- 'How many trees are there' -> yes\n"
       "- 'Show the trees on the site' -> yes\n"
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
   response = model_local.generate_content(
       full_prompt,
       generation_config={"temperature": 0.0, "max_output_tokens": 20},
   )
   answer = response.text.strip().lower()
   return answer.startswith("yes")


def want_gis_task(prompt: str) -> bool:
   if wants_gis_task_keyword(prompt):
       return True
   return wants_gis_task_genai(prompt)


def prompt_suggetions(task_name: str, user_prompt: str) -> list[str]:
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
       "Find the tree with the largest shape area",
       "Show me all trees with signs of stress within 50 metres of Clumber Road East",
       "Highlight any unhealthy trees adjacent to [put road here]",
       "Show all mature trees in poor health within 100 metres of the schools",
       "What is the health of the trees bordering the school playground?",
       "Map all thriving trees within the car park areas.",
       "Show me all the ash trees that are in level 4 health in this area.",
       "Are there any (Species) trees near this road with a health level of 3 and 4?",
       "Show me all unhealthy trees in the site",
       "Generate a health report for all trees within the boundaries of the (Name) School",
       "What percentage of trees in the city centre [Specify area] are considered unhealthy?",
       "Count how many unhealthy trees are within 30m of this road",
       "List the unique species of all the healthy trees found in/within this park",
       "Compare the number of healthy trees on this side of the road versus the other side",
       "What is the average health score for trees within 20m of Cavendish Road East",
       "What is the overall health of trees in this site",
       "Show me the health status of all oak trees in this site",
       "Compare the health of street trees versus park trees in this site",
       "Is there evidence of Ash Dieback disease in the Peak District",
       "Which tree species in this site are showing the most signs of stress",
       "What are the primary environmental threats to pine trees in the Cairngorms?",
       "Identify the top 5 most vulnerable street trees in this site that require immediate inspection",
       "Where are the priority areas for monitoring in this site based on current health data",
       "What management interventions are recommended for an area showing early signs of pest infestation?",
       "What are the common visual signs of stress in a sycamore tree?",
       "Explain how satellite data is used to determine tree health.",
       "What is the most common tree species in UK urban environments?",
   ]
   chat_doc = db.collection("chat_histories").document(task_name).get()
   old_prompts = []


   if chat_doc.exists:
       data = chat_doc.to_dict()
       history = data.get("history", [])


       for i in range(len(history) - 1):
           if history[i].get("role") == "user" and history[i + 1].get("role") == "assistant":
               if "successfully" in history[i + 1].get("content", "").lower():
                   old_prompts.append(history[i].get("content"))


   combined_prompts = list(dict.fromkeys(prompt_list + old_prompts))
   user_embd = emd_model.encode(user_prompt, convert_to_tensor=True)
   prompt_embeddings = emd_model.encode(combined_prompts, convert_to_tensor=True)
   similarity_scores = util.pytorch_cos_sim(user_embd, prompt_embeddings)[0]
   top_results = torch.topk(similarity_scores, k=min(4, len(combined_prompts)))
   return [combined_prompts[idx] for idx in top_results.indices]


# ============================================================
#                     The debug agent
# ============================================================


def try_llm_fix(code, error_message=None, max_attempts=2):
   fixed_code = code
   exec_globals = {}


   for attempt in range(max_attempts):
       try:
           if error_message:
               prompt = (
                   f"The following Python code produced the error: \n"
                   f"{error_message}\n"
                   f"Please fix the code (e.g., fixing unmatched parentheses and ensuring the code compiles) "
                   f"and output only the corrected Python code:\n{fixed_code}\n"
               )
           else:
               prompt = f"Fix the following Python code and output only the corrected code:\n{fixed_code}\n"


           response = model.generate_content(prompt)
           fixed_code = helper.extract_code(response.text)


           # Gate before exec
           ok, gated, msg = safe_lint_fix_compile(fixed_code)
           if not ok:
               raise SyntaxError(msg)


           exec(gated, exec_globals)
           return True, gated
       except Exception as e:
           print(f"Error during LLM fix attempt {attempt + 1}: {e}")
           error_message = str(e)


   explanation_prompt = (
       f"The following Python code consistently failed to execute:\n{code}\n"
       f"The last error message was:\n{error_message}\n"
       f"As an expert GIS forestry assistant, explain in simple, concise, friendly terms "
       "what might be wrong and what the user can do to fix or provide clearer input. "
       "For example for a KeyError, suggest checking for typos, or making sure the data actually exists."
   )


   try:
       
        explanation = model.generate_content(explanation_prompt).text.strip()
   except Exception:
       explanation = (
           "There was an unexpected problem executing your request. "
           "Please check your input and try again."
       )


   return False, explanation


# ============================================================
#              The geospatial code llm pipeline
# ============================================================


def long_running_task(user_task: str, task_name: str, data_locations: list):
   message = None
   try:
       save_dir = os.path.join(os.getcwd(), task_name)
       os.makedirs(save_dir, exist_ok=True)


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


       # IMPORTANT: do NOT lstrip indentation.
       # If you need to normalize formatting here, use the gate:
       ok_graph, graph_code, graph_msg = safe_lint_fix_compile(solution.code_for_graph)
       if not ok_graph:
           # Attempt LLM fix once for graph stage
           success, fixed = try_llm_fix(solution.code_for_graph, error_message=graph_msg, max_attempts=1)
           if not success:
               return {"status": "completed", "message": fixed}
           ok_graph2, graph_code2, graph_msg2 = safe_lint_fix_compile(fixed)
           if not ok_graph2:
               return {"status": "completed", "message": f"Graph code failed compile gate: {graph_msg2}"}
           graph_code = graph_code2


       exec(graph_code, globals())


       # Load graph file
       solution_graph = solution.load_graph_file()
       G = nx.read_graphml(solution.graph_file)
       nt = helper.show_graph(G)
       html_name = os.path.join(os.getcwd(), solution.task_name + ".html")


       # Generate operations
       operations = solution.get_LLM_responses_for_operations(review=False)
       solution.save_solution()
       all_operation_code_str = "\n".join([operation["operation_code"] for operation in operations])


       # Generate assembly code
       assembly_LLM_response = solution.get_LLM_assembly_response(review=False)


       model_local = GenerativeModel("gemini-2.5-flash")
       for attempt in range(10):
           try:
               
               response = model_local.generate_content(solution.assembly_prompt)
               break
           except ResourceExhausted:
               if attempt < 9:
                   time.sleep(10)
               else:
                   raise


       code_for_assembly = helper.extract_code(response.text)


       print("Starting execution...")


       # ============================================================
       #              SAFE GATE: lint + fix + compile
       # ============================================================


       ok, gated_code, gate_msg = safe_lint_fix_compile(code_for_assembly)
       print("GATE:", gate_msg)


       if not ok:
           # Send through LLM fixer using gate error as context
           success, fixed_code_or_error = try_llm_fix(code_for_assembly, error_message=gate_msg)
           if not success:
               return {"status": "completed", "message": fixed_code_or_error}


           ok2, gated2, gate_msg2 = safe_lint_fix_compile(fixed_code_or_error)
           print("GATE2:", gate_msg2)
           if not ok2:
               return {"status": "completed", "message": f"Still failing compile gate: {gate_msg2}"}


           exec(gated2, globals())
           final_executed_code = gated2
       else:
           exec(gated_code, globals())
           final_executed_code = gated_code


       result = globals().get("result", None)
       print("result type:", type(result))
       print("Final result:", result)


       explanation_prompt = (
           f"For the task: {user_task}, I just ran the generated code.\n"
           f"Here's the output: {result}.\n"
           f"Explain in simple terms (one or two lines max) as a GIS expert what geospatial task was performed to obtain this result. "
           f"Do not mention or describe any code or programming; just summarize the GIS action you took in a simple friendly way "
           f"and what the result means for the user's query.\n"
       )
       explanation_response = model_local.generate_content(explanation_prompt)
       
       explanation_text = explanation_response.text.strip()


       is_empty_result = (
           result is None
           or (isinstance(result, list) and len(result) == 0)
           or (hasattr(result, "empty") and result.empty)
       )
       if is_empty_result:
           return {"status": "completed", "message": "Your query returned no data. Please check your input."}


       if wants_map_output(user_task):
           print("Execution completed.")


           if isinstance(result, str):
               message = f"{result} \n {explanation_text}"
               return {"status": "completed", "message": message}


           try:
               if hasattr(result, "to_json") and "GeoDataFrame" in str(type(result)):
                   print(result.head(2))
                   geojson = result.to_json()
                   push_to_map(geojson, task_name)


               elif isinstance(result, list):
                   preview = result[:2] if len(result) > 2 else result
                   print("Preview of list result:", preview)
                   push_to_map(result, task_name)


               else:
                   print(f"Potential unsupported result type: {type(result)}")
                   push_to_map(result, task_name)


               message = f"The task has been executed successfully, and the results should be on your screen. \n {explanation_text}"
               return {
                   "status": "completed",
                   "message": message
                   # "tree_ids": result
                   # if isinstance(result, list)
                   # else (
                   #     result.to_json()
                   #     if hasattr(result, "to_json") and "GeoDataFrame" in str(type(result))
                   #     else None
                   # ),
               }
           except Exception as map_error:
               print(f"Map push failed: {map_error}")
               return {
                   "status": "completed",
                   "message": f"The task completed but map display failed: {map_error}. \n {explanation_text}",
               }


       else:
           return {"status": "completed", "message": f"{result}. {explanation_text}"}


   except Exception as e:
       print(f"Error during execution: {e}")
       debug_response = long_debug(user_task, e)
       return f"Oops the server seems to be down! \n {debug_response}"


# ============================================================
#                     Simulated tools
# ============================================================


# def get_geospatial_context_tool(coords: str) -> str:
#    lat, lon = map(float, coords.split(","))
#    context = get_geospatial_context(lat, lon)
#    return json.dumps(context)

def get_geospatial_context_tool(bbox, project_name: str) -> dict:
    # bbox is now an ee.Geometry object, not a dict
    if bbox is None:
        return {"error": f"Missing geometry for project: {project_name}"}

    # Call the EE processing logic
    context = get_geospatial_context(bbox, project_name)
    return context

def get_zoning_info(bbox, project_name: str) -> str:
   # Ensure context tool handles the new MODIS 061 version internally
   context = get_geospatial_context_tool(bbox, project_name)
   
   land_cover = context.get("Land Cover Class (ESA)", "Unknown")
   # 'Forest Loss Year' usually comes from the Hansen GFC dataset (UMD)
   forest_loss_year = context.get("Forest Loss Year (avg)", "N/A")

   zoning_msg = f"In project area '{project_name}', the land cover class is {land_cover}."
   if forest_loss_year != "N/A":
       zoning_msg += f" Recent forest loss was observed (average year: {forest_loss_year})."
   
   zoning_msg += " Recommendation: Target reforestation in designated conservation zones."
   return zoning_msg

def get_climate_info(bbox, project_name: str) -> str:
   context = get_geospatial_context_tool(bbox, project_name)

   precip = context.get("Precipitation (mm)", 0)
   temp = context.get("Temperature (°C)", 0)
   # NDVI updated from MODIS 061
   ndvi = context.get("NDVI (mean)", 0)

   flood_risk = "High" if precip > 1000 else "Moderate" if precip > 500 else "Low"

   return (
       f"Climate summary for {project_name}:\n"
       f"- Precipitation: {precip} mm (Flood Risk: {flood_risk})\n"
       f"- Mean Temperature: {temp} °C\n"
       f"- Vegetation Health (NDVI): {ndvi:.2f}\n"
       f"- Sea-level rise vulnerability is currently estimated at 1.2m over coming decades."
   )

def check_tree_health(bbox, project_name: str) -> dict:
   context = get_geospatial_context_tool(bbox, project_name)
   
   ndvi = context.get("NDVI (mean)", 0)
   # Soil Moisture updated from NASA/SMAP/SPL4SMGP/007
   soil_m = context.get("Soil Moisture (m3/m3)", 0)
   
   health_comment = "Healthy canopy" if ndvi > 0.5 else "Canopy thinning or stress detected"
   drought_comment = "Low drought stress" if soil_m > 0.25 else "Signs of drought stress"
   
   return {
       "Project": project_name,
       "Canopy NDVI": f"{ndvi:.2f}",
       "Soil Moisture": f"{soil_m:.2f}",
       "Assessment": f"{health_comment}; {drought_comment}",
       "Recommendation": "Monitor for canopy decline; prioritize supplemental watering for stressed zones."
   }

def check_soil_suitability(bbox, project_name: str) -> str:
   context = get_geospatial_context_tool(bbox, project_name)

   soil_moisture = context.get("Soil Moisture (m3/m3)", 0)
   elevation = context.get("Elevation (m)", 0)
   land_cover = context.get("Land Cover Class (ESA)", "Unknown")

   if 0.2 <= soil_moisture <= 0.4:
       moisture_msg = "Soil moisture is in the ideal range for native tree species."
   else:
       moisture_msg = "Soil moisture is outside the ideal range; consider irrigation or drought-resistant species."

   return (
       f"Soil suitability for {project_name}:\n"
       f"- {moisture_msg} (Current: {soil_moisture:.2f})\n"
       f"- Elevation: {elevation} m\n"
       f"- Land Cover Type: {land_cover}"
   )


def assess_tree_benefit(bbox, project_name: str) -> dict:
   # 'bbox' is now the ee.Geometry object passed from the shield
   geo = get_geospatial_context_tool(bbox, project_name)
   
   # Using updated NDVI from MODIS/061/MOD13Q1
   ndvi = geo.get("NDVI (mean)", 0)
   precip = geo.get("Precipitation (mm)", 0)
   
   # Logic remains the same, but now uses the version 061 data
   benefit = "Excellent" if ndvi > 0.7 and precip > 600 else "Moderate"
   
   # Land Cover check (now pulling from versioned ESA WorldCover 2021/2026 data)
   land_cover = geo.get("Land Cover Class (ESA)")
   cooling = "Substantial" if land_cover == "Forest" else "Potential"
   
   return {
       "Project": project_name,
       "Carbon Capture Potential": benefit,
       "Shade Impact": f"{cooling} cooling effect observed/potential.",
       "NDVI Reference": round(ndvi, 3) # Added rounding for cleaner agent output
   }

def get_geospatial_context(geometry, location_label: str):
    # geometry is already an ee.Geometry.Rectangle passed from the shield
    center_point = geometry.centroid(maxError=1)

    today = datetime.date.today()
    start_date = ee.Date.fromYMD(today.year, 1, 1)
    end_date = ee.Date.fromYMD(today.year, today.month, today.day)

    def fetch_stat(collection_id, selector, scale):
        try:
            img = ee.ImageCollection(collection_id).filterBounds(geometry).filterDate(start_date, end_date).select(selector).mean()
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                bestEffort=True
            ).getInfo()
            return stats if stats else {}
        except Exception: return {}

    # Datasets updated for 2026
    ndvi_data = fetch_stat("MODIS/061/MOD13Q1", "NDVI", 250)
    precip_data = fetch_stat("UCSB-CHG/CHIRPS/DAILY", "precipitation", 5000)
    soil_data = fetch_stat("NASA/SMAP/SPL4SMGP/008", "sm_surface", 1000)

    # Elevation & Landcover
    elev_img = ee.Image("USGS/SRTMGL1_003")
    elevation = elev_img.reduceRegion(ee.Reducer.mean(), geometry, 30).getInfo()
    lc_img = ee.Image("ESA/WorldCover/v200/2021")
    landcover = lc_img.reduceRegion(ee.Reducer.first(), center_point, 10).getInfo()

    return {
        "Project": location_label,
        "NDVI (mean)": round((ndvi_data.get("NDVI") or 0) / 10000.0, 3),
        "Precipitation (mm)": round((precip_data.get("precipitation") or 0) * 30, 2),
        "Soil Moisture (m3/m3)": round(soil_data.get("sm_surface") or 0, 3),
        "Elevation (m)": round(elevation.get("elevation") or 0, 1),
        "Land Cover Class (ESA)": interpret_esa_code(landcover.get("Map"))
    }

def interpret_esa_code(code):
    """Helper to turn ESA numbers into readable names for the LLM."""
    mapping = {10: "Trees", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare/Sparse", 70: "Snow/Ice", 80: "Water", 90: "Wetland", 95: "Mangroves", 100: "Moss/Lichen"}
    return mapping.get(code, "Unknown")


# ============================================================
#        Initialize agent with tools and LangChain LLM
# ============================================================

# def sanitize_input(q):
#     """Ensures the agent input is a string, even if the LLM sends a list."""
#     if isinstance(q, list):
#         return str(q[0]) if q else ""
#     return str(q)

# def get_forestry_agent(bbox:dict, task_name: str, llm):
#     tools = [
#     Tool(
#         name="ZoningLookup",
#         func=lambda q: get_zoning_info(bbox=bbox, project_name=task_name),
#         description="Use this ONLY for zoning-related land cover and forest loss info as a proxy to guide tree planting recommendations. Not for general advice.",
#     ),
#     Tool(
#         name="ClimateLookUp",
#         func=lambda q: get_climate_info(bbox=bbox, project_name=task_name),
#         description="Returns precipitation, temperature, vegetation health (NDVI), flood risk, and sea level rise estimates for forestry planning.",
#     ),
#     Tool(
#         name="CheckTreeHealth",
#         func=lambda q: check_tree_health(bbox=bbox, project_name=task_name),
#         description="Assess how healthy the trees are using the canopy cover and soil.",
#     ),
#     Tool(
#         name="SoilSuitabilityCheck",
#         func=lambda q: check_soil_suitability(bbox=bbox, project_name=task_name),
#         description="Analyzes soil moisture, elevation, and land cover to evaluate suitability for native tree species planting.",
#     ),
#     Tool(
#         name="TreeBenefitAssessment",
#         func=lambda q: assess_tree_benefit(bbox=bbox, project_name=task_name),
#         description="Estimates carbon capture potential and cooling benefits based on NDVI, precipitation, and land cover data.",
#     ),
#     Tool(
#                 name="GeneralGeospatialExpert",
#                 func=lambda q: geospatial_helper(sanitize_input(q)), 
#                 description="Use this for general geospatial or forestry questions, like tree removal, pests, diseases, or best practices. Use this when the user asks 'how' or 'why' rather than 'what is the data'.",
#             )
#     ]

#     return create_react_agent(llm,tools)

def cosine_similarity(a, b):
   return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_rag_chunks(collection_name, query, top_k=5):
   root_ref = db.collection("knowledge_chunks").document("root")
   chunks_ref = root_ref.collection(collection_name).stream()


   query_emb = emd_model.encode([query])[0]


   scored_chunks = []
   for doc in chunks_ref:
       chunk = doc.to_dict()
       emb = chunk.get("embedding", None)
       if emb is not None:
           emb_np = np.array(emb)
           sim = cosine_similarity(query_emb, emb_np)
           scored_chunks.append((sim, chunk))


   scored_chunks.sort(key=lambda x: x[0], reverse=True)
   top_contents = [chunk["content"] for _, chunk in scored_chunks[:top_k]]
   return top_contents


def prompt_template(query: str, context: str, format_instructions: str) -> str:
   prompt = (
       "Use the following forestry data extracted from documents:\n"
       f"{context}\n\n"
       "Answer the query with geospatial reasoning:\n"
       f"{query}\n\n"
       f"{format_instructions}\n"
       "Return only valid JSON."
   )
   return prompt


def sanitize_input(q):
    """Shields the tool from list/object inputs by forcing a string return."""
    if isinstance(q, list):
        return str(q[0]) if q else ""
    return str(q)

class ToolInput(BaseModel):
    tool_input: str = Field(description="The query or input string")

def get_forestry_agent(bbox_dict: dict, task_name: str, llm):
    # Create the GEE Geometry once
    bbox_geom = ee.Geometry.Rectangle(
        coords=[bbox_dict['xmin'], bbox_dict['ymin'], bbox_dict['xmax'], bbox_dict['ymax']],
        proj=f'EPSG:{bbox_dict.get("wkid", 4326)}', 
        geodesic=False, evenOdd=True 
    )

    # 2. Define tools using StructuredTool to force string validation
    tools = [
        StructuredTool.from_function(
            name="ZoningLookup",
            func=lambda **kwargs: get_zoning_info(bbox=bbox_geom, project_name=task_name),
            description="Use for zoning, land cover, and forest loss info.",
            args_schema=ToolInput
        ),
        StructuredTool.from_function(
            name="ClimateLookUp",
            func=lambda **kwargs: get_climate_info(bbox=bbox_geom, project_name=task_name),
            description="Returns precipitation, temperature, and flood risk.",
            args_schema=ToolInput
        ),
        StructuredTool.from_function(
            name="CheckTreeHealth",
            func=lambda **kwargs: check_tree_health(bbox=bbox_geom, project_name=task_name),
            description="Assess tree health using canopy cover.",
            args_schema=ToolInput
        ),
        StructuredTool.from_function(
            name="SoilSuitabilityCheck",
            func=lambda **kwargs: check_soil_suitability(bbox=bbox_geom, project_name=task_name),
            description="Analyzes soil and land cover suitability.",
            args_schema=ToolInput
        ),
        StructuredTool.from_function(
            name="TreeBenefitAssessment",
            func=lambda **kwargs: assess_tree_benefit(bbox=bbox_geom, project_name=task_name),
            description="Estimates carbon and cooling benefits.",
            args_schema=ToolInput
        ),
        StructuredTool.from_function(
            name="GeneralGeospatialExpert",
            func=lambda query: geospatial_helper(str(query)),
            description="Use for questions about pests, diseases, or forestry advice.",
            args_schema=ToolInput
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a forestry expert. Use tools to provide data-driven advice."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)




#this should happen internally now! 
# agent = create_react_agent(
#    model=llm.bind_tools(tools),
#    tools=tools,
# )



