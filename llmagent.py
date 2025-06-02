import os
# import sys
import json
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

# Vertex AI SDK
from vertexai import generative_models as genai
import vertexai
from google.oauth2 import service_account

# === Load credentials ===
google_creds = os.environ.get("GOOGLE_CREDENTIALS")
if not google_creds:
    raise EnvironmentError("GOOGLE_CREDENTIALS env var is missing")

credentials_data = json.loads(google_creds)
credentials = service_account.Credentials.from_service_account_info(credentials_data)

# === Init Vertex AI ===
vertexai.init(
    project="disco-parsec-444415-c4",
    location="us-central1",
    credentials=credentials
)

# === Create Gemini model ===
model = genai.GenerativeModel("gemini-2.0-flash-001")

# === Wrap Gemini in a LangChain-compatible LLM ===
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult

class GeminiLLM(LLM):
    model: genai.GenerativeModel

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"

llm = GeminiLLM(model=model)

# === Simulated tools ===
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

tools = [
    Tool(name="ZoningLookup", func=get_zoning_info, description="Returns zoning rules..."),
    Tool(name="ClimateData", func=get_climate_info, description="Returns climate risk..."),
    Tool(name="PopulationStats", func=get_population_info, description="Returns population density..."),
    Tool(name="TreeHealthCheck", func=check_tree_health, description="Assesses existing tree health at given coordinates"),
    Tool(name="TreeBenefitAssessment", func=assess_tree_benefit, description="Estimates environmental impact of planting trees"),
    Tool(name="SoilSuitability", func=check_soil_suitability, description="Checks soil type, pH, and suitability for tree planting")
]

# === LangChain Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# === FastAPI App ===
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

class PromptRequest(BaseModel):
    query: str
    coords: str

# @app.post("/ask")
# async def ask_agent(req: PromptRequest):
#     full_prompt = f"{req.query} Coordinates: {req.coords}"

#     # Capture stdout to get internal reasoning trace
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()

#     try:
#         result = agent.run(full_prompt)
#         sys.stdout = old_stdout
#         trace = mystdout.getvalue()
#         return {"response": result, "trace": trace}
#     except Exception as e:
#         sys.stdout = old_stdout
#         trace = mystdout.getvalue()
#         return {"error": str(e), "trace": trace}
        
@app.post("/ask")
async def ask_agent(req: PromptRequest):
    full_prompt = f"{req.query} Coordinates: {req.coords}"
    try:
        result = agent.run(full_prompt)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}
