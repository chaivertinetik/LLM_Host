import os
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

tools = [
    Tool(name="ZoningLookup", func=get_zoning_info, description="Returns zoning rules..."),
    Tool(name="ClimateData", func=get_climate_info, description="Returns climate risk..."),
    Tool(name="PopulationStats", func=get_population_info, description="Returns population density...")
]

# === LangChain Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === FastAPI App ===
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

class PromptRequest(BaseModel):
    query: str
    coords: str

@app.post("/ask")
async def ask_agent(req: PromptRequest):
    full_prompt = f"{req.query} Coordinates: {req.coords}"
    try:
        response = agent.invoke({"input": full_prompt})
        return {
            "response": response.get("output", "No output returned"),
            "intermediate_steps": response.get("intermediate_steps", [])
        }
    except Exception as e:
        return {"error": str(e)}
