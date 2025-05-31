import os
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.llms import VertexAI

# Load Google credentials from environment
google_creds = os.environ.get("GOOGLE_CREDENTIALS")
if google_creds:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(google_creds.encode("utf-8"))
        temp_cred_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path
else:
    raise EnvironmentError("GOOGLE_CREDENTIALS env var is missing")

# Initialize LLM
llm = VertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=0,
    max_output_tokens=1024
)

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

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === FastAPI app ===
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
        result = agent.run(full_prompt)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}
