import os
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.llms.vertexai import VertexAI

google_creds = os.environ.get("GOOGLE_CREDENTIALS")
if google_creds:
    # Write the JSON string to a temp file for Google SDK to use
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(google_creds.encode("utf-8"))
        temp_cred_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path
else:
    raise EnvironmentError(
        "Environment variable GOOGLE_APPLICATION_CREDENTIALS not found or empty. "
        "Please set it with your service account JSON content."
    )
    
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

# === Tool definitions ===
tools = [
    Tool(
        name="ZoningLookup",
        func=get_zoning_info,
        description="Returns zoning rules for coordinates like '37.7749,-122.4194'"
    ),
    Tool(
        name="ClimateData",
        func=get_climate_info,
        description="Returns climate risk for coordinates like '37.7749,-122.4194'"
    ),
    Tool(
        name="PopulationStats",
        func=get_population_info,
        description="Returns population density for coordinates like '37.7749,-122.4194'"
    )
]


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


app = FastAPI()

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
