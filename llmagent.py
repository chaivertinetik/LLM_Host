import os
# import sys
import json
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import ee
import datetime


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
service_account_email = credentials_data.get("client_email")
print(service_account_email)

# === Init Vertex AI ===
vertexai.init(
    project="disco-parsec-444415-c4",
    location="us-central1",
    credentials=credentials
)
#testing earth engine service set up 
ee.Initialize(project='disco-parsec-444415-c4')
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
def get_geospatial_context_tool(coords: str) -> str:
    lat, lon = map(float, coords.split(","))
    context = get_geospatial_context(lat, lon)  # Your GEE function
    return json.dumps(context)
    
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

def get_geospatial_context(lat, lon):
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

tools = [
    Tool(name="ZoningLookup", func=get_zoning_info, description="Returns zoning rules..."),
    Tool(
        name="EarthEngineContext",
        func=get_geospatial_context_tool,
        description="Returns NDVI, precipitation, temperature, soil moisture, land cover, and elevation for given coordinates"
    ),
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
