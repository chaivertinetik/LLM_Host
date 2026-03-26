import os
import json
from functools import lru_cache

import vertexai
import ee
from google.oauth2 import service_account
from google.cloud import firestore
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
from google.oauth2.service_account import Credentials

try:
    from arcgis.gis import GIS
except Exception:
    GIS = None

# --------------------- SETUP and INIT---------------------

google_creds = os.environ.get("GOOGLE_CREDENTIALS")
if not google_creds:
    raise EnvironmentError("GOOGLE_CREDENTIALS env var is missing")

credentials_data = json.loads(google_creds)
credentials = service_account.Credentials.from_service_account_info(credentials_data)

vertexai.init(
    project="disco-parsec-444415-c4",
    location="us-east1",
    credentials=credentials
)

SERVICE_ACCOUNT = 'earthengine@disco-parsec-444415-c4.iam.gserviceaccount.com'
key_path = '/tmp/earthengine-key.json'
with open(key_path, 'w') as f:
    f.write(os.environ['EARTH_CREDENTIALS'])
earth_credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_path)
ee.Initialize(earth_credentials, project='disco-parsec-444415-c4')

db = firestore.Client(project="disco-parsec-444415-c4", credentials=credentials)
parser = JsonOutputParser()
hf_token = os.environ.get("HF_TOKEN")
emd_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=hf_token)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


@lru_cache(maxsize=1)
def get_arcgis_gis():
    """
    Shared authenticated ArcGIS GIS object for the API and LLM execution paths.
    The same login is reused everywhere so ArcGIS Online / Enterprise content can be
    queried without manually managing token env vars.
    """
    if GIS is None:
        raise RuntimeError(
            "ArcGIS API for Python is not installed. Add 'arcgis' to requirements.txt."
        )

    portal_url = os.getenv("ARCGIS_PORTAL_URL", "https://www.arcgis.com").strip()
    username = os.getenv("ARCGIS_USERNAME", "").strip()
    password = os.getenv("ARCGIS_PASSWORD", "")

    if not username or not password:
        raise RuntimeError(
            "ARCGIS_USERNAME and ARCGIS_PASSWORD must be set for ArcGIS-authenticated data access."
        )

    kwargs = {
        "verify_cert": _env_bool("ARCGIS_VERIFY_CERT", True),
        "set_active": False,
        "trust_env": True,
    }
    if _env_bool("ARCGIS_USE_GEN_TOKEN", False):
        kwargs["use_gen_token"] = True

    return GIS(portal_url, username, password, **kwargs)


@lru_cache(maxsize=1)
def get_arcgis_session():
    gis = get_arcgis_gis()
    session = getattr(gis, "session", None)
    if session is None:
        raise RuntimeError("ArcGIS GIS.session is unavailable.")
    return session
