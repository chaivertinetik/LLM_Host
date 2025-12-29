import os
import json
import vertexai
import ee
from google.oauth2 import service_account
from google.cloud import firestore
from google.cloud import aiplatform
from langchain_core.output_parsers import JsonOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
from google.oauth2.service_account import Credentials

# --------------------- SETUP and INIT---------------------

google_creds = os.environ.get("GOOGLE_CREDENTIALS")
if not google_creds:
    raise EnvironmentError("GOOGLE_CREDENTIALS env var is missing")

credentials_data = json.loads(google_creds)
credentials = service_account.Credentials.from_service_account_info(credentials_data)
# service_account_email = credentials_data.get("client_email")
# print(service_account_email)

# === Init Vertex AI ===
# vertexai.init(
#     project="disco-parsec-444415-c4",
#     location="us-east1",
#     api_endpoint="aiplatform.googleapis.com",
#     credentials=credentials
# )

aiplatform.init(
    project="disco-parsec-444415-c4",
    location="global",
    api_endpoint="aiplatform.googleapis.com", 
    credentials=credentials
)

SERVICE_ACCOUNT= 'earthengine@disco-parsec-444415-c4.iam.gserviceaccount.com'
key_path = '/tmp/earthengine-key.json'
with open(key_path, 'w') as f:
    f.write(os.environ['EARTH_CREDENTIALS'])
earth_credentials= ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_path)
ee.Initialize(earth_credentials, project='disco-parsec-444415-c4')
db = firestore.Client(project="disco-parsec-444415-c4", credentials=credentials)
parser = JsonOutputParser()
# rag_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)
hf_token = os.environ.get("HF_TOKEN")
emd_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",token=hf_token)

# emd_model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=hf_token)
