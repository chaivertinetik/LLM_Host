from flask import Flask, request, jsonify
import os
import json
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel
from LLM_Geo_kernel import Solution
import helper

app = Flask(__name__)

# Function to load credentials
def get_credentials():
    credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
    return service_account.Credentials.from_service_account_info(credentials_info)

@app.route("/", methods=["GET"])
def home():
    return "LLM Geo Processing API is running!"

@app.route("/process", methods=["POST"])
def process_task():
    try:
        data = request.get_json()  # Get input JSON from request
        
        task_name = data.get("task_name", "Tree_crown_quality")
        task = data.get("task", "1) To use a geoJSON file and return all the 'Tree ID' that are ash species ('Predicted Tree Species':'Ash').")
        data_locations = data.get("data_locations", ["https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/Hicks_Lodge_Trial_pred.geojson."])

        # Initialize Vertex AI
        credentials = get_credentials()
        vertexai.init(project="llmgis", location="us-central1", credentials=credentials)

        # Run your existing solution logic
        solution = Solution(
            task=task,
            task_name=task_name,
            save_dir=os.getcwd(),
            data_locations=data_locations,
        )

        # Generate response
        response_for_graph = solution.get_LLM_response_for_graph()
        solution.graph_response = response_for_graph
        solution.save_solution()

        return jsonify({"message": "Processing complete", "task_name": task_name, "response": response_for_graph})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use Cloud Run’s assigned port
    app.run(host="0.0.0.0", port=port)