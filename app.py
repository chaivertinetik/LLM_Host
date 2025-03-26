from flask import Flask, request, jsonify
import os
import json
import networkx as nx
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from LLM_Heroku_Kernel import Solution
import helper
from flask_cors import CORS
import time
from google.api_core.exceptions import ResourceExhausted
import LLM_Geo_Constants as constants
from pyvis.network import Network
import requests
from google.oauth2.service_account import Credentials
import uuid
import threading
    
   
# Initialize Flask app

app = Flask(__name__)
CORS(app)

# Global dictionary to track job statuses
job_status = {}

# Load credentials from environment variable or file
# def get_credentials():
#     credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
#     return service_account.Credentials.from_service_account_info(credentials_info)

def long_running_task(job_id, user_task, task_name, data_locations):
    try:
        job_status[job_id] = {"status": "running", "message": "Task is in progress"}
        # Set up task and directories
        save_dir = os.path.join(os.getcwd(), task_name)
        os.makedirs(save_dir, exist_ok=True)
        # Initialize Vertex AI
        credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
        credentials = service_account.Credentials.from_service_account_info(credentials_data)
        vertexai.init(project="llmgis", location="us-central1", credentials=credentials)
        user_task = r"""1) To use a geoJSON file and return all the "Tree ID" that are ash species ('Predicted Tree Species':'Ash').
        """
        task_name ='Tree_crown_quality'
        # Create Solution object
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

        file_path = "debug_tree_id.py"

        # Read the file content
        with open(file_path, "r") as file:
            debugged_code = file.read()

        # Store the code into solution.code_for_graph
        solution.code_for_graph = debugged_code

        exec(solution.code_for_graph)
        # Load graph file
        solution_graph = solution.load_graph_file()
        G = nx.read_graphml(solution.graph_file) 
        nt = helper.show_graph(G)
        html_name = os.path.join(os.getcwd(), solution.task_name + '.html') 

        # Generate operations
        operations = solution.get_LLM_responses_for_operations(review=False)
        solution.save_solution()
        all_operation_code_str = '\n'.join([operation['operation_code'] for operation in operations])

        # Generate assembly code
        assembly_LLM_response = solution.get_LLM_assembly_response(review=False)
        solution.assembly_LLM_response = assembly_LLM_response
        solution.save_solution()

        # Run the generated code
        model = GenerativeModel("gemini-1.5-flash-002")
        for attempt in range(10):
            try: 
                response = model.generate_content(solution.assembly_prompt)
                break
            except ResourceExhausted: 
                if attempt<10:
                    time.sleep(10)
                else:
                    raise
        # response = model.generate_content(solution.assembly_prompt)
        code_for_assembly = helper.extract_code(response.text)

        # Combine all code
        all_code = all_operation_code_str + '\n' + code_for_assembly + '\n' + 'assembely_solution()'
        with open('all_code_id.py', 'r') as file:
            all_code = file.read()

        print("Starting execution...")

        # Execute the code directly - this is the simplest approach
        exec(all_code, globals())

        print("Execution completed.")
        job_status[job_id] = {"status": "completed", "message": f"Task '{task_name}' executed successfully."}
        
    except Exception as e: 
        job_status[job_id] = {"status": "failed", "message": str(e)}
        
@app.route('/process', methods=['POST'])
def process_request():
    try:
        # Parse request data
        request_data = request.get_json()
        user_task = request_data.get('task', "No task provided.")
        task_name = request_data.get('task_name', "default_task")

        
        data_locations = ["Tree crown geoJSON shape file: https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/Hicks_Lodge_Trial_pred.geojson."]
        
        # Generate a unique job ID
        
        job_id = str(uuid.uuid4())
        job_status[job_id] = {"status": "queued", "message": "Task is queued for processing"}    
        
        
        # Start the task in a separate thread
        thread = threading.Thread(target=long_running_task, args=(job_id, user_task, task_name, data_locations))
        thread.start()
        
        # Change this to return the output to return the values generated. 
        return jsonify({"status": "success", "job_id": job_id, "message": "Task has been queued. Use the status endpoint to check progress."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
        
@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """
    Endpoint to fetch the status of a background job using its job ID.
    """
    status = job_status.get(job_id, {"status": "unknown", "message": "Job ID not found"})
    return jsonify(status)
# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
