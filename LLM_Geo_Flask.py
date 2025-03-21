from flask import Flask, request, jsonify
import os
import json
import networkx as nx
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from LLM_Geo_kernel import Solution
import helper
from flask_cors import CORS
# Initialize Flask app

app = Flask(__name__)
CORS(app)

# Load credentials from environment variable or file
def get_credentials():
    credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
    return service_account.Credentials.from_service_account_info(credentials_info)

@app.route('/process', methods=['POST'])
def process_request():
    try:
        # Parse request data
        request_data = request.get_json()
        user_task = request_data.get('task', "No task provided.")
        task_name = request_data.get('task_name', "default_task")

        # Set up task and directories
        save_dir = os.path.join(os.getcwd(), task_name)
        os.makedirs(save_dir, exist_ok=True)
        data_locations = ["Tree crown geoJSON shape file: https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/Hicks_Lodge_Trial_pred.geojson."]

        # Initialize Vertex AI
        credentials = get_credentials()
        vertexai.init(project="llmgis", location="us-central1", credentials=credentials)

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

        # Load graph file
        solution_graph = solution.load_graph_file()
        G = nx.read_graphml(solution.graph_file)
        helper.show_graph(G)

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
        response = model.generate_content(solution.assembly_prompt)
        code_for_assembly = helper.extract_code(response.text)

        # Combine all code
        all_code = all_operation_code_str + '\n' + code_for_assembly + '\n' + 'assembely_solution()'
        with open('all_code.txt', 'r') as file:
             all_code = file.read()
             
        exec(all_code)

        # Return a success response
        return jsonify({"status": "success", "task_name": task_name, "message": "Task executed successfully."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
