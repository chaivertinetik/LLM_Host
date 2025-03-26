import os
import requests
import networkx as nx
import pandas as pd
import geopandas as gpd
import os
import vertexai
from vertexai.generative_models import GenerativeModel
from pyvis.network import Network
from openai import OpenAI
from IPython.display import display, HTML, Code
from IPython.display import clear_output
import LLM_Geo_Constants as constants
import helper
from LLM_Geo_kernel import Solution
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import json
import time
from google.api_core.exceptions import ResourceExhausted

def get_credentials():
    credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
    return service_account.Credentials.from_service_account_info(credentials_info)

isReview = False

#change this section so its using the API request sent by a user : 

task_name ='Tree_crown_quality'

TASK = r""" 1) To use a geoJSON file and return all the "Tree ID" that are ash species ('Predicted Tree Species':'Ash').
"""

DATA_LOCATIONS = ["Tree crown geoJSON shape file: https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/Hicks_Lodge_Trial_pred.geojson."]

save_dir = os.path.join(os.getcwd(), task_name)
os.makedirs(save_dir, exist_ok=True)

# create graph
# model=r"gpt-4"
# model = "gemini-1.5-flash-002"
credentials=get_credentials()
vertexai.init(project="llmgis", location="us-central1", credentials=credentials)
solution = Solution(
                    task=TASK,
                    task_name=task_name,
                    save_dir=save_dir,
                    data_locations=DATA_LOCATIONS,
                    )
print("*"*100)
print("Prompt to get solution graph:\n")
print(solution.graph_prompt)


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

solution_graph = solution.load_graph_file()


# Show the graph
G = nx.read_graphml(solution.graph_file)  
nt = helper.show_graph(G)
html_name = os.path.join(os.getcwd(), solution.task_name + '.html') 

operations = solution.get_LLM_responses_for_operations(review=isReview)
solution.save_solution()

all_operation_code_str = '\n'.join([operation['operation_code'] for operation in operations])

assembly_LLM_response = solution.get_LLM_assembly_response(review=isReview)
solution.assembly_LLM_response = assembly_LLM_response
solution.save_solution()




# TODO(developer): Update and un-comment below line
PROJECT_ID = "llmgis"
vertexai.init(project=PROJECT_ID, location="us-central1", credentials=credentials)

model = GenerativeModel("gemini-1.5-flash-002")

for attempt in range(10):
    try:
        response = model.generate_content(solution.assembly_prompt)
        break  # If successful, break out of the loop
    except ResourceExhausted:
        if attempt < 10:  # If not the last attempt
            print(f"Resource exhausted. Retrying in 10 seconds... (Attempt {attempt + 1}/10)")
            time.sleep(10)  # Wait 10 seconds before retrying
        else:
            raise

# response = model.generate_content(
#     solution.assembly_prompt
# )

code_for_assembly = helper.extract_code(response.text)


all_code = all_operation_code_str + '\n' + code_for_assembly +  '\n' + 'assembely_solution()'

with open('all_code_id.py', 'r') as file:
    all_code = file.read()

print("Starting execution...")

# Execute the code directly - this is the simplest approach
exec(all_code)

print("Execution completed.")

# display(Code(all_code, language='python'))
# output=solution.execute_complete_program(code=all_code, try_cnt=10)
