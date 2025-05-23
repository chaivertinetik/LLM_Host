import LLM_Geo_Constants as constants
import helper
import os
import requests
import networkx as nx
import pandas as pd
import geopandas as gpd
# from pyvis.network import Network
from openai import OpenAI
import configparser
import pickle
import time
import sys
import traceback
import os
import vertexai
from vertexai.generative_models import GenerativeModel
import textwrap
from google.oauth2 import service_account
import json
from google.oauth2.service_account import Credentials


#load config
# config = configparser.ConfigParser()
# config.read('config.ini')

# use your KEY.
# OpenAI_key = config.get('API_Key', 'OpenAI_key')
# client = OpenAI(api_key=OpenAI_key)

import google.auth

# def get_credentials():
#     """Load Google credentials using Application Default Credentials (ADC)."""
#     credentials, project = google.auth.default()
#     return credentials
# def get_credentials():
#  # For GitHub Actions - load from environment secret
#  credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
#  return service_account.Credentials.from_service_account_info(credentials_info)
# Initialize Vertex AI with the credentials



credentials_data = json.loads(os.getenv("GOOGLE_CREDENTIALS"))

# Create credentials object
#Credentials.from_service_account_info(credentials_data)
credentials = service_account.Credentials.from_service_account_info(credentials_data)


class Solution():
    """
    class for the solution. Carefully maintain it.  

    """
    def __init__(self,
                 task,
                 task_name,
                 save_dir,
                 role=constants.graph_role,  # Default role
                 #gemini-1.5-flash-002
                 model="gemini-2.0-flash-001",
                 data_locations=[],
                 stream=True,
                 verbose=True,
                 project_id="disco-parsec-444415-c4",
                 location="us-central1", 
                 credentials=credentials):
        self.task = task
        self.solution_graph = None
        self.graph_response = None
        self.role = role
        self.data_locations = data_locations
        self.task_name = task_name
        self.save_dir = save_dir
        self.code_for_graph = ""
        self.graph_file = os.path.join(self.save_dir, f"{self.task_name}.graphml")
        self.source_nodes = None
        self.sink_nodes = None
        self.operations = []  # each operation is an element:
        # {node_name: "", function_descption: "", function_definition:"", return_line:""
        # operation_prompt:"", operation_code:""}
        self.assembly_prompt = ""

        self.parent_solution = None
        self.model_name = "gemini-2.0-flash-001"
        self.model = GenerativeModel("gemini-2.0-flash-001")
        self.stream = stream
        self.verbose = verbose
        self.assembly_LLM_response = ""
        self.code_for_assembly = ""
        self.graph_prompt = ""

        self.data_locations_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(self.data_locations)])

        # Example Graph Requirement (you might need to update this with your constants module)
        graph_requirement = constants.graph_requirement.copy()
        graph_requirement.append(f"Save the network into GraphML format, save it at: {self.graph_file}")
        graph_requirement_str =  '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(graph_requirement)])
        
        graph_prompt = f'Your role: {self.role} \n\n' + \
               f'Your task: {constants.graph_task_prefix} \n {self.task} \n\n' + \
               f'Your reply needs to meet these requirements: \n {graph_requirement_str} \n\n' + \
               f'Your reply example: {constants.graph_reply_exmaple} \n\n' + \
               f'Data locations (each data is a node): {self.data_locations_str} \n'
        self.graph_prompt = graph_prompt

        self.direct_request_LLM_response = ""
        self.direct_request_code = ""

        self.chat_history = [{'role': 'system', 'content': role}]
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location, credentials=credentials)
        # self.model = GenerativeModel(model_name)
    # def __init__(self, 
    #              task, 
    #              task_name,
    #              save_dir,                
    #              role=constants.graph_role,
    #              model=r"gpt-4o",
    #              data_locations=[],
    #              stream=True,
    #              verbose=True,
    #             ):        
    #     self.task = task        
    #     self.solution_graph = None
    #     self.graph_response = None
    #     self.role = role
    #     self.data_locations=data_locations
    #     self.task_name = task_name   
    #     self.save_dir = save_dir
    #     self.code_for_graph = ""
    #     self.graph_file = os.path.join(self.save_dir, f"{self.task_name}.graphml")
    #     self.source_nodes = None
    #     self.sink_nodes = None
    #     self.operations = []  # each operation is an element:
    #     # {node_name: "", function_descption: "", function_definition:"", return_line:""
    #     # operation_prompt:"", operation_code:""}
    #     self.assembly_prompt = ""
        
    #     self.parent_solution = None
    #     self.model = model
    #     self.stream = stream
    #     self.verbose = verbose

    #     self.assembly_LLM_response = ""
    #     self.code_for_assembly = ""
    #     self.graph_prompt = ""
         
    #     self.data_locations_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(self.data_locations)])     
        
    #     graph_requirement = constants.graph_requirement.copy()
    #     graph_requirement.append(f"Save the network into GraphML format, save it at: {self.graph_file}")
    #     graph_requirement_str =  '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(graph_requirement)])
        
    #     graph_prompt = f'Your role: {self.role} \n\n' + \
    #            f'Your task: {constants.graph_task_prefix} \n {self.task} \n\n' + \
    #            f'Your reply needs to meet these requirements: \n {graph_requirement_str} \n\n' + \
    #            f'Your reply example: {constants.graph_reply_exmaple} \n\n' + \
    #            f'Data locations (each data is a node): {self.data_locations_str} \n'
    #     self.graph_prompt = graph_prompt

    #     # self.direct_request_prompt = ''
    #     self.direct_request_LLM_response = ''
    #     self.direct_request_code = ''

    #     self.chat_history = [{'role': 'system', 'content': role}]
    """
    Generate a response from the Vertex AI Generative Model.

    Args:
        prompt (str): The input prompt for the LLM.
        verbose (bool): Whether to print progress and responses.
        temperature (float): The creativity level for the response.
        retry_cnt (int): Number of retries in case of failure.
        sleep_sec (int): Seconds to wait before retrying.
        system_role (str): Optional system role for the assistant.

    Returns:
        str: The generated response content.
    """
    def __getstate__(self):
        """
        Return the serializable state of the object.
        Exclude `self.model` or any other non-serializable attributes.
        """
        state = self.__dict__.copy()
        state['model'] = None  # Exclude the model from serialization
        return state

    def __setstate__(self, state):
        """
        Restore the object's state.
        """
        self.__dict__.update(state)
        # You can reinitialize the model here if needed, e.g.,
        # self.model = initialize_model()
        
    def get_LLM_reply(self,
                  prompt,
                  verbose=True,
                  temperature=1,
                  retry_cnt=10,
                  sleep_sec=20,
                  system_role=None):
        
        model = "gemini-2.0-flash-001"
        if system_role is None:
            system_role = self.role
        count = 0
        isSucceed = False
        self.chat_history.append({'role': 'user', 'content': prompt})
        response_text = ""
        sleep_time = sleep_sec 
        while not isSucceed and count < retry_cnt:
            try:
                count += 1
                if verbose:
                    print(f"Attempt {count}: Generating response...")
            
                # Call the model with the correct argument format
                response = self.model.generate_content(prompt)  # Pass the prompt directly as a string
                isSucceed = True  # Successfully generated response
                response_text = response.text  # Extract the text from the response

            except Exception as e:
                if "quota exceeded" in str(e).lower():
                    print(f"Error: {e}. Backoff for {sleep_time} seconds, then retry {count}/{retry_cnt}.")
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time * 2, 120)  # Exponential backoff capped at 120 seconds
                else:
                    print(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n", e)
                    time.sleep(sleep_sec)

        if not isSucceed:
            raise RuntimeError("Failed to get a response from the model after multiple retries.")

        # Save the generated response to chat history
        self.chat_history.append({'role': 'assistant', 'content': response_text})

        if verbose:
            print(f"\nResponse:\n{response_text}\n")

        return response_text

    # def get_LLM_reply(self,
    #               prompt,
    #               verbose=True,
    #               temperature=1,
    #               retry_cnt=3,
    #               sleep_sec=10,
    #               system_role=None):
    #     model="gemini-1.5-flash-002"
    #     if system_role is None:
    #         system_role = self.role
    #     count = 0
    #     isSucceed = False
    #     self.chat_history.append({'role': 'user', 'content': prompt})
    #     response_text = ""
        
    #     while not isSucceed and count < retry_cnt:
    #         try:
    #             count += 1
    #             if verbose:
    #                 print(f"Attempt {count}: Generating response...")
            
    #         # Generate content using Vertex AI GenerativeModel
    #             response = self.model.generate_content(
    #             prompt=prompt,
    #             temperature=temperature
    #             )
    #             isSucceed = True  # Successfully generated response
    #             response_text = response.text

    #         except Exception as e:
    #             print(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n", e)
    #             time.sleep(sleep_sec)

    #     if not isSucceed:
    #         raise RuntimeError("Failed to get a response from Vertex AI after multiple retries.")

    #     # Save the generated response to chat history
    #     self.chat_history.append({'role': 'assistant', 'content': response_text})

    #     if verbose:
    #         print(f"\nResponse:\n{response_text}\n")

    #     return response_text

    

    def get_LLM_response_for_graph(self, execuate=True):
        # self.chat_history.append()
        response = self.get_LLM_reply(prompt=self.graph_prompt,
                                        system_role=self.role
                                        # model=GenerativeModel("gemini-2.0-flash-001"),
                                         )
        self.graph_response = response
        try:
            self.code_for_graph = helper.extract_code(response=self.graph_response, verbose=False)
            # Normalize the indentation
            self.code_for_graph = textwrap.dedent(self.code_for_graph)
        except Exception as e:
            self.code_for_graph = ""
            print("Failed to extract graph Python code from LLM response:", e)
    
        if execuate:
            try:
                exec(self.code_for_graph)
                self.load_graph_file()
            except IndentationError as e:
                print("IndentationError in generated code:", e)
            except Exception as e:
                print("Error executing generated code:", e)
        # try:
        #     self.code_for_graph = helper.extract_code(response=self.graph_response, verbose=False)
        # except Exception as e:
        #     self.code_for_graph = ""
        #     print("Extract graph Python code rom LLM failed.")
        # if execuate:
        #     exec(self.code_for_graph)
        #     self.load_graph_file()
        return self.graph_response
        
    def load_graph_file(self, file=""):
        G = None
        if os.path.exists(file):
            self.graph_file = file
            G = nx.read_graphml(self.graph_file)
        else:
            
            if file == "" and os.path.exists(self.graph_file):
                G = nx.read_graphml(self.graph_file)
            else:
                print("Do not find the given graph file:", file)
                return None
        
        self.solution_graph = G
        
        self.source_nodes = helper.find_source_node(self.solution_graph)
        self.sink_nodes = helper.find_sink_node(self.solution_graph)
         
        return self.solution_graph 

    @property
    def operation_node_names(self):
        opera_node_names = []
        assert self.solution_graph, "The Soluction class instance has no solution graph. Please generate the graph"
        for node_name in self.solution_graph.nodes():
            node = self.solution_graph.nodes[node_name]
            if node['node_type'] == 'operation':
                opera_node_names.append(node_name)
        return opera_node_names

    def get_ancestor_operations(self, node_name):
        ancestor_operation_names = []
        ancestor_node_names = nx.ancestors(self.solution_graph, node_name)
        # for ancestor_node_name in ancestor_node_names:
        ancestor_operation_names = [node_name for node_name in ancestor_node_names if node_name in self.operation_node_names]

        ancestor_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in ancestor_operation_names:
                ancestor_operation_nodes.append(oper)

        return ancestor_operation_nodes

    def get_descendant_operations(self, node_name):
        descendant__operation_names = []
        descendant_node_names = nx.descendants(self.solution_graph, node_name)
        # for descendant_node_name in descendant_node_names:
        descendant__operation_names = [node_name for node_name in descendant_node_names if node_name in self.operation_node_names]
        # descendant_codes = '\n'.join([oper['operation_code'] for oper in descendant_node_names])
        descendant_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in descendant__operation_names:
                descendant_operation_nodes.append(oper)

        return descendant_operation_nodes

    def get_descendant_operations_definition(self, descendant_operations):

        keys = ['node_name', 'description', 'function_definition', 'return_line']
        operation_def_list = []
        for node in descendant_operations:
            operation_def = {key: node[key] for key in keys}
            operation_def_list.append(str(operation_def))
        defs = '\n'.join(operation_def_list)
        return defs

    def get_prompt_for_an_opearation(self, operation):
        assert self.solution_graph, "Do not find solution graph!"
        # operation_dict = function_def.copy()

        node_name = operation['node_name']

        # get ancestors code
        ancestor_operations = self.get_ancestor_operations(node_name)
        ancestor_operation_codes = '\n'.join([oper['operation_code'] for oper in ancestor_operations])
        descendant_operations = self.get_descendant_operations(node_name)
        descendant_defs = self.get_descendant_operations_definition(descendant_operations)
        descendant_defs_str = str(descendant_defs)

        pre_requirements = [
            f'The function description is: {operation["description"]}',
            f'The function definition is: {operation["function_definition"]}',
            f'The function return line is: {operation["return_line"]}'
        ]

        operation_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            pre_requirements + constants.operation_requirement)])

        operation_prompt = f'Your role: {constants.operation_role} \n\n' + \
                           f'operation_task: {constants.operation_task_prefix} {operation["description"]} \n\n' + \
                           f'This function is one step to solve the question/task: {self.task} \n\n' + \
                           f"This function is a operation node in a solution graph for the question/task, the Python code to build the graph is: \n{self.code_for_graph} \n\n" + \
                           f'Data locations: {self.data_locations_str} \n\n' + \
                           f'Your reply example: {constants.operation_reply_exmaple} \n\n' + \
                           f'Your reply needs to meet these requirements: \n {operation_requirement_str} \n\n' + \
                           f"The ancestor function code is (need to follow the generated file names and attribute names): \n {ancestor_operation_codes} \n\n" + \
                           f"The descendant function (if any) definitions for the question are (node_name is function name): \n {descendant_defs_str}"

        operation['operation_prompt'] = operation_prompt
        return operation_prompt
        # self.operations.append(operation_dict)
    # def get_prompts_for_operations(self):  ######## Not use ###########
    #     assert self.solution_graph, "Do not find solution graph!"
    #     def_list, data_node_list = helper.generate_function_def_list(self.solution_graph)
    #
    #
    #     for idx, function_def in enumerate(def_list):
    #         operation_dict = function_def.copy()
    #
    #         node_name = function_def['node_name']
    #
    #         # get ancestors code
    #         ancestor_operations = self.get_ancestor_operations(node_name)
    #         ancestor_operation_codes = '\n'.join([oper['operation_code'] for oper in ancestor_operations])
    #         descendant_operations = self.get_descendant_operations(node_name)
    #         descendant_defs = self.get_descendant_operations_definition(descendant_operations)
    #
    #         pre_requirements = [
    #                             f'The function description is: {function_def["description"]}',
    #                             f'The function definition is: {function_def["function_definition"]}',
    #                             f'The function return line is: {function_def["return_line"]}'
    #                            ]
    #
    #         operation_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
    #             pre_requirements + constants.operation_requirement)])
    #
    #         operation_prompt = f'Your role: {constants.operation_role} \n' + \
    #                            f'operation_task: {constants.operation_task_prefix} {function_def["description"]} \n' + \
    #                            f'This function is one step to solve the question/task: {self.task} \n' + \
    #                            f"This function is a operation node in a solution graph for the question/task, the Python code to build the graph is: \n{self.code_for_graph} \n" + \
    #                            f'Data locations: {self.data_locations_str} \n' + \
    #                            f'Reply example: {constants.operation_reply_exmaple} \n' + \
    #                            f'Your reply needs to meet these requirements: \n {operation_requirement_str} \n \n' + \
    #                            f"The ancestor function code is (need to follow the generated file names and attribute names): \n {ancestor_operation_codes}" + \
    #                            f"The descendant function definitions for the question are (node_name is function name): \n {descendant_defs}"
    #
    #
    #         operation_dict['operation_prompt'] = operation_prompt
    #         self.operations.append(operation_dict)
    #     return self.operations

    # initial the oepartion list
    def initial_operations(self):
        self.operations = []
        operation_names = self.operation_node_names
        for node_name in operation_names:
            function_def_returns = helper.generate_function_def(node_name, self.solution_graph)
            self.operations.append(function_def_returns)
    def get_LLM_responses_for_operations(self, review=True):
        # def_list, data_node_list = helper.generate_function_def_list(self.solution_graph)
        self.initial_operations()
        for idx, operation in enumerate(self.operations):
            node_name = operation['node_name']
            print(f"{idx + 1} / {len(self.operations)}, LLM is generating code for operation node: {operation['node_name']}")
            prompt = self.get_prompt_for_an_opearation(operation)

            response = self.get_LLM_reply(
                          prompt=prompt,
                          system_role=constants.operation_role
                          
                          # model=r"gpt-4",
                         )
            # print(response)
            operation['response'] = response
            try:
                operation_code = helper.extract_code(response=operation['response'], verbose=False)
                # print("operation_code:", operation_code)
            except Exception as e:
                operation_code = ""
            operation['operation_code'] = operation_code

            if review:
                operation = self.ask_LLM_to_review_operation_code(operation)
            
        return self.operations


    def prompt_for_assembly_program(self):
        all_operation_code_str = '\n'.join([operation['operation_code'] for operation in self.operations])
        # operation_code = solution.operations[-1]['operation_code']
        # assembly_prompt = f"" + \

        assembly_requirement = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(constants.assembly_requirement)])

        assembly_prompt = f"Your role: {constants.assembly_role} \n\n" + \
                          f"Your task is: use the given Python functions, return a complete Python program to solve the question: \n {self.task}" + \
                          f"Requirement: \n {assembly_requirement} \n\n" + \
                          f"Data location: \n {self.data_locations_str} \n" + \
                          f"Code: \n {all_operation_code_str}"
        
        self.assembly_prompt = assembly_prompt
        return self.assembly_prompt
    
    
    def get_LLM_assembly_response(self, review=True):
        self.prompt_for_assembly_program()
        assembly_LLM_response = helper.get_LLM_reply(prompt=self.assembly_prompt,
                          system_role=constants.assembly_role,
                          retry_cnt=10,
                          sleep_sec=20
                          # model=r"gpt-4",
                         )
        self.assembly_LLM_response = assembly_LLM_response
        self.code_for_assembly = helper.extract_code(self.assembly_LLM_response)
        
        try:
            code_for_assembly = helper.extract_code(response=self.assembly_LLM_response, verbose=False)
        except Exception as e:
                code_for_assembly = ""
                
        self.code_for_assembly = code_for_assembly

        if review:
            self.ask_LLM_to_review_assembly_code()
        
        return self.assembly_LLM_response
    
    def save_solution(self):
#         , graph=True
        new_name = os.path.join(self.save_dir, f"{self.task_name}.pkl")
        with open(new_name, "wb") as f:
            pickle.dump(self, f)
            
    def load_solution(file_path):
        with open(file_path, "rb") as f:
            solution = pickle.load(f)
        return solution
    
    def get_solution_at_one_time(self):
        pass

    @property
    def direct_request_prompt(self):

        direct_request_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            constants.direct_request_requirement)])

        direct_request_prompt = f'Your role: {constants.direct_request_role} \n' + \
                                f'Your task: {constants.direct_request_task_prefix} to address the question or task: {self.task} \n' + \
                           f'Location for data you may need: {self.data_locations_str} \n' + \
                           f'Your reply needs to meet these requirements: \n {direct_request_requirement_str} \n'
        return direct_request_prompt



    def get_direct_request_LLM_response(self, review=True):
        if not hasattr(self, 'model') or self.model is None:
            raise AttributeError("Model not initialized. Ensure the `model` attribute is set in the `__init__` method.")

        # Generate content using the model
        response = self.model.generate_content(self.direct_request_prompt)
   

        self.direct_request_LLM_response = response.text
        self.direct_request_code = helper.extract_code(response.text)
        if review:
            self.ask_LLM_to_review_direct_code()

        return self.direct_request_LLM_response
        
    # def get_direct_request_LLM_response(self, review=True):

    #     response = helper.get_LLM_reply(prompt=self.direct_request_prompt,
    #                                     model=GenerativeModel("gemini-1.5-flash-002"),
    #                                     stream=self.stream,
    #                                     verbose=self.verbose,
    #                                     )

    #     self.direct_request_LLM_response = response

    #     self.direct_request_code = helper.extract_code(response=response)

    #     if review:
    #         self.ask_LLM_to_review_direct_code()

    #     return self.direct_request_LLM_response

    def execute_complete_program(self, code: str, try_cnt: int = 10) -> str:

        count = 0
        while count < try_cnt:
            print(f"\n\n-------------- Running code (trial # {count + 1}/{try_cnt}) --------------\n\n")
            try:
                count += 1
                compiled_code = compile(code, 'Complete program', 'exec')
                exec(compiled_code, globals())  # #pass only globals() not locals()
                #!!!!    all variables in code will become global variables! May cause huge issues!     !!!!
                print("\n\n--------------- Done ---------------\n\n")
                return code

            # except SyntaxError as err:
            #     error_class = err.__class__.__name__
            #     detail = err.args[0]
            #     line_number = err.lineno
            #
            except Exception as err:

                # cl, exc, tb = sys.exc_info()

                # print("An error occurred: ", traceback.extract_tb(tb))

                if count == try_cnt:
                    print(f"Failed to execute and debug the code within {try_cnt} times.")
                    return code

                debug_prompt = self.get_debug_prompt(exception=err, code=code)
                print("Sending error information to LLM for debugging...")
                # print("Prompt:\n", debug_prompt)
                response = helper.get_LLM_reply(prompt=debug_prompt,
                                                system_role=constants.debug_role,
                                                
                                                verbose=True,
                                                stream=True,
                                                retry_cnt=10,
                                                )
                code = helper.extract_code(response)

        return code


    def get_debug_prompt(self, exception, code):
        etype, exc, tb = sys.exc_info()
        exttb = traceback.extract_tb(tb)  # Do not quite understand this part.
        # https://stackoverflow.com/questions/39625465/how-do-i-retain-source-lines-in-tracebacks-when-running-dynamically-compiled-cod/39626362#39626362

        ## Fill the missing data:
        exttb2 = [(fn, lnnr, funcname,
                   (code.splitlines()[lnnr - 1] if fn == 'Complete program'
                    else line))
                  for fn, lnnr, funcname, line in exttb]

        # Print:
        error_info_str = 'Traceback (most recent call last):\n'
        for line in traceback.format_list(exttb2[1:]):
            error_info_str += line
        for line in traceback.format_exception_only(etype, exc):
            error_info_str += line

        print(f"Error_info_str: \n{error_info_str}")

        # print(f"traceback.format_exc():\n{traceback.format_exc()}")

        debug_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(constants.debug_requirement)])

        debug_prompt = f"Your role: {constants.debug_role} \n" + \
                          f"Your task: correct the code of a program according to the error information, then return the corrected and completed program. \n\n" + \
                          f"Requirement: \n {debug_requirement_str} \n\n" + \
                          f"The given code is used for this task: {self.task} \n\n" + \
                          f"The data location associated with the given code: \n {self.data_locations_str} \n\n" + \
                          f"The error information for the code is: \n{str(error_info_str)} \n\n" + \
                          f"The code is: \n{code}"

        return debug_prompt

    def ask_LLM_to_review_operation_code(self, operation):
        code = operation['operation_code']
        operation_prompt = operation['operation_prompt']
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_review_requirement)])
        review_prompt = f"Your role: {constants.operation_review_role} \n" + \
                          f"Your task: {constants.operation_review_task_prefix} \n\n" + \
                          f"Requirement: \n{review_requirement_str} \n\n" + \
                          f"The code is: \n----------\n{code}\n----------\n\n" + \
                          f"The requirements for the code is: \n----------\n{operation_prompt} \n----------\n"

            # {node_name: "", function_descption: "", function_definition:"", return_line:""
        # operation_prompt:"", operation_code:""}
        print("LLM is reviewing the operation code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.operation_review_role,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=15,
                                        )
        new_code = helper.extract_code(response)
        reply_content = helper.extract_content_from_LLM_reply(response)
        if (reply_content == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code
        operation['code'] = new_code

        return operation

    def ask_LLM_to_review_assembly_code(self):
        code = self.code_for_assembly
        assembly_prompt = self.assembly_prompt
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.assembly_review_requirement)])
        review_prompt = f"Your role: {constants.assembly_review_role} \n" + \
                          f"Your task: {constants.assembly_review_task_prefix} \n\n" + \
                          f"Requirement: \n{review_requirement_str} \n\n" + \
                          f"The code is: \n----------\n{code} \n----------\n\n" + \
                          f"The requirements for the code is: \n----------\n{assembly_prompt} \n----------\n\n"

        print("LLM is reviewing the assembly code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.assembly_review_role,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=15,
                                        )
        new_code = helper.extract_code(response)
        if (new_code == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code

        self.code_for_assembly = new_code

    def ask_LLM_to_review_direct_code(self):
        code = self.direct_request_code
        direct_prompt = self.direct_request_prompt
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.direct_review_requirement)])
        review_prompt = f"Your role: {constants.direct_review_role} \n" + \
                          f"Your task: {constants.direct_review_task_prefix} \n\n" + \
                          f"Requirement: \n{review_requirement_str} \n\n" + \
                          f"The code is: \n----------\n{code} \n----------\n\n" + \
                          f"The requirements for the code is: \n----------\n{direct_prompt} \n----------\n\n"

        print("LLM is reviewing the direct request code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        # response = helper.get_LLM_reply(prompt=review_prompt,
        #                                 system_role=constants.direct_review_role,
        #                                 model=GenerativeModel("gemini-2.0-flash-001"),
        #                                 verbose=True,
        #                                 stream=True,
        #                                 retry_cnt=5,
        #                                 )
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.direct_review_role,
                                        model_name="gemini-2.0-flash-001",  # Pass the model name as a string
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=15,
                                        )
        new_code = helper.extract_code(response)
        if (new_code == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code

        self.direct_request_code = new_code



        
    def ask_LLM_to_sample_data(self, operation_code):


        sampling_data_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.sampling_data_requirement)])
        sampling_data_review_prompt = f"Your role: {constants.sampling_data_role} \n" + \
                          f"Your task: {constants.sampling_task_prefix} \n\n" + \
                          f"Requirement: \n{sampling_data_requirement_str} \n\n" + \
                          f"The function code is: \n----------\n{code} \n----------\n\n" #+ \
                          # f"The requirements for the code is: \n----------\n{sampling_data_requirement_str} \n----------\n\n"

        print("LLM is reviewing the direct request code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=sampling_data_review_prompt,
                                        system_role=constants.sampling_data_role,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=15,
                                        )
        code = helper.extract_code(response)
        return code
        # if (new_code == "PASS") or (new_code == ""):  # if no modification.
        #     print("Code review passed, no revision.\n\n")
        #     new_code = code

        # self.direct_request_code = new_code


