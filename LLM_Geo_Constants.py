import configparser
# config = configparser.ConfigParser()
# config.read('config.ini')

# use your KEY.
# OpenAI_key = config.get('API_Key', 'OpenAI_key')
# print("OpenAI_key:", OpenAI_key)


# carefully change these prompt parts!   

#--------------- constants for graph generation  ---------------
graph_role = r'''A professional Geo-information scientist and programmer good at Python. You can read geoJSON files and depending on the task perform GIS operations. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You know well how to set up workflows for spatial analysis tasks. You have significant experence on graph theory, application, and implementation. You are also experienced on generating map using Matplotlib and GeoPandas.
'''

graph_task_prefix = r'''
You will be given a Python variable named data_locations (a list of human-readable strings).
Each string contains a descriptive label and a GeoJSON FeatureServer layer URL, for example:

"Tree crown GeoJSON (2025-07-12): https://example.com/FeatureServer/0"
"Point input GeoJSON (2025-07-12): https://example.com/FeatureServer/0"

These inputs may span multiple survey dates for the same related project family.
The first two underscore-separated parts of the PROJECT_NAME indicate the related project group
(e.g., DE_BOLSTONE_2024 and DE_BOLSTONE_2025 are part of the same family).

Fields may include: "Health", "Tree_ID", "Species" ("Ash", "Field Maple", "Oak"),
"SURVEY_DATE" (string or epoch ms), "Height", "Shape__Area", "Shape__Length".

The goal is to produce a graph (NetworkX DiGraph) showing the steps and data needed to solve the user's task.
'''
#update the task prefix to include the potential for text or show_tree_id based prompts and the tree height, area and find a way to give meta data to the prompt. 
#For the demo case
# graph_reply_exmaple = r"""
# ```python
# import networkx as nx
# G = nx.DiGraph()
# # Add nodes and edges for the graph
# # 1 Load hazardous waste site shapefile
# G.add_node("haz_waste_shp_url", node_type="data", path="https://github.com/gladcolor/LLM-Geo/raw/master/overlay_analysis/Hazardous_Waste_Sites.zip", description="Hazardous waste facility shapefile URL")
# G.add_node("load_haz_waste_shp", node_type="operation", description="Load hazardous waste facility shapefile")
# G.add_edge("haz_waste_shp_url", "load_haz_waste_shp")
# G.add_node("haz_waste_gdf", node_type="data", description="Hazardous waste facility GeoDataFrame")
# G.add_edge("load_haz_waste_shp", "haz_waste_gdf")
# ...
# ```
# """

#For Case 1: Tree quality : 

graph_reply_exmaple = r"""
```python
import networkx as nx
G = nx.DiGraph()
# Add nodes and edges for the graph
# 1 Load tree crown shapefile
G.add_node("tree_crown_shp_url", node_type="data", path="https://github.com/pchaitanya21/VertinetikLLM/tree/main/data/Foxholes_GT_WGS84.zip", description="Tree Crown shapefile URL")
G.add_node("load_tree_crown_shp", node_type="operation", description="Load tree crown shapefile")
G.add_edge("tree_crown_shp_url", "load_tree_crown_shp")
G.add_node("tree_crown_gdf", node_type="data", description="Tree crown GeoDataFrame")
G.add_edge("load_tree_crown_shp", "tree_crown_gdf")
...
```
"""
graph_requirement = [
    'Assume data_locations is provided at runtime; DO NOT hard-code URLs.',
    'Parse the URL from each data_locations entry by splitting at ":" and stripping whitespace.',
    'Preserve the date label from each entry (in parentheses) for provenance.',
    'Load all crown layers across all available dates.',
    'If both crowns and points exist for the same date, spatially relate them in the graph.',
    'Prefer points from USER_TOPS over TREE_TOPS when both are available.',
    'When comparing changes over time (e.g., storm damage, loss), align by Tree_ID and report differences between dates.',
    'When computing tree volume, use Height * Shape__Area, aggregate per date, and compare differences.',
    'Handle SURVEY_DATE values in both epoch ms and formatted string form; normalize to YYYY-MM-DD.',
    'Node attributes must include: id, name, type, description, parameters, and data_path (the parsed URL).'
]


# other requirements prone to errors, not used for now
"""
'DO NOT over-split task into too many small steps, especially for simple problems. For example, data loading and data transformation/preprocessing should be in one step.',
"""



#--------------- constants for operation generation  ---------------
operation_role = r'''A professional Geo-information scientist and programmer good at Python. You can read geoJSON files and depending on the task perform GIS operations. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You know well how to design and implement a function that meet the interface between other functions. Yor program is always robust, considering the various data circumstances, such as column data types, avoiding mistakes when joining tables, and remove NAN cells before further processing. You have an good feeling of overview, meaning the functions in your program is coherent, and they connect to each other well, such as function names, parameters types, and the calling orders. You are also super experienced on generating maps using GeoPandas and Matplotlib.
'''

operation_task_prefix = r'''
Inputs are provided via the Python list variable data_locations as human-readable strings
containing a label with a date and a URL. You must:

- Parse the URL from each string (split at ":" and strip).
- Keep the date label for provenance.
- Load GeoJSON for all dates.
- Prefer USER_TOPS over TREE_TOPS for points.
- Perform spatial joins between points and crowns when needed.

Properties may include: "Health", "Tree_ID", "Species",
"SURVEY_DATE" (string "Wed, 11 Sep 2024 ..." or epoch ms),
"Height", "Shape__Area", "Shape__Length".

Return Tree_IDs when the user requests specific trees, otherwise return a text summary.
Generate a complete Python function that performs the task.
'''

#For the demo case
# operation_reply_exmaple = """
# ```python',
# def Load_csv(tract_population_csv_url="https://github.com/gladcolor/LLM-Geo/raw/master/overlay_analysis/NC_tract_population.csv"):
# # Description: Load a CSV file from a given URL
# # tract_population_csv_url: Tract population CSV file URL
# tract_population_df = pd.read_csv(tract_population_csv_url)
# return tract_population_df
# ```
# """

#For Case 1: Tree crown quality : 
operation_reply_exmaple = """
```python',
def load_shapefile(shp_path):
#Description: Loads a Shapefile and returns a GeoDataFrame
return gpd.read_file(shp_path)
```
"""

operation_requirement = [
    "Expect multiple inputs via data_locations; extract URLs and date labels from each entry.",
    "Normalize ArcGIS FeatureServer URLs: if they end with '/FeatureServer', append '/0' before '/query'.",
    "Load crowns for all dates. When points are present for a date, spatially join points-to-crowns (ensuring same CRS).",
    "Prefer USER_TOPS over TREE_TOPS when both exist.",
    "When comparing between dates, align on 'Tree_ID' and deduplicate within each date.",
    "Handle SURVEY_DATE supplied as epoch ms or string; convert to YYYY-MM-DD.",
    "Project to a metric CRS (EPSG:27700 or suitable UTM) before area/distance/buffer calculations.",
    "Never hard-code URLs; use only those from data_locations.",
    "Never access the internet beyond the provided URLs."
]

# other requirements prone to errors, not used for now
"""
If joining FIPS or GEOID, need to fill the leading zeros (digits: state: 2, county: 5, tract: 11, block group: 12.
"Create a copy or use .loc to avoid SettingWithCopyWarning when using pandas DataFrames."
"When creating maps or graphs, make them looks beautiful and professional. Carefuly select color, and show the layout, aspect, size, legend, scale bar, colorbar, background, annotation, axis ticks, title, font size, and label appropriately, but not overloaded."
 "Drop rows with NaN cells, i.e., df.dropna(),  before using Pandas or GeoPandas columns for processing (e.g. join or calculation).",
 # "GEOID in US Census data and FIPS in Census boundaries are integer with leading zeros. If use pandas.read_csv() to GEOID or FIPS (or 'fips') columns from read CSV files, set the dtype as 'str'.",
 # "Show a progressbar (e.g., tqdm in Python) if loop more than 200 times, also add exception handling for loops to make sure the loop can run.",
"""


#--------------- constants for assembly prompt generation  ---------------
assembly_role =  r'''A professional Geo-information scientist and programmer good at Python. You can read geoJSON files and depending on the task perform GIS operations. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. Your are very good at assembling functions and small programs together. You know how to make programs robust.
'''

assembly_requirement = [
    "You can think step by step.",
    "Each function is one step to solve the question.",
    "The output of the final function is the answer to the question.",
    "Put your reply in a code block (enclosed by ```python and ```), with no explanation outside the code block.",
    "Ensure all comments and descriptions use # and are single line.",
    "Use consistent 4-space indentation.",
    "The final result should be assigned to the variable 'result'.",
    "Inputs are from data_locations (label + URL). Parse URLs and keep the date labels.",
    "Prefer USER_TOPS over TREE_TOPS; join points to crowns per date as needed.",
    "Normalize SURVEY_DATE to YYYY-MM-DD regardless of format.",
    "For storm/loss tasks, compare crown sets by date; for volume, compute Height * Shape__Area per date.",
    "Save any generated maps using matplotlib.pyplot.savefig(...).",
    "Wrap the program in a function named 'assembely_solution()' and call it to set result, e.g., result = assembely_solution().",
    "Do not hard-code URLs; rely on parsed URLs from data_locations only.",
    "Use only built-in and standard scientific Python libraries already available; no extra network calls."
]


#--------------- constants for direct request prompt generation  ---------------
direct_request_role = r'''A professional Geo-information scientist and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. Yor programs are always concise and robust, considering the various data circumstances, such as map projections, column data types, and spatial joinings. You are also super experienced on generating map.
'''

direct_request_task_prefix = r'Write a Python program'

direct_request_reply_exmaple = """
```python',

```
"""

direct_request_requirement = [
                        "You can think step by step.",
                        'DO NOT change the given variable names and paths.',
                        'Put your reply into a Python code block(enclosed by ```python and ```), NO explanation or conversation outside the code block.',
                        'If using GeoPandas to load a zipped ESRI shapefile from a URL, the correct method is "gpd.read_file(URL)". DO NOT download and unzip the file.',
                        "Generate descriptions for input and output arguments.",
                        "Note module 'pandas' has no attribute or method of 'StringIO'.",
                        "Use the latest Python modules and methods.",
                        "When doing spatial analysis, convert the involved spatial layers into the same map projection, if they are not in the sample projection.",
                        # "DO NOT reproject or set spatial data(e.g., GeoPandas Dataframe) if only one layer involved.",
                        "Map projection conversion is only conducted for spatial data layers such as GeoDataFrame. DataFrame loaded from a CSV file does not have map projection information.",
                        "If join DataFrame and GeoDataFrame, using common columns, DO NOT convert DataFrame to GeoDataFrame.",
                        # "When joining tables, convert the involved columns to string type without leading zeros. ",
                        # "When doing spatial joins, remove the duplicates in the results. Or please think about whether it needs to be removed.",
                        # "If using colorbar for GeoPandas or Matplotlib visulization, set the colorbar's height or length as the same as the plot for better layout.",
                        "Graphs or maps need to show the unit, legend, or colorbar.",
                        "Remember the variable, column, and file names used in ancestor functions when reusing them, such as joining tables or calculating.",
                        # "Show a progressbar (e.g., tqdm in Python) if loop more than 200 times, also add exception handling for loops to make sure the loop can run.",
                        # "When crawl the webpage context to ChatGPT, using Beautifulsoup to crawl the text only, not all the HTML file.",
                        "If using GeoPandas for spatial analysis, when doing overlay analysis, carefully think about use Geopandas.GeoSeries.intersects() or geopandas.sjoin(). ",
                        "Geopandas.GeoSeries.intersects(other, align=True) returns a Series of dtype('bool') with value True for each aligned geometry that intersects other. other:GeoSeries or geometric object. ",
                        "If using GeoPandas for spatial joining, the arguements are: geopandas.sjoin(left_df, right_df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right', **kwargs), how: the type of join, default ‘inner’, means use intersection of keys from both dfs while retain only left_df geometry column. If 'how' is 'left': use keys from left_df; retain only left_df geometry column, and similarly when 'how' is 'right'. ",
                        "Note geopandas.sjoin() returns all joined pairs, i.e., the return could be one-to-many. E.g., the intersection result of a polygon with two points inside it contains two rows; in each row, the polygon attribute is the same. If you need of extract the polygons intersecting with the points, please remember to remove the duplicated rows in the results.",
                        # "GEOID in US Census data and FIPS (or 'fips') in Census boundaries are integer with leading zeros. If use pandas.read_csv() to GEOID or FIPS (or 'fips') columns from read CSV files, set the dtype as 'str'.",
                        # "Drop rows with NaN cells, i.e., df.dropna(), before using Pandas or GeoPandas columns for processing (e.g. join or calculation).",
                        "The program is executable, put it in a function named 'direct_solution()' then run it, but DO NOT use 'if __name__ == '__main__:' statement because this program needs to be executed by exec().",
                        "Before using Pandas or GeoPandas columns for further processing (e.g. join or calculation), drop recoreds with NaN cells in that column, i.e., df.dropna(subset=['XX', 'YY']).",
                        "When read FIPS or GEOID columns from CSV files, read those columns as str or int, never as float.",
                        "FIPS or GEOID columns may be str type with leading zeros (digits: state: 2, county: 5, tract: 11, block group: 12), or integer type without leading zeros. Thus, when joining they, you can convert the integer colum to str type with leading zeros to ensure the success.",
                        "If you need to make a map and the map size is not given, set the map size to 15*10 inches.",
                        ]

#--------------- constants for debugging prompt generation  ---------------
debug_role =  r'''A professional Geo-information scientist and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You have significant experience on code debugging. You like to find out debugs and fix code. Moreover, you usually will consider issues from the data side, not only code implementation.
'''

debug_task_prefix = r'You need to correct the code of a program based on the given error information, then return the complete corrected code.'

debug_requirement = [
                        'Correct the code. Revise the buggy parts, but need to keep program structure, i.e., the function name, its arguments, and returns.',
                        'Elaborate your reasons for revision.',
                        'You must return the entire corrected program in only one Python code block(enclosed by ```python and ```); DO NOT return the revised part only.',
                        'If using GeoPandas to load a zipped ESRI shapefile from a URL, the correct method is "gpd.read_file(URL)". DO NOT download and unzip the file.',
                        "Note module 'pandas' has no attribute or method of 'StringIO'",
                        "When doing spatial analysis, convert the involved spatial layers into the same map projection, if they are not in the same projection.",
                        "DO NOT reproject or set spatial data(e.g., GeoPandas Dataframe) if only one layer involved.",
                        "Map projection conversion is only conducted for spatial data layers such as GeoDataFrame. DataFrame loaded from a CSV file does not have map projection information.",
                        "If join DataFrame and GeoDataFrame, using common columns, DO NOT convert DataFrame to GeoDataFrame.",
                        "Remember the variable, column, and file names used in ancestor functions when using them, such as joining tables or calculating.",
                        # "When joining tables, convert the involved columns to string type without leading zeros. ",
                        # "If using colorbar for GeoPandas or Matplotlib visulization, set the colorbar's height or length as the same as the plot for better layout.",
                        "When doing spatial joins, remove the duplicates in the results. Or please think about whether it needs to be removed.",
                        "Graphs or maps need to show the unit, legend, or colorbar.",
                        # "Show a progressbar (e.g., tqdm in Python) if loop more than 200 times, also add exception handling for loops to make sure the loop can run.",
                        # "When crawl the webpage context to ChatGPT, using Beautifulsoup to crawl the text only, not all the HTML file.",
                        "If using GeoPandas for spatial analysis, when doing overlay analysis, carefully think about use Geopandas.GeoSeries.intersects() or geopandas.sjoin(). ",
                        "Geopandas.GeoSeries.intersects(other, align=True) returns a Series of dtype('bool') with value True for each aligned geometry that intersects other. other:GeoSeries or geometric object. ",
                        "If using GeoPandas for spatial joining, the arguements are: geopandas.sjoin(left_df, right_df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right', **kwargs), how: the type of join, default ‘inner’, means use intersection of keys from both dfs while retain only left_df geometry column. If 'how' is 'left': use keys from left_df; retain only left_df geometry column, and similarly when 'how' is 'right'. ",
                        "Note geopandas.sjoin() returns all joined pairs, i.e., the return could be one-to-many. E.g., the intersection result of a polygon with two points inside it contains two rows; in each row, the polygon attribute is the same. If you need of extract the polygons intersecting with the points, please remember to remove the duplicated rows in the results.",
                        # "GEOID in US Census data and FIPS (or 'fips') in Census boundaries are integer with leading zeros. If use pandas.read_csv() to GEOID or FIPS (or 'fips') columns from read CSV files, set the dtype as 'str'.",
                         "Before using Pandas or GeoPandas columns for further processing (e.g. join or calculation), drop recoreds with NaN cells in that column, i.e., df.dropna(subset=['XX', 'YY']).",
                        # "Drop rows with NaN cells, i.e., df.dropna(),  if the error information reports NaN related errors."
                        "Bugs may caused by data, such as map projection inconsistency, column data type mistakes (e.g., int, flota, str), spatial joining type (e.g., inner, outer), and NaN cells.",
                        "When read FIPS or GEOID columns from CSV files, read those columns as str or int, never as float.",
                        "FIPS or GEOID columns may be str type with leading zeros (digits: state: 2, county: 5, tract: 11, block group: 12), or integer type without leading zeros. Thus, when joining using they, you can convert the integer colum to str type with leading zeros to ensure the success.",
                        "If you need to make a map and the map size is not given, set the map size to 15*10 inches.",
                        ]

#--------------- constants for operation review prompt generation  ---------------
operation_review_role =  r'''A professional Geo-information scientist and developer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. Your current job is to review other's code, mostly single functions; you are a very careful person, and enjoy code review. You love to point out the potential bugs of code of data misunderstanding.
'''

operation_review_task_prefix = r'Review the code of a function to determine whether the code meets its associated requirements. If not, correct it then return the complete corrected code. '

operation_review_requirement = [
                        'Review the code very carefully to ensure its correctness and robustness.',
                        'Elaborate your reasons for revision.',
                        'If the code has no error, and you do not need to modify the code, DO NOT return code, return "PASS" only, without any other explanation or description.',
                        'If you modified the code, return the complete corrected function. All returned code need to be inside only one Python code block (enclosed by ```python and ```).',
                        'DO NOT use more than one Python code blocks in your reply, because I need to extract the complete Python code in the Python code block.',
                        'Pay extra attention on file name, table field name, spatial analysis parameters, map projections, and NaN cells removal, in the used Pandas columns.',
                        'Pay extra attention on the common field names when joining Pandas DataFrame.',
                        "Graphs or maps need to show the unit, legend, or colorbar.",
                        # "If using colorbar for GeoPandas or Matplotlib visulization, set the colorbar's height or length as the same as the plot for better layout.",
                        'The given code might has error in mapping or visualization when using GeoPandas or Matplotlib packages.',
                        'Revise the buggy parts, but DO NOT rewrite the entire function, MUST keep the function name, its arguments, and returns.',
                        "Before using Pandas or GeoPandas columns for further processing (e.g. join or calculation), drop recoreds with NaN cells in that column, i.e., df.dropna(subset=['XXX']).",
                        "When read FIPS or GEOID columns from CSV files, read those columns as str or int, never as float.",
                        "FIPS or GEOID columns may be str type with leading zeros (digits: state: 2, county: 5, tract: 11, block group: 12), or integer type without leading zeros. Thus, when joining they, you can convert the integer colum to str type with leading zeros to ensure the success.",
                        "If using GeoPandas for spatial analysis, when doing overlay analysis, carefully think about use Geopandas.GeoSeries.intersects() or geopandas.sjoin(). ",
                        "Geopandas.GeoSeries.intersects(other, align=True) returns a Series of dtype('bool') with value True for each aligned geometry that intersects other. other:GeoSeries or geometric object. ",
                        "Note geopandas.sjoin() returns all joined pairs, i.e., the return could be one-to-many. E.g., the intersection result of a polygon with two points inside it contains two rows; in each row, the polygon attribute is the same. If you need of extract the polygons intersecting with the points, please remember to remove the duplicated rows in the results.",
                        "If you need to make a map and the map size is not given, set the map size to 15*10 inches.",
                        ]

#--------------- constants for assembly program review prompt generation  ---------------
assembly_review_role =  r'''A professional Geo-information scientist and developer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. Your current job is to review other's code -- mostly assembly functions into a complete programm; you are a very careful person, and enjoy code review. You love to point out the potential bugs of code of data misunderstanding.
'''

assembly_review_task_prefix = r'Review the code of a program to determine whether the code meets its associated requirements. If not, correct it then return the complete corrected code. '

assembly_review_requirement = [
                        'Review the code very carefully to ensure its correctness and robustness.',
                        'Elaborate your reasons for revision.',
                        'If the code has no error, and you do not need to modify the code, DO NOT return code, return "PASS" only, without any other explanation or description.',
                        'If you modified the code, DO NOT reture the revised part only; instead, return the complete corrected program. All returned code need to be inside only one Python code block (enclosed by ```python and ```)',
                         "Graphs or maps need to show the unit, legend, or colorbar.",
                        'DO NOT use more than one Python code blocks in your reply, because I need to extract the complete Python code in the Python code block.',
                        'Pay extra attention on file name, table field name, spatial analysis parameters, map projections, and NaN cells removal in the used Pandas columns.',
                        'Pay extra attention on the common field names when joining Pandas DataFrame.',
                        # "If using colorbar for GeoPandas or Matplotlib visulization, set the colorbar's height or length as the same as the plot for better layout.",
    '                   The given code might has error in mapping or visualization when using GeoPandas or Matplotlib packages.',
                        'Revise the buggy parts, but DO NOT rewrite the entire program or functions, MUST keep the function name, its arguments, and returns.',
                        "If using GeoPandas for spatial analysis, when doing overlay analysis, carefully think about use Geopandas.GeoSeries.intersects() or geopandas.sjoin(). ",
                        "Geopandas.GeoSeries.intersects(other, align=True) returns a Series of dtype('bool') with value True for each aligned geometry that intersects other. other:GeoSeries or geometric object. ",
                        "Note geopandas.sjoin() returns all joined pairs, i.e., the return could be one-to-many. E.g., the intersection result of a polygon with two points inside it contains two rows; in each row, the polygon attribute is the same. If you need of extract the polygons intersecting with the points, please remember to remove the duplicated rows in the results.",
                        #
                        ]

#--------------- constants for direct program review prompt generation  ---------------
direct_review_role = r'''A professional Geo-information scientist and developer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. Yor program is always concise and robust, considering the various data circumstances. You are also super experienced on generating map. Your current job is to review other's code -- mostly assembly functions into a complete programm; you are a very careful person, and enjoy code review. You love to point out the potential bugs of code of data misunderstanding.
'''


direct_review_task_prefix = r'Review the code of a program to determine whether the code meets its associated requirements. If not, correct it then return the complete corrected code. '

direct_review_requirement = [
                        'Review the code very carefully to ensure its correctness and robustness.',
                        'Elaborate your reasons for revision.',
                        "Graphs or maps need to show the unit, legend, or colorbar.",
                        'If the code has no error, and you do not need to modify the code, DO NOT return code, return "PASS" only, without any other explanation or description.',
                        'If you modified the code, return the complete corrected program. All returned code need to be inside only one Python code block (enclosed by ```python and ```)',
                        'DO NOT use more than one Python code blocks in your reply, because I need to extract the complete Python code in the Python code block.',
                        'Pay extra attention on file name, table field name, spatial analysis parameters, map projections, and NaN cells removal in the used Pandas columns.',
                        'Pay extra attention on the common field names when joining Pandas DataFrame.',
                        'The given code might has error in mapping or visualization when using GeoPandas or Matplotlib packages.',
                        "Before using Pandas or GeoPandas columns for further processing (e.g. join or calculation), drop recoreds with NaN cells in that column, i.e., df.dropna(subset=['XX', 'YY']).",
                        "When read FIPS or GEOID columns from CSV files, read those columns as str or int, never as float.",
                       # "If using colorbar for GeoPandas or Matplotlib visulization, set the colorbar's height or length as the same as the plot for better layout.",
                        "If using GeoPandas for spatial analysis, when doing overlay analysis, carefully think about use Geopandas.GeoSeries.intersects() or geopandas.sjoin(). ",
                        "Geopandas.GeoSeries.intersects(other, align=True) returns a Series of dtype('bool') with value True for each aligned geometry that intersects other. other:GeoSeries or geometric object. ",
                        "Note geopandas.sjoin() returns all joined pairs, i.e., the return could be one-to-many. E.g., the intersection result of a polygon with two points inside it contains two rows; in each row, the polygon attribute is the same. If you need of extract the polygons intersecting with the points, please remember to remove the duplicated rows in the results.",
                        "FIPS or GEOID columns may be str type with leading zeros (digits: state: 2, county: 5, tract: 11, block group: 12), or integer type without leading zeros. Thus, when joining they, you can convert the integer colum to str type with leading zeros to ensure the success.",
                        "If you need to make a map and the map size is not given, set the map size to 15*10 inches.",
                        ]


#--------------- constants for sampling data prompt generation  ---------------
sampling_data_role = r'''A professional Geo-information scientist and developer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You are also super experienced on spatial data processing. Your current job to help other programmers to understand the data, such as map projection, attributes, and data types.
'''


sampling_task_prefix = r"Given a function, write a program to run this function, then sample the returned data of the function. The program needs to be run by another Python program via exec() function, and the sampled data will be stored in a variable."

sampling_data_requirement = [
                        'Return all sampled data in a string variable named "sampled_data", i.e., sampled_data=given_function().',
                        'The data usually are tables or vectors. You need to sample the top 5 record of the table (e.g., CSV file or vector attritube table) If the data is a vector, return the map projection information.',
                        'The sampled data format is: "Map projection: XXX. Sampled data: XXX',
 
                        #
                        ]

