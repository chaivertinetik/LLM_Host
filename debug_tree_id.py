import networkx as nx

G = nx.DiGraph()

# Node 1: GeoJSON data
G.add_node("geojson_data", node_type="data", data_path="https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/Hicks_Lodge_Trial_pred.geojson", description="Tree crown GeoJSON data")

# Node 2: Load GeoJSON operation
G.add_node("load_geojson", node_type="operation", description="Load GeoJSON file into GeoDataFrame")
G.add_edge("geojson_data", "load_geojson")

# Node 3: GeoDataFrame
G.add_node("tree_gdf", node_type="data", description="GeoDataFrame containing tree data")
G.add_edge("load_geojson", "tree_gdf")

# Node 4: Filter Ash species operation
G.add_node("filter_ash", node_type="operation", description="Filter GeoDataFrame to include only Ash species ('Predicted Tree Species':'Ash')")
G.add_edge("tree_gdf", "filter_ash")

# Node 5: Ash tree IDs
G.add_node("ash_tree_ids", node_type="data", description="List of 'Tree ID' for Ash trees")
G.add_edge("filter_ash", "ash_tree_ids")


#Print graph information for verification (optional)
#print(G.nodes(data=True))
#print(G.edges())

#This part is not required by the prompt, but included for completeness.
# nx.write_graphml(G, "C:\\Users\\chait\\Projects\\LLM_Geo_GCP\\Hosting env\\Tree_crown_quality\\Tree_crown_quality.graphml")
