import networkx as nx


G = nx.DiGraph()


# Load GeoJSON data
G.add_node("geojson_url", node_type="data", data_path="https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/Hicks_Lodge_Trial_pred.geojson", description="URL to the GeoJSON file containing tree crown data.")
G.add_node("load_geojson", node_type="operation", description="Load GeoJSON data into a GeoDataFrame.")
G.add_edge("geojson_url", "load_geojson")
G.add_node("tree_gdf", node_type="data", description="GeoDataFrame containing tree crown data.")
G.add_edge("load_geojson", "tree_gdf")


 # Filter Ash species
G.add_node("filter_ash", node_type="operation", description="Filter GeoDataFrame to select only Ash trees ('Predicted Tree Species':'Ash').")
G.add_edge("tree_gdf", "filter_ash")
G.add_node("ash_trees_gdf", node_type="data", description="GeoDataFrame containing only Ash trees.")
G.add_edge("filter_ash", "ash_trees_gdf")


 # Create map
G.add_node("create_map", node_type="operation", description="Create a map using Matplotlib and GeoPandas, highlighting Ash trees in red.")
G.add_edge("tree_gdf", "create_map")
G.add_edge("ash_trees_gdf", "create_map")
G.add_node("map_figure", node_type="data", data_path="", description="Matplotlib figure object representing the map.")
G.add_edge("create_map", "map_figure")


 # Set map size (optional, might be handled within 'create_map' operation)
G.add_node("set_map_size", node_type="operation", description="Set the figure size of the map to 15x10 inches.")
G.add_edge("map_figure", "set_map_size")
G.add_node("sized_map_figure", node_type="data", description="Matplotlib figure object with the specified size.")
G.add_edge("set_map_size", "sized_map_figure")


 #Final output
G.add_node("final_map", node_type="data", data_path="", description="The final map showing all tree crowns, with ash trees highlighted in red.")
G.add_edge("sized_map_figure", "final_map")


nx.write_graphml(G, "Tree_crown_quality\\Tree_crown_quality.graphml")