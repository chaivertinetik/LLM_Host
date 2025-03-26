import geopandas as gpd
import pandas as pd

def load_geojson(geojson_data='https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/TreeCrowns_Foxholes_21032025.geojson'):
    #Load GeoJSON file into GeoDataFrame
    tree_gdf = gpd.read_file(geojson_data)
    return tree_gdf

def filter_ash(tree_gdf=None):
    #Filter GeoDataFrame to include only Ash species ('Predicted Tree Species':'Ash')
    if tree_gdf is None:
        raise ValueError("Input GeoDataFrame is None.")
    tree_gdf = tree_gdf.dropna(subset=['Species'])
    ash_tree_gdf = tree_gdf[tree_gdf['Species'] == 'Ash']
    ash_tree_ids = ash_tree_gdf['Tree_ID'].tolist()
    print(f"Found {len(ash_tree_ids)} ash trees")
    return ash_tree_ids

def assembely_solution():
    try:
        print("Loading GeoJSON...")
        tree_gdf = load_geojson()
        print("Filtering for ash trees...")
        ash_tree_ids = filter_ash(tree_gdf)
        print(f"Result: {ash_tree_ids}")
        return ash_tree_ids
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

# This line actually executes the function
result = assembely_solution()
