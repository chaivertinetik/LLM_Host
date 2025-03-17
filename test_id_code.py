# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install geopandas pandas

import geopandas as gpd
import pandas as pd

def load_geojson(geojson_data='https://raw.githubusercontent.com/pchaitanya21/VertinetikLLM/main/data/Hicks_Lodge_Trial_pred.geojson'):
    #Load GeoJSON file into GeoDataFrame
    tree_gdf = gpd.read_file(geojson_data)
    return tree_gdf

def filter_ash(tree_gdf=None):
    #Filter GeoDataFrame to include only Ash species ('Predicted Tree Species':'Ash')
    if tree_gdf is None:
        raise ValueError("Input GeoDataFrame is None.")

    tree_gdf = tree_gdf.dropna(subset=['Predicted Tree Species'])
    ash_tree_gdf = tree_gdf[tree_gdf['Predicted Tree Species'] == 'Ash']
    ash_tree_ids = ash_tree_gdf['Tree ID'].tolist()
    print(ash_tree_ids)
    return ash_tree_ids

def assembely_solution():
    try:
        tree_gdf = load_geojson()
        ash_tree_ids = filter_ash(tree_gdf)
        return ash_tree_ids
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

result=assembely_solution()
print(result)