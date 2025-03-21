import requests

url = "https://services-eu1.arcgis.com/8uHkpVrXUjYCyrO4/arcgis/rest/services/TreeCrowns_BE_Bolstone_13032025_/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson"
response = requests.get(url)

if response.status_code == 200:
    geojson_data = response.json()
    # Process the geojson_data as needed
else:
    print(f"Error fetching data: {response.status_code}")
print(geojson_data)