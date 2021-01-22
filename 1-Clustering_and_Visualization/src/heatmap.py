import folium
from folium.plugins import HeatMap
import pandas as pd


def generateBaseMap(default_location=[40.693943, -73.985880]):  # coordinates of NYC
    base_map = folium.Map( location=default_location )
    return base_map


df = pd.read_csv("nyc_listings.csv")
df.dropna()
base_map = generateBaseMap()
HeatMap(data=df[['latitude', 'longitude', 'price']].groupby(['latitude', 'longitude']).mean().
        reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)
base_map.save('index.html')
