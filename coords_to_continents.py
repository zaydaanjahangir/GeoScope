import time
import requests
import pandas as pd
import pycountry
import pycountry_convert as pc
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

MAPBOX_TOKEN = os.getenv("MAPBOX_PUBLIC_TOKEN")
print(MAPBOX_TOKEN)

def latlong_to_continent_mapbox(lat, lon):
    url = "https://api.mapbox.com/search/geocode/v6/reverse"
    params = {
        "latitude": lat,
        "longitude": lon,
        "types": "country",  
        "access_token": MAPBOX_TOKEN
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        features = data.get("features", [])
        if not features:
            return None

        feature = features[0]
        country_name = feature.get("text", None)
        if not country_name:
            return None
        
        # Use pycountry to look up the country and extract the alpha-2 code.
        try:
            country = pycountry.countries.lookup(country_name)
            country_code = country.alpha_2

            # Convert the country code to a continent code and then to a continent name.
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            return continent_name
        except Exception as e:
            print(f"Country lookup error for '{country_name}': {e}")
            return None
    except Exception as e:
        print(f"Error for ({lat}, {lon}): {e}")
        return None

df = pd.read_csv("Streetview_Image_Dataset/coordinates.csv")


continents = []


for index, row in tqdm(df.iterrows(), total=len(df)):
    lat, lon = row["latitude"], row["longitude"]
    continent = latlong_to_continent_mapbox(lat, lon)
    continents.append(continent)
    
    # Sleep for 0.05 seconds to be cautious.
    time.sleep(0.05)
    

    if (index + 1) % 1000 == 0:
        df.loc[:index, "continent"] = continents
        intermediate_filename = f"coordinates_with_continents_{index+1}.csv"
        df.to_csv(intermediate_filename, index=False)
        print(f"Intermediate save: {intermediate_filename}")


df["continent"] = continents
df.to_csv("coordinates_with_continents_mapbox.csv", index=False)
print("Finished processing. Output saved as 'coordinates_with_continents_mapbox.csv'.")
