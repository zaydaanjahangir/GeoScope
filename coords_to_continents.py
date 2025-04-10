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


def extract_country_from_feature(feature):
    # Try getting the country name from the top-level 'text' field
    country_name = feature.get("text", None)
    if country_name:
        return country_name
    
    # If not available, look into the 'context' array for a country
    context = feature.get("context", [])
    for ctx in context:
        if ctx.get("id", "").startswith("country"):

            # Country name can be under "text" or "name"
            return ctx.get("text", ctx.get("name", None))
    return None

def latlong_to_continent_mapbox(lat, lon):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "types": "country",  # Restrict results to countries
        "limit": 1
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        features = data.get("features", [])
        if not features:
            return None
        
        # Extract country name from the first feature.
        feature = features[0]
        country_name = extract_country_from_feature(feature)
        if not country_name:
            return None
        
        # Use pycountry to look up the country and extract the alpha-2 code
        try:
            
            country = pycountry.countries.lookup(country_name.title())
            country_code = country.alpha_2

            # Convert the country code to a continent code and then to a continent name
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            return continent_name
        except Exception as e:
            if country_name.strip().lower() == "russia":
                country_name = "Russia"
                return "Russia"  
            else:
                print(f"Country lookup error for '{country_name}': {e}")
            return None
    except Exception as e:
        print(f"Error for ({lat}, {lon}): {e}")
        return None

df = pd.read_csv("Streetview_Image_Dataset/coordinates.csv")
continents = []

# Process each row with progress tracking.
for index, row in tqdm(df.iterrows(), total=len(df)):
    lat, lon = row["latitude"], row["longitude"]
    continent = latlong_to_continent_mapbox(lat, lon)
    continents.append(continent)
    
    # Sleep for 0.05 seconds for Mapbox's higher rate limit
    time.sleep(0.05)
    
    # Save intermediate results every 1000 iterations
    if (index + 1) % 1000 == 0:
        df.loc[:index, "continent"] = continents
        intermediate_filename = f"coordinates_with_continents_{index+1}.csv"
        df.to_csv(intermediate_filename, index=False)
        print(f"Intermediate save: {intermediate_filename}")

# Complete continent assignment for CSV
df["continent"] = continents
df.to_csv("coordinates_with_continents_mapbox.csv", index=False)
print("Finished processing. Output saved as 'coordinates_with_continents_mapbox.csv'.")
