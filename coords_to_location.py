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

def extract_location_info(feature):
    location_info = {
        'city': None,
        'region': None,
        'country': None
    }
    
    # Extract country from context
    context = feature.get("context", [])
    for ctx in context:
        if ctx.get("id", "").startswith("country"):
            location_info['country'] = ctx.get("text", ctx.get("name", None))
        elif ctx.get("id", "").startswith("region"):
            location_info['region'] = ctx.get("text", ctx.get("name", None))
        elif ctx.get("id", "").startswith("place"):
            location_info['city'] = ctx.get("text", ctx.get("name", None))
    
    # If city not found in context, check the main feature
    if not location_info['city'] and feature.get("place_type", [""])[0] == "place":
        location_info['city'] = feature.get("text", None)
    
    return location_info

def latlong_to_location_info(lat, lon):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "types": "place,region,country",  # Include places (cities), regions, and countries
        "limit": 1
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        features = data.get("features", [])
        if not features:
            return None, None, None, None
        
        # Extract location information from the first feature
        feature = features[0]
        location_info = extract_location_info(feature)
        
        # Get country name and continent
        country_name = location_info['country']
        if not country_name:
            return None, None, None, None
        
        try:
            country = pycountry.countries.lookup(country_name.title())
            country_code = country.alpha_2
            
            # Convert country code to continent
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            
            return (
                location_info['city'],
                location_info['region'],
                country_name,
                continent_name
            )
        except Exception as e:
            if country_name.strip().lower() == "russia":
                return location_info['city'], location_info['region'], "Russia", "Asia"
            else:
                print(f"Country lookup error for '{country_name}': {e}")
            return None, None, None, None
    except Exception as e:
        print(f"Error for ({lat}, {lon}): {e}")
        return None, None, None, None

# Read the coordinates file
df = pd.read_csv("Streetview_Image_Dataset/coordinates.csv")

# Initialize lists to store location information
cities = []
regions = []
countries = []
continents = []

# Process each row with progress tracking
for index, row in tqdm(df.iterrows(), total=len(df)):
    lat, lon = row["latitude"], row["longitude"]
    city, region, country, continent = latlong_to_location_info(lat, lon)
    
    cities.append(city)
    regions.append(region)
    countries.append(country)
    continents.append(continent)
    
    # Sleep for 0.05 seconds for Mapbox's higher rate limit
    time.sleep(0.05)
    
    # Save intermediate results every 1000 iterations
    if (index + 1) % 1000 == 0:
        df.loc[:index, "city"] = cities
        df.loc[:index, "region"] = regions
        df.loc[:index, "country"] = countries
        df.loc[:index, "continent"] = continents
        intermediate_filename = f"coordinates_with_locations_{index+1}.csv"
        df.to_csv(intermediate_filename, index=False)
        print(f"Intermediate save: {intermediate_filename}")

# Complete location assignment for CSV
df["city"] = cities
df["region"] = regions
df["country"] = countries
df["continent"] = continents

# Save the final results
df.to_csv("coordinates_with_locations_mapbox.csv", index=False)
print("Finished processing. Output saved as 'coordinates_with_locations_mapbox.csv'.")
