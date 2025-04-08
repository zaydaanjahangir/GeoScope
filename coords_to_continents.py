import time
import pandas as pd
from geopy.geocoders import Nominatim
import pycountry_convert as pc
from tqdm import tqdm

# Initialize the geolocator with a valid user agent.
geolocator = Nominatim(user_agent="GeoScopeApp")

def latlong_to_continent(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en', timeout=10)
        if location is None or 'address' not in location.raw:
            return None
        address = location.raw['address']
        country_code = address.get('country_code', None)
        if not country_code:
            return None
        
        # Convert country code to continent code and then to continent name.
        continent_code = pc.country_alpha2_to_continent_code(country_code.upper())
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except Exception as e:
        print(f"Error for ({lat}, {lon}): {e}")
        return None

# Read the coordinates CSV. Assume the CSV has columns 'latitude' and 'longitude'
df = pd.read_csv("Streetview_Image_Dataset/coordinates.csv")

# Create a list to store continent labels.
continents = []

# Process each row with progress tracking.
for index, row in tqdm(df.iterrows(), total=len(df)):
    lat, lon = row['latitude'], row['longitude']
    continent = latlong_to_continent(lat, lon)
    continents.append(continent)
    time.sleep(1)  # Enforce a 1-second delay between requests

    # Every 1000 iterations, save intermediate results.
    if (index + 1) % 1000 == 0:
        df['continent'] = continents
        intermediate_filename = f"coordinates_with_continents_{index+1}.csv"
        df.to_csv(intermediate_filename, index=False)
        print(f"Intermediate save: {intermediate_filename}")

# Assign final continent labels and save complete CSV.
df['continent'] = continents
df.to_csv("coordinates_with_continents.csv", index=False)
print("Finished processing. Output saved as 'coordinates_with_continents.csv'.")
