from mapbox import Directions, Geocoder
import json
from dotenv import load_dotenv
import os
from geopy.geocoders import Nominatim
load_dotenv()
# Replace with your Mapbox access token
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
service = Directions(access_token=MAPBOX_ACCESS_TOKEN)
def get_directions(origin, destination):
        
    # Define origin and destination coordinates (longitude, latitude)
    origin = (-122.42, 37.78)  # San Francisco
    destination = (-122.39, 37.79)  # Oakland

    # Initialize the Directions service
    

    # Fetch directions
    response = service.directions([origin, destination], profile='mapbox/walking')

    # Access the actual route content (it's a nested structure)
    route = response.json()['routes'][0]

    # Extract steps (verbal instructions)
    steps = route['legs'][0]['steps']

    # Print verbal directions
    for idx, step in enumerate(steps, start=1):
        instruction = step['maneuver']['instruction']
        distance = round(step['distance'], 1)
        print(f"{idx}. {instruction} ({distance} meters)")

    

# Your Mapbox token


# Initialize clients
geocoder = Geocoder(access_token=MAPBOX_ACCESS_TOKEN)
directions = Directions(access_token=MAPBOX_ACCESS_TOKEN)

def resolve_location(loc):
    """
    Accepts either:
      - A (lon, lat) tuple: returns it unchanged
      - A string: forwards‐geocodes it and returns (lon, lat) of the top result
    """
    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        return loc
    # Otherwise assume text: do forward geocoding
    
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(loc)
    if location is None:
        return None
    print(location.address)       # full normalized address
    print(location.longitude, location.latitude)
    
    return (location.longitude, location.latitude)

def get_walking_instructions(origin, destination):
    

    try:
        # Resolve both endpoints
        orig_coord = resolve_location(origin)
        dest_coord = resolve_location(destination)
        print(orig_coord, dest_coord)
        # Fetch turn‐by‐turn directions
        resp = directions.directions(
            [orig_coord, dest_coord],
            profile="mapbox/walking",
            steps=True,
            geometries="geojson",
            overview="full"
        )
        data = resp.json()

        if 'code' in data:
            if data['code'] == 'InvalidInput':
                return "Internal Server error"
        # Extract steps (verbal instructions)
        steps = data['routes'][0]['legs'][0]['steps']
        steps_str =  ""
        for idx, step in enumerate(steps, start=1):
            instruction = step['maneuver']['instruction']
            
            distance = round(step['distance'], 1)
            print(f"{idx}. {instruction} ({distance} meters)")
            steps_str += f"{idx}. {instruction} ({distance} meters)\n"
        if (steps_str).strip()== "":
            return "no data found or an error."
        return steps_str
    except Exception as e:
        print(f"Exception while getting walking instructions: {e}")
        return "no Data  found or an error."

