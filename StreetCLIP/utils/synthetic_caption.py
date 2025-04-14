import pandas as pd
from typing import Dict, List, Optional

class SyntheticCaptionGenerator:
    """Generates synthetic captions for geolocalization using location metadata."""
    
    @staticmethod
    def generate_caption(location_metadata: Dict) -> str:
        """
        Generate a synthetic caption for a location using the template.
        
        Args:
            location_metadata: Dictionary containing location information
                with keys: city, region, country, continent
        
        Returns:
            str: Generated synthetic caption
        """
        try:
            return (
                f"A Street View photo close to the town of {location_metadata['city']} "
                f"in the region of {location_metadata['region']} "
                f"in {location_metadata['country']} "
                f"in the continent of {location_metadata['continent']}"
            )
        except KeyError as e:
            # Build caption with available fields, skipping missing ones
            parts = []
            if location_metadata.get('city'):
                parts.append(f"A Street View photo close to the town of {location_metadata['city']}")
            if location_metadata.get('region'):
                parts.append(f"in the region of {location_metadata['region']}")
            if location_metadata.get('country'):
                parts.append(f"in {location_metadata['country']}")
            if location_metadata.get('continent'):
                parts.append(f"in the continent of {location_metadata['continent']}")
            return " ".join(parts) if parts else "A Street View photo"
    
    @staticmethod
    def process_csv(input_csv_path: str, output_csv_path: str) -> None:
        """
        Process a CSV file to add synthetic captions for each location.
        
        Args:
            input_csv_path: Path to input CSV with location data
            output_csv_path: Path where the output CSV will be saved
        """
        # Read the CSV
        df = pd.read_csv(input_csv_path)
        
        # Verify required columns exist
        required_columns = ['city', 'region', 'country', 'continent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # Generate captions for each row
        df['caption'] = df.apply(
            lambda row: SyntheticCaptionGenerator.generate_caption({
                'city': row['city'],
                'region': row['region'],
                'country': row['country'],
                'continent': row['continent']
            }),
            axis=1
        )
        
        # Save the updated DataFrame to a new CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Generated captions and saved to {output_csv_path}")
    
    @staticmethod
    def generate_country_caption(country: str) -> str:
        """Generate a caption for country-level prediction."""
        return f"A photo in {country}."
    
    @staticmethod
    def generate_city_caption(city: str) -> str:
        """Generate a caption for city-level prediction."""
        return f"A photo from {city}." 