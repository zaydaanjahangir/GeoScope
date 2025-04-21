import pandas as pd
from typing import Dict


class SyntheticCaptionGenerator:
    @staticmethod
    def generate_caption(location_metadata: Dict) -> str:
        # primary template; KeyError triggers fallback below
        try:
            return (
                f"A Street View photo close to the town of {location_metadata['city']} "
                f"in the region of {location_metadata['region']} "
                f"in {location_metadata['country']} "
                f"in the continent of {location_metadata['continent']}"
            )
        except KeyError:
            # assemble whatever fields are present
            parts = []
            if location_metadata.get('city'):
                parts.append(f"A Street View photo close to the town of {location_metadata['city']}")
            if location_metadata.get('region'):
                parts.append(f"in the region of {location_metadata['region']}")
            if location_metadata.get('country'):
                parts.append(f"in {location_metadata['country']}")
            if location_metadata.get('continent'):
                parts.append(f"in the continent of {location_metadata['continent']}")
            return " ".join(parts) or "A Street View photo"

    @staticmethod
    def process_csv(input_csv_path: str, output_csv_path: str) -> None:
        df = pd.read_csv(input_csv_path)

        # ensure all required columns are present before proceeding
        required_columns = ['city', 'region', 'country', 'continent']
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # generate one caption per row using the same fallback logic
        df['caption'] = df.apply(
            lambda row: SyntheticCaptionGenerator.generate_caption({
                'city': row['city'],
                'region': row['region'],
                'country': row['country'],
                'continent': row['continent']
            }),
            axis=1
        )

        df.to_csv(output_csv_path, index=False)
        print(f"Generated captions and saved to {output_csv_path}")

    @staticmethod
    def generate_country_caption(country: str) -> str:
        return f"A photo in {country}."

    @staticmethod
    def generate_city_caption(city: str) -> str:
        return f"A photo from {city}."
