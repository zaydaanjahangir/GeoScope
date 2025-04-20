import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
import os
import json
from utils.synthetic_caption import SyntheticCaptionGenerator
from utils.plotting import plot_evaluation_metrics

class StreetCLIPEvaluator:
    """Evaluator for StreetCLIP model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: StreetCLIP model
            device: Device to evaluate on
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'evaluation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self, 
        test_dataloader: DataLoader,
        candidate_locations: List[Dict]
    ) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_dataloader: DataLoader for test data
            candidate_locations: List of candidate locations
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_samples = 0
        correct_country = 0
        correct_city = 0
        correct_continent = 0  # Track continent accuracy
        
        # Progress bar
        pbar = tqdm(test_dataloader, desc='Evaluating')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                collated_metadata = batch['metadata']
                batch_size = images.size(0)

                # Reconstruct individual metadata dictionaries
                individual_metadata = [
                    {key: collated_metadata[key][i] for key in collated_metadata}
                    for i in range(batch_size)
                ]
                
                # Get predictions
                for i, image in enumerate(images):
                    metadata = individual_metadata[i]
                    pred = self.model.predict_location(image.unsqueeze(0), candidate_locations)
                    
                    total_samples += 1
                    
                    # Check continent accuracy
                    if 'continent' in pred and 'continent' in metadata and pred['continent'] == metadata['continent']:
                        correct_continent += 1
                    
                    # Country accuracy
                    if 'country' in pred and 'country' in metadata and pred['country'] == metadata['country']:
                        correct_country += 1
                    
                    # City accuracy
                    if 'city' in pred and 'city' in metadata and pred['city'] == metadata['city']:
                        correct_city += 1
        
        # Calculate final metrics
        continent_accuracy = correct_continent / total_samples if total_samples > 0 else 0
        country_accuracy = correct_country / total_samples if total_samples > 0 else 0
        city_accuracy = correct_city / total_samples if total_samples > 0 else 0
        
        # Log results
        self.logger.info(f'Total samples: {total_samples}')
        self.logger.info(f'Continent accuracy: {continent_accuracy:.4f}')
        self.logger.info(f'Country accuracy: {country_accuracy:.4f}')
        self.logger.info(f'City accuracy: {city_accuracy:.4f}')
        
        metrics = {
            'total_samples': total_samples,
            'continent_accuracy': continent_accuracy,
            'country_accuracy': country_accuracy,
            'city_accuracy': city_accuracy
        }
        
        # Save metrics to file
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot evaluation metrics
        plot_evaluation_metrics(metrics, self.output_dir)
        
        return metrics
    
    def hierarchical_evaluation(
        self, 
        test_dataloader: DataLoader,
        country_candidates: List[str],
        city_candidates: Dict[str, List[str]],
        continent_candidates: Optional[List[str]] = None,
        continent_countries: Optional[Dict[str, List[str]]] = None
    ) -> Dict:
        """
        Perform hierarchical evaluation (continent -> country -> city).
        
        Args:
            test_dataloader: DataLoader for test data
            country_candidates: List of candidate countries
            city_candidates: Dictionary mapping countries to lists of cities
            continent_candidates: List of candidate continents (optional)
            continent_countries: Dictionary mapping continents to lists of countries (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_samples = 0
        correct_country = 0
        correct_city = 0
        correct_continent = 0
        
        # Progress bar
        pbar = tqdm(test_dataloader, desc='Evaluating')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                collated_metadata = batch['metadata']
                batch_size = images.size(0)

                # Reconstruct individual metadata dictionaries
                individual_metadata = [
                    {key: collated_metadata[key][i] for key in collated_metadata}
                    for i in range(batch_size)
                ]
                
                for i, image in enumerate(images):
                    metadata = individual_metadata[i]
                    image_features = self.model.encode_image(image.unsqueeze(0))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Continent prediction (if available)
                    pred_continent = None
                    if continent_candidates:
                        continent_captions = [
                            f"A photo from the continent of {continent}"
                            for continent in continent_candidates
                        ]
                        text_features = self.model.encode_text(continent_captions)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        continent_similarities = (image_features @ text_features.T).squeeze()
                        pred_continent_idx = continent_similarities.argmax().item()
                        pred_continent = continent_candidates[pred_continent_idx]
                        
                        if pred_continent == metadata['continent']:
                            correct_continent += 1
                            
                    # Country prediction
                    # If continent was predicted correctly and we have continent_countries mapping,
                    # only consider countries in that continent
                    country_list = country_candidates
                    if pred_continent and continent_countries and pred_continent in continent_countries:
                        country_list = continent_countries[pred_continent]
                        
                    country_captions = [
                        SyntheticCaptionGenerator.generate_country_caption(country)
                        for country in country_list
                    ]
                    text_features = self.model.encode_text(country_captions)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    country_similarities = (image_features @ text_features.T).squeeze()
                    pred_country_idx = country_similarities.argmax().item()
                    pred_country = country_list[pred_country_idx]
                    
                    # City prediction (only for correct country)
                    if pred_country == metadata['country']:
                        correct_country += 1
                        
                        # Get cities for predicted country
                        if pred_country in city_candidates:
                            cities = city_candidates[pred_country]
                            city_captions = [
                                SyntheticCaptionGenerator.generate_city_caption(city)
                                for city in cities
                            ]
                            
                            # Get city prediction
                            text_features = self.model.encode_text(city_captions)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                            city_similarities = (image_features @ text_features.T).squeeze()
                            pred_city_idx = city_similarities.argmax().item()
                            pred_city = cities[pred_city_idx]
                            
                            if pred_city == metadata['city']:
                                correct_city += 1
                    
                    total_samples += 1
        
        # Calculate final metrics
        continent_accuracy = correct_continent / total_samples if continent_candidates and total_samples > 0 else None
        country_accuracy = correct_country / total_samples if total_samples > 0 else 0
        city_accuracy = correct_city / total_samples if total_samples > 0 else 0
        
        # Log results
        self.logger.info(f'Total samples: {total_samples}')
        if continent_accuracy is not None:
            self.logger.info(f'Continent accuracy: {continent_accuracy:.4f}')
        self.logger.info(f'Country accuracy: {country_accuracy:.4f}')
        self.logger.info(f'City accuracy: {city_accuracy:.4f}')
        
        metrics = {
            'total_samples': total_samples,
            'country_accuracy': country_accuracy,
            'city_accuracy': city_accuracy
        }
        
        if continent_accuracy is not None:
            metrics['continent_accuracy'] = continent_accuracy
        
        # Save metrics to file
        metrics_path = os.path.join(self.output_dir, 'hierarchical_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot evaluation metrics
        plot_evaluation_metrics(metrics, self.output_dir)
        
        return metrics 