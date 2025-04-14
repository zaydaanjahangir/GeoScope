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
        
        # Progress bar
        pbar = tqdm(test_dataloader, desc='Evaluating')
        
        with torch.no_grad():
            for images, metadata in pbar:
                # Move data to device
                images = images.to(self.device)
                
                # Get predictions
                predictions = [
                    self.model.predict_location(image.unsqueeze(0), candidate_locations)
                    for image in images
                ]
                
                # Calculate metrics
                for pred, true in zip(predictions, metadata):
                    total_samples += 1
                    
                    # Country accuracy
                    if pred['country'] == true['country']:
                        correct_country += 1
                    
                    # City accuracy
                    if pred['city'] == true['city']:
                        correct_city += 1
        
        # Calculate final metrics
        country_accuracy = correct_country / total_samples
        city_accuracy = correct_city / total_samples
        
        # Log results
        self.logger.info(f'Total samples: {total_samples}')
        self.logger.info(f'Country accuracy: {country_accuracy:.4f}')
        self.logger.info(f'City accuracy: {city_accuracy:.4f}')
        
        metrics = {
            'total_samples': total_samples,
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
        city_candidates: Dict[str, List[str]]
    ) -> Dict:
        """
        Perform hierarchical evaluation (country then city).
        
        Args:
            test_dataloader: DataLoader for test data
            country_candidates: List of candidate countries
            city_candidates: Dictionary mapping countries to lists of cities
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_samples = 0
        correct_country = 0
        correct_city = 0
        
        # Progress bar
        pbar = tqdm(test_dataloader, desc='Evaluating')
        
        with torch.no_grad():
            for images, metadata in pbar:
                # Move data to device
                images = images.to(self.device)
                
                for image, true_meta in zip(images, metadata):
                    # Country prediction
                    country_captions = [
                        SyntheticCaptionGenerator.generate_country_caption(country)
                        for country in country_candidates
                    ]
                    image_features = self.model.encode_image(image.unsqueeze(0))
                    text_features = self.model.encode_text(country_captions)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Get country prediction
                    country_similarities = (image_features @ text_features.T).squeeze()
                    pred_country_idx = country_similarities.argmax().item()
                    pred_country = country_candidates[pred_country_idx]
                    
                    # City prediction (only for correct country)
                    if pred_country == true_meta['country']:
                        correct_country += 1
                        
                        # Get cities for predicted country
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
                        
                        if pred_city == true_meta['city']:
                            correct_city += 1
                    
                    total_samples += 1
        
        # Calculate final metrics
        country_accuracy = correct_country / total_samples
        city_accuracy = correct_city / total_samples
        
        # Log results
        self.logger.info(f'Total samples: {total_samples}')
        self.logger.info(f'Country accuracy: {country_accuracy:.4f}')
        self.logger.info(f'City accuracy: {city_accuracy:.4f}')
        
        metrics = {
            'total_samples': total_samples,
            'country_accuracy': country_accuracy,
            'city_accuracy': city_accuracy
        }
        
        # Save metrics to file
        metrics_path = os.path.join(self.output_dir, 'hierarchical_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot evaluation metrics
        plot_evaluation_metrics(metrics, self.output_dir)
        
        return metrics 