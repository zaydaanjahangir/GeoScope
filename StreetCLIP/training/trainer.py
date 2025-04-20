import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import os
import random
from utils.synthetic_caption import SyntheticCaptionGenerator
from utils.plotting import plot_training_metrics
import json

class StreetCLIPTrainer:
    """Trainer for StreetCLIP model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str,
        learning_rate: float = 1e-6,
        weight_decay: float = 1e-4,
        gradient_accumulation_steps: int = 12,
        warmup_epochs: float = 0.6
    ):
        """
        Initialize trainer: attempt to enable CLIP checkpointing if available, always set up AMP.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_epochs = warmup_epochs

        # Setup logging EARLY so we can warn
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Try to enable gradient checkpointing in CLIP’s visual transformer
        try:
            self.model.clip_model.visual.gradient_checkpointing_enable()
            self.logger.info("Enabled gradient checkpointing on CLIP visual transformer.")
        except Exception:
            self.logger.warning(
                " CLIP visual transformer has no gradient_checkpointing_enable(); skipping checkpointing."
            )

        # AMP GradScaler for mixed‐precision
        self.scaler = GradScaler()



        
        

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        One epoch of training: FP16 forward, FP32 loss, gradient clipping,
        per‐step LR warmup, and AMP‐step fallback on error.
        """
        self.model.train()
        total_loss = total_gzsl_loss = total_vision_loss = 0.0
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}')
        self.optimizer.zero_grad()

        warmup_steps = max(1, int(self.warmup_epochs * num_batches))

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            collated = batch['metadata']
            batch_size = images.size(0)

            # build captions
            individual = [
                {k: collated[k][i] for k in collated}
                for i in range(batch_size)
            ]
            captions = [SyntheticCaptionGenerator.generate_caption(md)
                        for md in individual]

            # FP16 forward
            with autocast("cuda"):
                img_feats, txt_feats = self.model(images, captions)

            # FP32 loss (move compute out of autocast)
            loss, comps = self.model.compute_loss(img_feats, txt_feats)
            loss = loss / self.gradient_accumulation_steps

            # skip bad batches
            if not torch.isfinite(loss):
                self.logger.warning(f"Non‐finite loss {loss} at batch {batch_idx}, skipping")
                self.optimizer.zero_grad()
                continue

            # backward
            self.scaler.scale(loss).backward()

            # step & clip every accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # warmup LR
                current_step = (epoch - 1) * num_batches + (batch_idx + 1)
                if current_step <= warmup_steps:
                    lr = self.learning_rate * (current_step / warmup_steps)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr

                # attempt AMP step, fallback on plain step
                try:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except ValueError:
                    self.logger.warning("AMP step failed, falling back to optimizer.step()")
                    self.optimizer.step()
                # zero grads
                self.optimizer.zero_grad()

            # accumulate metrics
            total_loss        += comps['total_loss']
            total_gzsl_loss   += comps['gzsl_loss']
            total_vision_loss += comps['vision_loss']

            pbar.set_postfix({
                'loss':        total_loss        / (batch_idx + 1),
                'gzsl_loss':   total_gzsl_loss   / (batch_idx + 1),
                'vision_loss': total_vision_loss / (batch_idx + 1),
            })

        return {
            'loss':        total_loss        / num_batches,
            'gzsl_loss':   total_gzsl_loss   / num_batches,
            'vision_loss': total_vision_loss / num_batches,
        }



    
    def generate_candidate_locations(self, dataloader: DataLoader, max_candidates: int = 100) -> List[Dict]:
        """
        Generate candidate locations from the dataloader.
        
        Args:
            dataloader: DataLoader containing location data
            max_candidates: Maximum number of candidate locations to generate
            
        Returns:
            List of candidate locations
        """
        self.logger.info("Generating candidate locations for geographic evaluation...")
        all_locations = set()
        location_metadata = []
        
        # Collect unique locations
        for batch in tqdm(dataloader, desc="Collecting locations"):
            collated_metadata = batch['metadata']
            batch_size = len(next(iter(collated_metadata.values())))
            
            # Reconstruct individual metadata dictionaries
            metadata_list = [
                {key: collated_metadata[key][i] for key in collated_metadata}
                for i in range(batch_size)
            ]
            
            for metadata in metadata_list:
                location_key = (
                    metadata.get('city', ''), 
                    metadata.get('country', ''),
                    metadata.get('continent', '')
                )
                if location_key not in all_locations:
                    all_locations.add(location_key)
                    location_metadata.append({
                        'city': metadata.get('city', ''),
                        'region': metadata.get('region', ''),
                        'country': metadata.get('country', ''),
                        'continent': metadata.get('continent', '')
                    })
        
        # Sample candidates if we have more than max_candidates
        candidates = location_metadata
        if len(candidates) > max_candidates:
            candidates = random.sample(candidates, max_candidates)
        
        self.logger.info(f"Generated {len(candidates)} candidate locations")
        return candidates
    
    def evaluate_geographic_accuracy(
        self, 
        val_loader: DataLoader,
        candidates: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate geographic accuracy on validation data.
        
        Args:
            val_loader: Validation data loader
            candidates: List of candidate locations
            
        Returns:
            Dictionary with accuracy metrics
        """
        self.logger.info("Evaluating geographic accuracy...")
        self.model.eval()
        
        total_samples = 0
        correct_country = 0
        correct_city = 0
        correct_continent = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating geography"):
                images = batch['image'].to(self.device)
                collated_metadata = batch['metadata']
                batch_size = images.size(0)
                
                # Reconstruct individual metadata dictionaries
                individual_metadata = [
                    {key: collated_metadata[key][i] for key in collated_metadata}
                    for i in range(batch_size)
                ]
                
                # Get image features
                for i, image in enumerate(images):
                    metadata = individual_metadata[i]
                    
                    # Skip samples with missing metadata
                    if not (metadata.get('country') and metadata.get('city') and metadata.get('continent')):
                        continue
                    
                    # Encode image
                    image_features = self.model.encode_image(image.unsqueeze(0))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # For each candidate location, create a caption and encode it
                    captions = [
                        SyntheticCaptionGenerator.generate_caption(candidate)
                        for candidate in candidates
                    ]
                    text_features = self.model.encode_text(captions)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarities and get prediction
                    similarities = (image_features @ text_features.T).squeeze()
                    pred_idx = similarities.argmax().item()
                    prediction = candidates[pred_idx]
                    
                    # Update counters
                    total_samples += 1
                    if prediction['continent'] == metadata['continent']:
                        correct_continent += 1
                    if prediction['country'] == metadata['country']:
                        correct_country += 1
                    if prediction['city'] == metadata['city']:
                        correct_city += 1
        
        # Calculate accuracies
        if total_samples == 0:
            self.logger.warning("No valid samples for geographic evaluation")
            return {
                'continent_accuracy': 0.0,
                'country_accuracy': 0.0,
                'city_accuracy': 0.0
            }
        
        continent_accuracy = correct_continent / total_samples
        country_accuracy = correct_country / total_samples
        city_accuracy = correct_city / total_samples
        
        self.logger.info(f"Geographic Evaluation Results (on {total_samples} samples):")
        self.logger.info(f"  Continent Accuracy: {continent_accuracy:.4f}")
        self.logger.info(f"  Country Accuracy: {country_accuracy:.4f}")
        self.logger.info(f"  City Accuracy: {city_accuracy:.4f}")
        
        return {
            'continent_accuracy': continent_accuracy,
            'country_accuracy': country_accuracy,
            'city_accuracy': city_accuracy
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        geo_eval_frequency: int = 1  # Evaluate geographic accuracy every N epochs
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            geo_eval_frequency: Frequency of geographic accuracy evaluation
            
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'train_gzsl_loss': [],
            'train_vision_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            # Geographic accuracy metrics that will be populated during evaluation
            'country_accuracy': [],
            'city_accuracy': [],
            'continent_accuracy': []
        }
        
        best_val_loss = float('inf')
        
        # Generate candidate locations from validation set if available, otherwise from training set
        candidates = None
        if val_loader is not None:
            candidates = self.generate_candidate_locations(val_loader)
        else:
            # Sample a subset of train_loader to generate candidates
            # (avoiding using the same data for training and evaluation)
            candidates = self.generate_candidate_locations(train_loader)
        
        for epoch in range(num_epochs):
            # Track current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch + 1, num_epochs)
            
            # Log training metrics
            self.logger.info(
                f'Epoch {epoch + 1}/{num_epochs} - '
                f'Train Loss: {train_metrics["loss"]:.4f} - '
                f'GZSL Loss: {train_metrics["gzsl_loss"]:.4f} - '
                f'Vision Loss: {train_metrics["vision_loss"]:.4f}'
            )
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_gzsl_loss'].append(train_metrics['gzsl_loss'])
            history['train_vision_loss'].append(train_metrics['vision_loss'])
            
            # Validate if validation loader is provided
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                val_accuracy = val_metrics['accuracy']
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                self.logger.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.output_dir, 'best_model.pt'),
                        epoch + 1,
                        val_loss
                    )
                
                # Evaluate geographic accuracy at specified frequency
                if (epoch + 1) % geo_eval_frequency == 0 or epoch == num_epochs - 1:
                    geo_metrics = self.evaluate_geographic_accuracy(val_loader, candidates)
                    
                    # Update history with current geo metrics
                    history['continent_accuracy'].append(geo_metrics['continent_accuracy'])
                    history['country_accuracy'].append(geo_metrics['country_accuracy'])
                    history['city_accuracy'].append(geo_metrics['city_accuracy'])
                    
                    # Fill in missing values for previous epochs
                    while len(history['continent_accuracy']) < epoch + 1:
                        history['continent_accuracy'].insert(0, 0.0)
                    while len(history['country_accuracy']) < epoch + 1:
                        history['country_accuracy'].insert(0, 0.0)
                    while len(history['city_accuracy']) < epoch + 1:
                        history['city_accuracy'].insert(0, 0.0)
            
            # Save checkpoint after each epoch
            self.save_checkpoint(
                os.path.join(self.output_dir, f'epoch_{epoch+1}.pt'),
                epoch + 1,
                val_loss if val_loader is not None else train_metrics['loss']
            )
            
            # Save and plot intermediate history
            with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=4)
            
            plot_training_metrics(history, self.output_dir)
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.output_dir, 'final_model.pt'),
            num_epochs,
            val_loss if val_loader is not None else train_metrics['loss']
        )
        
        # Final save of training history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        # Plot final training metrics
        plot_training_metrics(history, self.output_dir)
        
        return history
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_gzsl_loss = 0
        total_vision_loss = 0
        num_batches = len(val_loader)
        
        # For accuracy calculation
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                images = batch['image'].to(self.device)
                collated_metadata = batch['metadata'] # Collated dictionary
                batch_size = images.size(0)

                # Reconstruct individual metadata dictionaries
                individual_metadata = [
                    {key: collated_metadata[key][i] for key in collated_metadata}
                    for i in range(batch_size)
                ]
                
                # Generate synthetic captions
                captions = [
                    SyntheticCaptionGenerator.generate_caption(meta)
                    for meta in individual_metadata
                ]
                
                # Forward pass
                image_features, text_features = self.model(images, captions)
                
                # Compute loss
                loss, loss_components = self.model.compute_loss(image_features, text_features)
                
                # Update metrics
                total_loss += loss_components['total_loss']
                total_gzsl_loss += loss_components['gzsl_loss']
                total_vision_loss += loss_components['vision_loss']
                
                # Calculate accuracy (similarity-based)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity matrix
                similarities = (image_features @ text_features.T)
                
                # Get predictions (diagonal should be highest for correct matches)
                predictions = torch.argmax(similarities, dim=1)
                targets = torch.arange(len(predictions)).to(self.device)
                
                # Calculate correct predictions
                correct = (predictions == targets).sum().item()
                correct_predictions += correct
                total_samples += len(predictions)
        
        # Compute average metrics
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        metrics = {
            'loss': total_loss / num_batches,
            'gzsl_loss': total_gzsl_loss / num_batches,
            'vision_loss': total_vision_loss / num_batches,
            'accuracy': accuracy  # Add accuracy to validation metrics
        }
        
        return metrics
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, path)
        self.logger.info(f'Saved checkpoint to {path}')
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f'Loaded checkpoint from {path}') 