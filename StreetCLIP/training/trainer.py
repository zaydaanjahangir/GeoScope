import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import os
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
        Initialize the trainer.
        
        Args:
            model: StreetCLIP model
            optimizer: Optimizer for training
            device: Device to train on
            output_dir: Directory to save checkpoints
            learning_rate: Learning rate for training
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Number of gradient accumulation steps
            warmup_epochs: Number of warmup epochs
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_epochs = warmup_epochs
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_gzsl_loss = 0
        total_vision_loss = 0
        num_batches = len(train_loader)
        
        # Setup progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}')
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            collated_metadata = batch['metadata'] # This is the collated dictionary
            batch_size = images.size(0)

            # Reconstruct individual metadata dictionaries for the batch
            individual_metadata = [
                {key: collated_metadata[key][i] for key in collated_metadata}
                for i in range(batch_size)
            ]
            
            # Generate synthetic captions using the reconstructed list
            captions = [
                SyntheticCaptionGenerator.generate_caption(meta)
                for meta in individual_metadata
            ]
            
            # Forward pass
            image_features, text_features = self.model(images, captions)
            
            # Compute loss
            loss, loss_components = self.model.compute_loss(image_features, text_features)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if gradient accumulation is complete
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Apply learning rate warmup
                if epoch < self.warmup_epochs:
                    warmup_factor = min(1.0, (epoch * num_batches + batch_idx + 1) / 
                                     (self.warmup_epochs * num_batches))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate * warmup_factor
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss_components['total_loss']
            total_gzsl_loss += loss_components['gzsl_loss']
            total_vision_loss += loss_components['vision_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'gzsl_loss': total_gzsl_loss / (batch_idx + 1),
                'vision_loss': total_vision_loss / (batch_idx + 1)
            })
        
        # Compute average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'gzsl_loss': total_gzsl_loss / num_batches,
            'vision_loss': total_vision_loss / num_batches
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 3
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'train_gzsl_loss': [],
            'train_vision_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'val_accuracy': []  # Added for plotting, even if not populated
        }
        
        best_val_loss = float('inf')
        
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
                history['val_loss'].append(val_loss)
                
                self.logger.info(f'Validation Loss: {val_loss:.4f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.output_dir, 'best_model.pt'),
                        epoch + 1,
                        val_loss
                    )
            
            # Save checkpoint after each epoch
            self.save_checkpoint(
                os.path.join(self.output_dir, f'epoch_{epoch+1}.pt'),
                epoch + 1,
                val_loss if val_loader is not None else train_metrics['loss']
            )
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.output_dir, 'final_model.pt'),
            num_epochs,
            val_loss if val_loader is not None else train_metrics['loss']
        )
        
        # Save training history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        # Plot training metrics
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
        
        # Compute average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'gzsl_loss': total_gzsl_loss / num_batches,
            'vision_loss': total_vision_loss / num_batches
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