import torch
import clip
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

class StreetCLIP(torch.nn.Module):
    """StreetCLIP model for geolocalization using CLIP."""
    
    def __init__(self, clip_model_version: str = "ViT-L/14"):
        """
        Initialize StreetCLIP model.
        
        Args:
            clip_model_version: Version of CLIP model to use (default: ViT-L/14)
        """
        super().__init__()
        # Load pretrained CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model_version)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Add projection layer for vision representation
        self.vision_projection = torch.nn.Linear(
            self.clip_model.visual.output_dim,
            self.clip_model.visual.output_dim
        )
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image using CLIP's image encoder.
        
        Args:
            image: Image tensor
            
        Returns:
            Image features
        """
        return self.clip_model.encode_image(image)
    
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode text using CLIP's text encoder.
        
        Args:
            text: List of text strings
            
        Returns:
            Text features
        """
        # Tokenize text
        text_tokens = clip.tokenize(text).to(next(self.parameters()).device)
        return self.clip_model.encode_text(text_tokens)
    
    def compute_loss(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined loss for StreetCLIP training.
        
        Args:
            image_features: Image feature embeddings
            text_features: Text feature embeddings
            temperature: Temperature parameter for scaling logits
            
        Returns:
            Tuple of (total loss, loss components dictionary)
        """
        # Ensure consistent data types by converting to the same type as the vision projection layer
        weight_dtype = self.vision_projection.weight.dtype
        image_features = image_features.to(dtype=weight_dtype)
        text_features = text_features.to(dtype=weight_dtype)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Project image features for vision representation
        projected_image_features = self.vision_projection(image_features)
        projected_image_features = F.normalize(projected_image_features, dim=-1)
        
        # Compute GZSL loss (L_GZSL)
        logits = (image_features @ text_features.T) / temperature
        labels = torch.arange(len(image_features)).to(image_features.device)
        gzsl_loss = F.cross_entropy(logits, labels)
        
        # Compute vision representation loss (L_Vision Representation)
        vision_logits = (projected_image_features @ text_features.T) / temperature
        vision_loss = F.cross_entropy(vision_logits, labels)
        
        # Combined loss (L_CLIP = 0.5 * (L_GZSL + L_Vision Representation))
        total_loss = (gzsl_loss + vision_loss) / 2
        
        # Return loss components for logging
        loss_components = {
            'gzsl_loss': gzsl_loss.item(),
            'vision_loss': vision_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def forward(
        self, 
        images: torch.Tensor, 
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            images: Batch of images
            texts: List of text strings
            
        Returns:
            Tuple of (image_features, text_features)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        # Ensure consistent data types
        weight_dtype = self.vision_projection.weight.dtype
        image_features = image_features.to(dtype=weight_dtype)
        text_features = text_features.to(dtype=weight_dtype)
        
        return image_features, text_features
    
    def predict_location(
        self, 
        image: torch.Tensor, 
        candidate_locations: List[Dict],
        hierarchical: bool = True
    ) -> Dict:
        """
        Predict location for a single image using hierarchical prediction.
        
        Args:
            image: Input image
            candidate_locations: List of candidate location dictionaries
            hierarchical: Whether to use hierarchical prediction (country then city)
            
        Returns:
            Predicted location dictionary
        """
        self.eval()
        with torch.no_grad():
            # Encode image
            image_features = self.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
            
            if hierarchical:
                # First predict country
                from ..utils.synthetic_caption import SyntheticCaptionGenerator
                country_candidates = list(set(loc['country'] for loc in candidate_locations))
                country_captions = [
                    f"A photo in {country}." for country in country_candidates
                ]
                
                country_features = self.encode_text(country_captions)
                country_features = F.normalize(country_features, dim=-1)
                country_similarities = (image_features @ country_features.T).squeeze()
                pred_country_idx = country_similarities.argmax().item()
                pred_country = country_candidates[pred_country_idx]
                
                # Then predict city within the country
                city_candidates = [
                    loc for loc in candidate_locations 
                    if loc['country'] == pred_country
                ]
                city_captions = [
                    f"A photo from {loc['city']}." for loc in city_candidates
                ]
                
                city_features = self.encode_text(city_captions)
                city_features = F.normalize(city_features, dim=-1)
                city_similarities = (image_features @ city_features.T).squeeze()
                pred_city_idx = city_similarities.argmax().item()
                return city_candidates[pred_city_idx]
            else:
                # Non-hierarchical prediction (direct city prediction)
                candidate_captions = [
                    SyntheticCaptionGenerator.generate_caption(loc) 
                    for loc in candidate_locations
                ]
                
                text_features = self.encode_text(candidate_captions)
                text_features = F.normalize(text_features, dim=-1)
                
                similarities = (image_features @ text_features.T).squeeze()
                pred_idx = similarities.argmax().item()
                return candidate_locations[pred_idx] 