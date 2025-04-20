import torch
import clip
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

class StreetCLIP(torch.nn.Module):
    """StreetCLIP model for geolocalization using CLIP (head‑only fine‑tuning)."""
    
    def __init__(self, clip_model_version: str = "ViT-L/14"):
        """
        Initialize StreetCLIP model.
        
        Args:
            clip_model_version: Version of CLIP model to use (default: ViT-L/14)
        """
        super().__init__()
        # Load pretrained CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model_version)
        
        # ───→ HEAD‑ONLY FINE‑TUNING: freeze the entire CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Add projection layer for vision representation (this is the only trainable layer)
        self.vision_projection = torch.nn.Linear(
            self.clip_model.visual.output_dim,
            self.clip_model.visual.output_dim
        )
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image using CLIP's (now‑frozen) image encoder.
        """
        return self.clip_model.encode_image(image)
    
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode text using CLIP's (now‑frozen) text encoder.
        """
        tokens = clip.tokenize(text).to(next(self.parameters()).device)
        return self.clip_model.encode_text(tokens)
    
    def compute_loss(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined loss for StreetCLIP training.
        """
        # Cast to projection head dtype
        dtype = self.vision_projection.weight.dtype
        image_features = image_features.to(dtype)
        text_features  = text_features.to(dtype)
        
        # Normalize embeddings
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features, dim=-1)
        
        # 1) GZSL loss on frozen embeddings
        logits = (image_features @ text_features.T) / temperature
        labels = torch.arange(len(image_features), device=logits.device)
        gzsl_loss = F.cross_entropy(logits, labels)
        
        # 2) Vision‑projection loss on trainable head
        proj = self.vision_projection(image_features)
        proj = F.normalize(proj, dim=-1)
        vision_logits = (proj @ text_features.T) / temperature
        vision_loss  = F.cross_entropy(vision_logits, labels)
        
        # Total = 0.5*(both)
        total_loss = 0.5 * (gzsl_loss + vision_loss)
        comps = {
            'gzsl_loss':   gzsl_loss.item(),
            'vision_loss': vision_loss.item(),
            'total_loss':  total_loss.item()
        }
        return total_loss, comps
    
    def forward(
        self, 
        images: torch.Tensor, 
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (frozen encoders + head).
        """
        img_feats = self.encode_image(images)
        txt_feats = self.encode_text(texts)
        return img_feats, txt_feats
    
    def predict_location(
        self, 
        image: torch.Tensor, 
        candidate_locations: List[Dict],
        hierarchical: bool = True
    ) -> Dict:
        """
        Predict location for a single image using hierarchical prediction.
        (Unchanged from your version.)
        """
        self.eval()
        with torch.no_grad():
            image_features = self.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
            
            if hierarchical:
                # Country‑level
                from ..utils.synthetic_caption import SyntheticCaptionGenerator
                country_candidates = list({loc['country'] for loc in candidate_locations})
                country_captions   = [f"A photo in {c}." for c in country_candidates]
                country_feats      = self.encode_text(country_captions)
                country_feats      = F.normalize(country_feats, dim=-1)
                sim_c = (image_features @ country_feats.T).squeeze()
                pred_country = country_candidates[sim_c.argmax().item()]
                
                # City‑level
                city_cands = [loc for loc in candidate_locations if loc['country']==pred_country]
                city_caps  = [f"A photo from {loc['city']}." for loc in city_cands]
                city_feats = self.encode_text(city_caps)
                city_feats = F.normalize(city_feats, dim=-1)
                sim_city   = (image_features @ city_feats.T).squeeze()
                return city_cands[sim_city.argmax().item()]
            else:
                # Flat city
                caps = [SyntheticCaptionGenerator.generate_caption(loc) for loc in candidate_locations]
                txt = self.encode_text(caps)
                txt = F.normalize(txt, dim=-1)
                sim = (image_features @ txt.T).squeeze()
                return candidate_locations[sim.argmax().item()]
