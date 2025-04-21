import torch
import clip
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

class StreetCLIP(torch.nn.Module):
    """StreetCLIP model for geolocalization using CLIP."""
    
    def __init__(self, clip_model_version: str = "ViT-L/14"):
        super().__init__()
        # 1) Load pretrained CLIP
        self.clip_model, self.preprocess = clip.load(clip_model_version)

        # 2) Freeze all CLIP weights by default
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 3) But unfreeze the last 2 transformer blocks + the final layer norm in the visual tower
        visual = self.clip_model.visual
        # `resblocks` is an nn.Sequential of the transformer blocks
        resblocks = visual.transformer.resblocks  
        for idx in (-1, -2):   
            for p in resblocks[idx].parameters():
                p.requires_grad = True

        # Unfreeze the final norm (ln_post) as well
        for p in visual.ln_post.parameters():
            p.requires_grad = True

        # 4) Your trainable projection head
        self.vision_projection = torch.nn.Linear(
            visual.output_dim,
            visual.output_dim
        )
        # (vision_projection.parameters() are True by default)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(image)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(text).to(next(self.parameters()).device)
        return self.clip_model.encode_text(tokens)

    def compute_loss(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # (same as before)
        weight_dtype = self.vision_projection.weight.dtype
        image_features = image_features.to(dtype=weight_dtype)
        text_features  = text_features.to(dtype=weight_dtype)

        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features,  dim=-1)

        # project + normalize
        proj_img = self.vision_projection(image_features)
        proj_img = F.normalize(proj_img, dim=-1)

        labels = torch.arange(len(image_features), device=image_features.device)

        # GZSL loss
        logits = (image_features @ text_features.T) / temperature
        gzsl_loss = F.cross_entropy(logits, labels)

        # vision‐head loss
        vision_logits = (proj_img @ text_features.T) / temperature
        vision_loss = F.cross_entropy(vision_logits, labels)

        total_loss = 0.5 * (gzsl_loss + vision_loss)
        return total_loss, {
            'gzsl_loss':   gzsl_loss.item(),
            'vision_loss': vision_loss.item(),
            'total_loss':  total_loss.item()
        }

    def forward(self, images: torch.Tensor, texts: List[str]):
        img_f = self.encode_image(images)
        txt_f = self.encode_text(texts)
        # cast & return
        wdt = self.vision_projection.weight.dtype
        return img_f.to(wdt), txt_f.to(wdt)
    
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
