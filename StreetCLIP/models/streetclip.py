import torch
import clip
import torch.nn.functional as F
from typing import List, Dict, Optional


class StreetCLIP(torch.nn.Module):
    def __init__(self, clip_model_version: str = "ViT-L/14"):
        super().__init__()
        # load CLIP and its preprocessing pipeline
        self.clip_model, self.preprocess = clip.load(clip_model_version)

        # freeze all CLIP parameters…
        for p in self.clip_model.parameters():
            p.requires_grad = False
        # …then unfreeze the last two transformer blocks and final layer norm so they can adapt
        visual = self.clip_model.visual
        blocks = visual.transformer.resblocks
        for idx in (-1, -2):
            for p in blocks[idx].parameters():
                p.requires_grad = True
        for p in visual.ln_post.parameters():
            p.requires_grad = True

        # trainable projection head mapping visual features back to CLIP’s embedding space
        self.vision_projection = torch.nn.Linear(visual.output_dim, visual.output_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(images)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts).to(next(self.parameters()).device)
        return self.clip_model.encode_text(tokens)

    def compute_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: float = 0.07
    ) -> (torch.Tensor, Dict[str, float]):
        # align dtypes to avoid precision mismatch in projection head
        wdtype = self.vision_projection.weight.dtype
        img_f = image_features.to(wdtype)
        txt_f = text_features.to(wdtype)

        # normalize before similarity
        img_f = F.normalize(img_f, dim=-1)
        txt_f = F.normalize(txt_f, dim=-1)

        # project and renormalize vision features
        proj = self.vision_projection(img_f)
        proj = F.normalize(proj, dim=-1)

        labels = torch.arange(len(img_f), device=img_f.device)

        # generalized zero‐shot loss (image ↔ text)
        logits_gzsl = (img_f @ txt_f.T) / temperature
        gzsl_loss = F.cross_entropy(logits_gzsl, labels)

        # vision‐head loss (projected image ↔ text)
        logits_proj = (proj @ txt_f.T) / temperature
        vision_loss = F.cross_entropy(logits_proj, labels)

        total = 0.5 * (gzsl_loss + vision_loss)
        return total, {
            "gzsl_loss": gzsl_loss.item(),
            "vision_loss": vision_loss.item(),
            "total_loss": total.item()
        }

    def forward(self, images: torch.Tensor, texts: List[str]):
        img_feats = self.encode_image(images)
        txt_feats = self.encode_text(texts)
        # cast features to projection head dtype for consistency downstream
        wdtype = self.vision_projection.weight.dtype
        return img_feats.to(wdtype), txt_feats.to(wdtype)

    def predict_location(
        self,
        image: torch.Tensor,
        candidate_locations: List[Dict],
        hierarchical: bool = True
    ) -> Dict:
        self.eval()
        with torch.no_grad():
            img_f = self.encode_image(image)
            # normalization is critical for cosine-similarity ranking
            img_f = F.normalize(img_f, dim=-1)

            if hierarchical:
                # first predict country
                countries = list({loc["country"] for loc in candidate_locations})
                caps = [f"A photo in {c}." for c in countries]
                txt_f = self.encode_text(caps)
                txt_f = F.normalize(txt_f, dim=-1)
                idx = (img_f @ txt_f.T).squeeze().argmax().item()
                pred_country = countries[idx]

                # then predict city within that country
                city_locs = [loc for loc in candidate_locations if loc["country"] == pred_country]
                caps = [f"A photo from {loc['city']}." for loc in city_locs]
                txt_f = self.encode_text(caps)
                txt_f = F.normalize(txt_f, dim=-1)
                idx = (img_f @ txt_f.T).squeeze().argmax().item()
                return city_locs[idx]

            # flat prediction over all cities
            caps = [loc["city"] for loc in candidate_locations]
            captions = [f"A photo from {c}." for c in caps]
            txt_f = self.encode_text(captions)
            txt_f = F.normalize(txt_f, dim=-1)
            idx = (img_f @ txt_f.T).squeeze().argmax().item()
            return candidate_locations[idx]
