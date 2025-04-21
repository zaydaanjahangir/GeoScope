import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.synthetic_caption import SyntheticCaptionGenerator
from utils.plotting import plot_evaluation_metrics


class StreetCLIPEvaluator:
    def __init__(self, model: nn.Module, device: torch.device, output_dir: str):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Log both to file and console with timestamps
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "evaluation.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        test_dataloader: DataLoader,
        candidate_locations: List[Dict]
    ) -> Dict:
        self.model.eval()
        total_samples = correct_continent = correct_country = correct_city = 0
        pbar = tqdm(test_dataloader, desc="Evaluating")

        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                meta = batch["metadata"]
                batch_size = images.size(0)

                # Reassemble per-sample metadata from collated lists
                individual_meta = [
                    {k: meta[k][i] for k in meta}
                    for i in range(batch_size)
                ]

                for img, md in zip(images, individual_meta):
                    pred = self.model.predict_location(img.unsqueeze(0), candidate_locations)
                    total_samples += 1

                    if pred.get("continent") == md.get("continent"):
                        correct_continent += 1
                    if pred.get("country") == md.get("country"):
                        correct_country += 1
                    if pred.get("city") == md.get("city"):
                        correct_city += 1

        # Avoid division by zero
        continent_acc = correct_continent / total_samples if total_samples else 0
        country_acc   = correct_country   / total_samples if total_samples else 0
        city_acc      = correct_city      / total_samples if total_samples else 0

        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Continent accuracy: {continent_acc:.4f}")
        self.logger.info(f"Country accuracy:   {country_acc:.4f}")
        self.logger.info(f"City accuracy:      {city_acc:.4f}")

        metrics = {
            "total_samples": total_samples,
            "continent_accuracy": continent_acc,
            "country_accuracy": country_acc,
            "city_accuracy": city_acc
        }
        # Persist metrics and generate plots
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
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
        self.model.eval()
        total = correct_cont = correct_cnty = correct_city = 0
        pbar = tqdm(test_dataloader, desc="Hierarchical Evaluating")

        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                meta = batch["metadata"]
                bs = images.size(0)
                indiv = [{k: meta[k][i] for k in meta} for i in range(bs)]

                for img, md in zip(images, indiv):
                    # Encode and normalize once for all levels
                    feats = self.model.encode_image(img.unsqueeze(0))
                    feats = feats / feats.norm(dim=-1, keepdim=True)  # for cosine similarity

                    pred_cont = None
                    if continent_candidates:
                        caps = [f"A photo from the continent of {c}" for c in continent_candidates]
                        txt_feats = self.model.encode_text(caps)
                        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                        idx = (feats @ txt_feats.T).argmax().item()
                        pred_cont = continent_candidates[idx]
                        if pred_cont == md["continent"]:
                            correct_cont += 1

                    # Restrict countries if continent was predicted
                    cnty_list = (
                        continent_countries.get(pred_cont, country_candidates)
                        if pred_cont and continent_countries else country_candidates
                    )
                    caps = [SyntheticCaptionGenerator.generate_country_caption(c) for c in cnty_list]
                    txt_feats = self.model.encode_text(caps)
                    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                    idx = (feats @ txt_feats.T).argmax().item()
                    pred_cnty = cnty_list[idx]
                    if pred_cnty == md["country"]:
                        correct_cnty += 1

                        if pred_cnty in city_candidates:
                            cities = city_candidates[pred_cnty]
                            caps = [SyntheticCaptionGenerator.generate_city_caption(c) for c in cities]
                            txt_feats = self.model.encode_text(caps)
                            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                            idx = (feats @ txt_feats.T).argmax().item()
                            if cities[idx] == md["city"]:
                                correct_city += 1

                    total += 1

        # Compute metrics
        cont_acc = correct_cont / total if continent_candidates and total else None
        cnty_acc = correct_cnty / total if total else 0
        city_acc = correct_city / total if total else 0

        self.logger.info(f"Total samples: {total}")
        if cont_acc is not None:
            self.logger.info(f"Continent accuracy: {cont_acc:.4f}")
        self.logger.info(f"Country accuracy:   {cnty_acc:.4f}")
        self.logger.info(f"City accuracy:      {city_acc:.4f}")

        results = {
            "total_samples": total,
            "country_accuracy": cnty_acc,
            "city_accuracy": city_acc
        }
        if cont_acc is not None:
            results["continent_accuracy"] = cont_acc

        with open(os.path.join(self.output_dir, "hierarchical_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)
        plot_evaluation_metrics(results, self.output_dir)

        return results
