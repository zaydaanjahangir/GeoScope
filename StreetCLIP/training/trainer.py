import os
import json
import random
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.synthetic_caption import SyntheticCaptionGenerator
from utils.plotting import plot_training_metrics


class StreetCLIPTrainer:
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
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_epochs = warmup_epochs

        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Try to enable gradient checkpointing to save memory during backprop
        try:
            self.model.clip_model.visual.gradient_checkpointing_enable()
            self.logger.info("Enabled gradient checkpointing on CLIP visual transformer.")
        except AttributeError:
            self.logger.warning("No gradient_checkpointing_enable() available; skipping.")

        # Mixed‐precision scaler
        self.scaler = GradScaler()

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = total_gzsl = total_vision = 0.0
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
        self.optimizer.zero_grad()

        # number of steps over which to warm up the learning rate
        warmup_steps = max(1, int(self.warmup_epochs * num_batches))
        eps = 1e-6           # avoid zero‑division in norms
        clamp_val = 20.0     # clamp logits to stabilize loss
        temperature = 0.07

        for idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            meta = batch["metadata"]
            bs = images.size(0)

            # reconstruct per-sample metadata dicts
            samples = [
                {k: meta[k][i] for k in meta}
                for i in range(bs)
            ]
            captions = [
                SyntheticCaptionGenerator.generate_caption(md)
                for md in samples
            ]

            # FP16 forward for speed/memory
            with autocast("cuda"):
                img_f, txt_f = self.model(images, captions)

            # cast back to FP32 for stable loss computation
            img_f = img_f.float()
            txt_f = txt_f.float()

            # normalize features (clamped to avoid zeros)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True).clamp_min(eps)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True).clamp_min(eps)

            # project vision features and normalize
            proj = self.model.vision_projection(img_f)
            proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(eps)

            labels = torch.arange(bs, device=self.device)

            # GZSL loss on raw CLIP features
            logits_gzsl = (img_f @ txt_f.T) / temperature
            logits_gzsl = logits_gzsl.clamp(-clamp_val, clamp_val)
            loss_gzsl = F.cross_entropy(logits_gzsl, labels)

            # Loss on the projection head
            logits_proj = (proj @ txt_f.T) / temperature
            logits_proj = logits_proj.clamp(-clamp_val, clamp_val)
            loss_proj = F.cross_entropy(logits_proj, labels)

            loss = 0.5 * (loss_gzsl + loss_proj)
            loss = loss / self.gradient_accumulation_steps

            # scaled backward for mixed precision and accumulation
            self.scaler.scale(loss).backward()

            # step optimizer & update scaler after accumulation
            if (idx + 1) % self.gradient_accumulation_steps == 0:
                # clip gradients to avoid exploding
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # linear warmup of learning rate
                step_num = (epoch - 1) * num_batches + (idx + 1)
                if step_num <= warmup_steps:
                    warm_scale = step_num / warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate * warm_scale

                # try AMP step, fallback to regular step if error
                try:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except ValueError:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # accumulate stats for logging
            total_loss  += loss.item() * self.gradient_accumulation_steps
            total_gzsl  += loss_gzsl.item()
            total_vision += loss_proj.item()

            pbar.set_postfix({
                "loss":        total_loss  / (idx + 1),
                "gzsl_loss":   total_gzsl  / (idx + 1),
                "vision_loss": total_vision/ (idx + 1),
            })

        return {
            "loss":        total_loss   / num_batches,
            "gzsl_loss":   total_gzsl   / num_batches,
            "vision_loss": total_vision / num_batches
        }

    def generate_candidate_locations(
        self,
        dataloader: DataLoader,
        max_candidates: int = 100
    ) -> List[Dict]:
        self.logger.info("Collecting unique candidate locations...")
        seen = set()
        locations = []

        for batch in tqdm(dataloader, desc="Collecting locations"):
            meta = batch["metadata"]
            bs = len(next(iter(meta.values())))
            samples = [
                {k: meta[k][i] for k in meta}
                for i in range(bs)
            ]
            for md in samples:
                key = (md.get("city", ""), md.get("country", ""), md.get("continent", ""))
                if key not in seen:
                    seen.add(key)
                    locations.append({
                        "city":      md.get("city", ""),
                        "region":    md.get("region", ""),
                        "country":   md.get("country", ""),
                        "continent": md.get("continent", "")
                    })

        # random subset if too many
        if len(locations) > max_candidates:
            locations = random.sample(locations, max_candidates)
        self.logger.info(f"Generated {len(locations)} candidates")
        return locations

    def evaluate_geographic_accuracy(
        self,
        val_loader: DataLoader,
        candidates: List[Dict]
    ) -> Dict[str, float]:
        self.logger.info("Evaluating geographic accuracy...")
        self.model.eval()
        total = corr_city = corr_country = corr_cont = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Geo Eval"):
                images = batch["image"].to(self.device)
                meta = batch["metadata"]
                bs = images.size(0)
                samples = [
                    {k: meta[k][i] for k in meta}
                    for i in range(bs)
                ]

                for img, md in zip(images, samples):
                    # skip if metadata missing
                    if not (md.get("city") and md.get("country") and md.get("continent")):
                        continue

                    feats = self.model.encode_image(img.unsqueeze(0))
                    feats = feats / feats.norm(dim=-1, keepdim=True)

                    caps = [SyntheticCaptionGenerator.generate_caption(c) for c in candidates]
                    txt_feats = self.model.encode_text(caps)
                    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

                    sims = (feats @ txt_feats.T).squeeze()
                    pred = candidates[sims.argmax().item()]

                    total += 1
                    if pred["continent"] == md["continent"]:
                        corr_cont += 1
                    if pred["country"] == md["country"]:
                        corr_country += 1
                    if pred["city"] == md["city"]:
                        corr_city += 1

        if total == 0:
            self.logger.warning("No valid samples found for geographic accuracy.")
            return {"continent_accuracy": 0.0, "country_accuracy": 0.0, "city_accuracy": 0.0}

        return {
            "continent_accuracy": corr_cont  / total,
            "country_accuracy":   corr_country/ total,
            "city_accuracy":      corr_city   / total
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        geo_eval_frequency: int = 1
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [], "train_gzsl_loss": [], "train_vision_loss": [],
            "val_loss": [], "val_accuracy": [], "learning_rate": [],
            "continent_accuracy": [], "country_accuracy": [], "city_accuracy": []
        }
        best_val = float("inf")

        # prepare candidate locations
        candidates = self.generate_candidate_locations(val_loader or train_loader)

        for epoch in range(1, num_epochs + 1):
            lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(lr)

            train_m = self.train_epoch(train_loader, epoch, num_epochs)
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} – Loss: {train_m['loss']:.4f}, "
                f"GZSL: {train_m['gzsl_loss']:.4f}, Vision: {train_m['vision_loss']:.4f}"
            )
            history["train_loss"].append(train_m["loss"])
            history["train_gzsl_loss"].append(train_m["gzsl_loss"])
            history["train_vision_loss"].append(train_m["vision_loss"])

            if val_loader:
                val_m = self.validate(val_loader)
                history["val_loss"].append(val_m["loss"])
                history["val_accuracy"].append(val_m["accuracy"])
                self.logger.info(f"Validation – Loss: {val_m['loss']:.4f}, Acc: {val_m['accuracy']:.4f}")

                # save best model
                if val_m["loss"] < best_val:
                    best_val = val_m["loss"]
                    self.save_checkpoint(os.path.join(self.output_dir, "best_model.pt"), epoch, best_val)

                # geographic eval at intervals
                if epoch % geo_eval_frequency == 0 or epoch == num_epochs:
                    geo = self.evaluate_geographic_accuracy(val_loader, candidates)
                    for k in ["continent_accuracy", "country_accuracy", "city_accuracy"]:
                        history[k].append(geo[k])

            # checkpoint and save history/plots every epoch
            ckpt_loss = val_m["loss"] if val_loader else train_m["loss"]
            self.save_checkpoint(os.path.join(self.output_dir, f"epoch_{epoch}.pt"), epoch, ckpt_loss)

            with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=4)
            plot_training_metrics(history, self.output_dir)

        # final save
        self.save_checkpoint(os.path.join(self.output_dir, "final_model.pt"), num_epochs, best_val or train_m["loss"])
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)
        plot_training_metrics(history, self.output_dir)

        return history

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_gzsl = total_vision = 0.0
        correct = total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch["image"].to(self.device)
                meta = batch["metadata"]
                bs = images.size(0)
                samples = [
                    {k: meta[k][i] for k in meta}
                    for i in range(bs)
                ]
                captions = [SyntheticCaptionGenerator.generate_caption(md) for md in samples]

                img_f, txt_f = self.model(images, captions)
                loss, comps = self.model.compute_loss(img_f, txt_f)
                total_loss  += comps["total_loss"]
                total_gzsl  += comps["gzsl_loss"]
                total_vision+= comps["vision_loss"]

                # accuracy: highest similarity along diagonal
                img_n = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_n = txt_f / txt_f.norm(dim=-1, keepdim=True)
                sims = img_n @ txt_n.T
                preds = sims.argmax(dim=1)
                correct += (preds == torch.arange(len(preds), device=self.device)).sum().item()
                total   += len(preds)

        accuracy = correct / total if total else 0.0
        return {
            "loss":         total_loss   / len(val_loader),
            "gzsl_loss":    total_gzsl   / len(val_loader),
            "vision_loss":  total_vision / len(val_loader),
            "accuracy":     accuracy
        }

    def save_checkpoint(self, path: str, epoch: int, val_loss: float) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss
        }
        torch.save(ckpt, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.logger.info(f"Loaded checkpoint from {path}")
