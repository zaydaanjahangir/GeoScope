import os
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms


class StreetViewDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        print("Loading dataset...")
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        # CLIP normalization means and stds—these come directly from the original CLIP preprocessing
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        path = os.path.join(self.image_dir, f"{idx}.png")
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            raise
        img = self.transform(img) if self.transform else img

        row = self.data.iloc[idx]
        metadata = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "city": row["city"],
            "region": row["region"],
            "country": row["country"],
            "continent": row["continent"],
        }
        return {"image": img, "metadata": metadata}


def custom_collate_fn(batch):
    # Stack image tensors into shape (batch_size, 3, 224, 224)
    images = torch.stack([item["image"] for item in batch])

    metadata = {
        key: [item["metadata"][key] for item in batch]
        for key in batch[0]["metadata"]
    }
    return {"image": images, "metadata": metadata}


def create_dataloader(
    csv_path: str,
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    ds = StreetViewDataset(csv_path, image_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )


def create_train_test_split(
    csv_path: str,
    image_dir: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    print("Creating train-test split...")
    df = pd.read_csv(csv_path).sample(frac=1, random_state=random_state)
    split = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    # Write out temp CSVs for train/test and clean up immediately after loading
    train_csv = csv_path.replace(".csv", "_train_temp.csv")
    test_csv  = csv_path.replace(".csv", "_test_temp.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv,  index=False)
    print(f"Split into {len(train_df)} train and {len(test_df)} test samples")

    train_loader = create_dataloader(train_csv, image_dir, batch_size, num_workers, shuffle=True)
    test_loader  = create_dataloader(test_csv,  image_dir, batch_size, num_workers, shuffle=False)

    os.remove(train_csv)
    os.remove(test_csv)

    return train_loader, test_loader
