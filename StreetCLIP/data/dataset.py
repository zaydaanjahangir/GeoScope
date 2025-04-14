import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from torchvision import transforms
from tqdm import tqdm

class StreetViewDataset(Dataset):
    """Dataset for StreetCLIP geolocalization."""
    
    def __init__(self, csv_path: str, image_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV with location data
            image_dir: Directory containing the street view images
            transform: Optional transforms to be applied to images
        """
        print("Loading dataset...")
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),  # CLIP expects 224x224 images
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)
                )
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing:
                - 'image': Image tensor
                - 'metadata': Dictionary with location metadata
        """
        # Generate image filename based on index
        image_filename = f"{idx}.png"
        
        # Load image
        image_path = os.path.join(self.image_dir, image_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise

        if self.transform:
            image = self.transform(image)

        # Get location metadata
        row = self.data.iloc[idx]
        location_metadata = {
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'city': str(row['city']),
            'region': str(row['region']),
            'country': str(row['country']),
            'continent': str(row['continent'])
        }

        # Return as a dictionary with metadata directly
        return {
            'image': image,
            'metadata': location_metadata
        }

def custom_collate_fn(batch):
    """
    Custom collate function to properly handle mixed types (tensors and metadata).
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Collated batch with properly handled metadata
    """
    # Extract images
    images = torch.stack([item['image'] for item in batch])
    
    # Collect metadata dictionaries separately to avoid PyTorch's default collation
    metadata = {}
    for key in batch[0]['metadata'].keys():
        # For each metadata field, collect values from all items in the batch
        metadata[key] = [item['metadata'][key] for item in batch]
    
    return {
        'image': images,
        'metadata': metadata
    }

def create_dataloader(
    csv_path: str,
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the StreetViewDataset.
    
    Args:
        csv_path: Path to CSV with location data
        image_dir: Directory containing the street view images
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the dataset
    """
    dataset = StreetViewDataset(csv_path, image_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn  # Use custom collate function
    )

def create_train_test_split(
    csv_path: str,
    image_dir: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders from a single CSV file.
    """
    print("Creating train-test split...")
    # Read the full dataset
    full_data = pd.read_csv(csv_path)
    
    # Randomly shuffle and split the data
    shuffled_data = full_data.sample(frac=1, random_state=random_state)
    split_idx = int(len(shuffled_data) * (1 - test_size))
    
    # Create train and test DataFrames
    train_data = shuffled_data.iloc[:split_idx]
    test_data = shuffled_data.iloc[split_idx:]
    
    # Save split datasets to temporary CSV files
    train_csv = csv_path.replace('.csv', '_train_temp.csv')
    test_csv = csv_path.replace('.csv', '_test_temp.csv')
    
    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)
    
    print(f"Split dataset into {len(train_data)} training and {len(test_data)} test samples")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_csv,
        image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    test_dataloader = create_dataloader(
        test_csv,
        image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    # Clean up temporary files
    os.remove(train_csv)
    os.remove(test_csv)
    
    return train_dataloader, test_dataloader