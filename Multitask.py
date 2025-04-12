import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import efficientnet_b3, resnet50
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
import kagglehub
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Constants
IMG_SIZE = (640, 640)
BATCH_SIZE = 16
EPOCHS = 1
NUM_CLASSES = 7
LEARNING_RATE = 0.0001
TEST_SPLIT = 0.1
plt.style.use('ggplot')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).view(-1, self.channels))
        max_out = self.mlp(self.max_pool(x).view(-1, self.channels))
        channel_out = self.sigmoid(avg_out + max_out).view(-1, self.channels, 1, 1)
        x = x * channel_out

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid(self.conv(concat))
        x = x * spatial_out

        return x


class GeoKnowr(nn.Module):
    def __init__(self, num_classes=7):
        super(GeoKnowr, self).__init__()

        self.effnet = efficientnet_b3(weights='DEFAULT')
        self.resnet = resnet50(weights='DEFAULT')

        self.effnet_features = nn.Sequential(*list(self.effnet.children())[:-2])
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-2])

        self.effnet_conv = nn.Conv2d(1536, 512, kernel_size=1)
        self.resnet_conv = nn.Conv2d(2048, 512, kernel_size=1)

        self.cbam1 = CBAM(512)
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(512)
        self.cbam2 = CBAM(512)

        self.continent_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.coord_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        eff_features = self.effnet_conv(self.effnet_features(x))
        res_features = self.resnet_conv(self.resnet_features(x))

        fused_features = eff_features + res_features

        x = self.cbam1(fused_features)
        x = self.conv(x)
        x = self.bn(x)
        x = self.cbam2(x)

        x = torch.mean(x, dim=[2, 3])

        continent_out = self.continent_head(x)
        coord_out = self.coord_head(x)

        return continent_out, coord_out


class GeoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        continent = self.df.iloc[idx]['continent_encoded']
        coords = self.df.iloc[idx][['lat_norm', 'lon_norm']].values.astype(np.float32)

        return image, (continent, coords)


def prepare_data(csv_path, image_dir):
    try:
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} entries in CSV")

        image_paths = []
        for i, row in df.iterrows():
            possible_paths = [
                os.path.join(image_dir, f"{i}.png"),
                os.path.join(image_dir, f"{i}.jpg"),
                os.path.join(image_dir, f"{row['latitude']:.6f}_{row['longitude']:.6f}.png"),
                os.path.join(image_dir, f"image_{i}.png")
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    image_paths.append(path)
                    break
            else:
                image_paths.append("")

        df['image_path'] = image_paths
        df['image_exists'] = df['image_path'].apply(lambda x: os.path.exists(x) if x else False)

        print(f"\nImage status:")
        print(f"- Found: {df['image_exists'].sum()}")
        print(f"- Missing: {len(df) - df['image_exists'].sum()}")

        if df['image_exists'].sum() == 0:
            try:
                sample_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))][:5]
                print(f"Sample files in directory: {sample_files}")
            except Exception as e:
                print(f"Could not list directory contents: {str(e)}")
            raise ValueError("No matching images found. Please check the image naming pattern.")

        df = df[df['image_exists']].copy()

        le = LabelEncoder()
        df['continent_encoded'] = le.fit_transform(df['continent'])

        scaler = StandardScaler()
        coords = df[['latitude', 'longitude']].values
        df[['lat_norm', 'lon_norm']] = scaler.fit_transform(coords)

        # Split into train/val/test
        train_df = df.sample(frac=0.7, random_state=42)  # 70% train
        val_df = df.drop(train_df.index).sample(frac=0.67, random_state=42)  # 20% val
        test_df = df.drop(train_df.index).drop(val_df.index)  # 10% test

        print("\nData split:")
        print(f"- Training samples: {len(train_df)}")
        print(f"- Validation samples: {len(val_df)}")
        print(f"- Test samples: {len(test_df)}")

        return train_df, val_df, test_df, le, scaler

    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        raise


def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def evaluate_test(model, test_loader, le):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_true = []

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.HuberLoss()

    with torch.no_grad():
        for images, (continents, coords) in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            continents = continents.long().to(DEVICE)
            coords = coords.to(DEVICE)

            continent_pred, coord_pred = model(images)
            loss_cls = criterion_cls(continent_pred, continents)
            loss_reg = criterion_reg(coord_pred, coords)
            test_loss += 0.6 * loss_cls.item() + 0.4 * loss_reg.item()

            _, predicted = torch.max(continent_pred.data, 1)
            total += continents.size(0)
            correct += (predicted == continents).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(continents.cpu().numpy())

    test_acc = 100 * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return test_acc


def train():
    try:
        print("Downloading dataset...")
        dataset_path = kagglehub.dataset_download("ayuseless/streetview-image-dataset")
        print(f"Dataset downloaded to: {dataset_path}")

        image_dir = None
        for root, dirs, files in os.walk(dataset_path):
            if any(f.lower().endswith(('.png', '.jpg')) for f in files):
                image_dir = root
                break

        if image_dir is None:
            print("\nDirectory structure:")
            os.system(f"tree -L 3 {dataset_path}")
            raise FileNotFoundError("Could not find any images in the downloaded dataset")

        print(f"\nFound images in: {image_dir}")

        csv_path = os.path.join("data", "coordinates_with_continents_mapbox.csv")
        if not os.path.exists(csv_path):
            csv_path = "coordinates_with_continents_mapbox.csv"

        print("\nVerifying paths:")
        print(f"CSV path: {os.path.abspath(csv_path)}")
        print(f"Image dir: {image_dir}")
        print(f"Sample image files: {os.listdir(image_dir)[:5]}")

        print("\nPreparing data...")
        train_df, val_df, test_df, le, scaler = prepare_data(csv_path, image_dir)

        train_dataset = GeoDataset(train_df, transform=get_transforms())
        val_dataset = GeoDataset(val_df, transform=get_transforms())
        test_dataset = GeoDataset(test_df, transform=get_transforms())

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        print("Initializing model...")
        model = GeoKnowr(num_classes=NUM_CLASSES).to(DEVICE)
        print(f"Using device: {DEVICE}")

        criterion_cls = nn.CrossEntropyLoss()
        criterion_reg = nn.HuberLoss()

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

        print("Starting training...")

        # Store metrics for plotting
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(EPOCHS):
            model.train()
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=False)
            train_loss = 0.0

            for images, (continents, coords) in train_pbar:
                images = images.to(DEVICE)
                continents = continents.long().to(DEVICE)
                coords = coords.to(DEVICE)

                optimizer.zero_grad()
                continent_pred, coord_pred = model(images)

                loss_cls = criterion_cls(continent_pred, continents)
                loss_reg = criterion_reg(coord_pred, coords)
                loss = 0.6 * loss_cls + 0.4 * loss_reg

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{train_loss / (train_pbar.n + 1):.4f}"
                })

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]", leave=False)
            with torch.no_grad():
                for images, (continents, coords) in val_pbar:
                    images = images.to(DEVICE)
                    continents = continents.long().to(DEVICE)
                    coords = coords.to(DEVICE)

                    continent_pred, coord_pred = model(images)
                    loss_cls = criterion_cls(continent_pred, continents)
                    loss_reg = criterion_reg(coord_pred, coords)
                    val_loss += 0.6 * loss_cls.item() + 0.4 * loss_reg.item()

                    _, predicted = torch.max(continent_pred.data, 1)
                    total += continents.size(0)
                    correct += (predicted == continents).sum().item()

                    val_pbar.set_postfix({
                        "Acc": f"{100 * correct / total:.2f}%",
                        "Val Loss": f"{val_loss / (val_pbar.n + 1):.4f}"
                    })

            # Store epoch metrics
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(100 * correct / total)

            print(f"\nEpoch {epoch + 1}/{EPOCHS} Summary:")
            print(f"Train Loss: {train_losses[-1]:.4f}")
            print(f"Val Loss: {val_losses[-1]:.4f}")
            print(f"Val Accuracy: {val_accuracies[-1]:.2f}%")
            print("-" * 50)

        # Plot training curves
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-o', label='Train Loss')
        plt.plot(val_losses, 'r-o', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, 'g-o', label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()

        # Test evaluation
        test_acc = evaluate_test(model, test_loader, le)

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'le_classes': le.classes_,
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'test_accuracy': test_acc,
            'training_metrics': {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_acc': val_accuracies
            }
        }, "geoknowr_model.pth")

        print("\nTraining complete!")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print("Visualizations saved:")
        print("- confusion_matrix.png")
        print("- training_curves.png")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise


if __name__ == "__main__":
    train()
