import os
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from cnn import Conv_Net
from tqdm import tqdm


class StreetviewDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data['continent'] = self.data['continent'].astype(str)
        self.img_dir = img_dir
        self.transform = transform

        # Builds a mapping from continent names to numeric labels
        continents = sorted(self.data['continent'].unique())
        self.label_map = {label: idx for idx, label in enumerate(continents)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_str = self.data.iloc[idx]['continent']
        label = self.label_map[label_str]
        return image, label


transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize each channel
])

batch_size = 16

dataset = StreetviewDataset(csv_file="data/coordinates_with_continents_mapbox.csv", 
                              img_dir="Streetview_Image_Dataset", 
                              transform=transform)


testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))
conv_net.eval() 

# Evaluate the model
correct = 0
total = 0

with torch.no_grad(): 
    for images, labels in tqdm(testloader, desc="Evaluating"):
        outputs = conv_net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy: ", correct / total)
