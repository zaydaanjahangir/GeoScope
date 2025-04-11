import os
import torch
import pandas as pd
from PIL import Image
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from cnn import *      # Assume this file now defines a CNN with in_channels=3 for RGB images
from tqdm import tqdm  # for progress bar in terminal

'''
In this file you will write end-to-end code to train a convolutional neural network to categorize street view images.
The images are contained in a directory called Streetview_Image_Dataset and their labels (continents) are provided 
in the CSV file "coordinates_with_continents_mapbox.csv". The classification for each image is found in the column 
named 'continent'. The images are not pre-split, so we will perform a random train-test split.
'''

# Create device object: use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Custom dataset for the Streetview images using the CSV file for labels
class StreetviewDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        # Convert all values in the 'continent' column to strings
        self.data['continent'] = self.data['continent'].astype(str)
        self.img_dir = img_dir
        self.transform = transform

        # Build a mapping from continent names to numeric labels
        continents = sorted(self.data['continent'].unique())
        self.label_map = {label: idx for idx, label in enumerate(continents)}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Construct image path assuming images are named "0.png", "1.png", etc.
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Get the label from the 'continent' column
        label_str = self.data.iloc[idx]['continent']
        label = self.label_map[label_str]
        return image, label

'''
PART 1:
Preprocess the street view dataset images. The transforms are updated for RGB images.
'''

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize each channel
])

batch_size = 16

'''
PART 2:
Load the dataset. Images are in the directory "Streetview_Image_Dataset" and the labels come from the CSV.
We perform a train/test split (80/20 split).
'''

dataset = StreetviewDataset(csv_file="data/coordinates_with_continents_mapbox.csv", 
                              img_dir="Streetview_Image_Dataset", 
                              transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

'''
PART 3:
Complete the model defintion in cnn.py. We instantiate the CNN model below.
Assume the CNN has been updated to accept RGB images (in_channels=3).
'''

conv_net = Conv_Net()
conv_net.to(device)  # Move the model to the appropriate device

'''
PART 4:
Choose a good loss function and optimizer.
'''

criterion = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001, weight_decay=1e-5)

'''
PART 5:
Train your model.
'''

cnn_train_losses = []
num_epochs_cnn = 15

for epoch in range(num_epochs_cnn):
    running_loss_cnn = 0.0
    conv_net.train()
    
    # Wrap the DataLoader iterator with tqdm to show batch progress
    for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs_cnn}"), 0):
        inputs, labels = data
        # Move inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer_cnn.zero_grad()
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()
    
    epoch_loss = running_loss_cnn / len(trainloader)
    cnn_train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs_cnn}, CNN Training loss: {epoch_loss:.4f}')

print('Finished Training CNN')

torch.save(conv_net.state_dict(), 'cnn.pth')  # Save model file (upload with submission)

'''
PART 6:
Evaluate your model. Accuracy should be greater or equal to 80%.
'''

correct_cnn = 0
total_cnn = 0

conv_net.eval()

with torch.no_grad():
    for data in tqdm(testloader, desc="Evaluating"):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs_cnn = conv_net(images)
        _, predicted_cnn = torch.max(outputs_cnn, 1)
        total_cnn += labels.size(0)
        correct_cnn += (predicted_cnn == labels).sum().item()

print('Accuracy for convolutional network: ', correct_cnn / total_cnn)

'''
PART 7:
Generate a plot of the training loss.
'''

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs_cnn + 1), cnn_train_losses, label='Convolutional Network')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Convolutional Network Training Loss')
plt.legend()
plt.savefig('cnn_training_loss.png')
plt.close()
