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
from cnn import *  
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class StreetviewDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data['continent'] = self.data['continent'].astype(str)
        self.img_dir = img_dir
        self.transform = transform
        continents = sorted(self.data['continent'].unique())
        self.label_map = {label: idx for idx, label in enumerate(continents)}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Get the label from the 'continent' column
        label_str = self.data.iloc[idx]['continent']
        label = self.label_map[label_str]
        return image, label

# Data preprocessing: transforms for RGB images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize each channel
])

batch_size = 16


# Split dataset into train, validation, and test 70/10/20 split
dataset = StreetviewDataset(csv_file="data/coordinates_with_continents_mapbox.csv", 
                              img_dir="Streetview_Image_Dataset", 
                              transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Instantiate CNN model 
conv_net = Conv_Net()
conv_net.to(device)

# Loss function and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001, weight_decay=1e-5)


cnn_train_losses = []
cnn_val_losses = []
cnn_val_accs = []  # List to record validation accuracy per epoch
best_val_acc = 0.0  # To track the best validation accuracy
best_epoch = 0      # To track which epoch had the best accuracy
num_epochs_cnn = 15

# Training loop with live validation evaluation
for epoch in range(num_epochs_cnn):
    running_loss_cnn = 0.0
    conv_net.train()
    
    for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs_cnn}"), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer_cnn.zero_grad()
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()
    
    epoch_loss = running_loss_cnn / len(trainloader)
    cnn_train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs_cnn}, CNN Training Loss: {epoch_loss:.4f}')
    
    # Evaluate on validation set
    conv_net.eval()
    val_loss = 0.0
    total_val = 0
    correct_val = 0
    with torch.no_grad():
        for data in tqdm(valloader, desc="Validation"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = conv_net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_loss /= len(valloader)
    val_acc = correct_val / total_val
    cnn_val_losses.append(val_loss)
    cnn_val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs_cnn}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    
    # Checkpoint if current validation accuracy is the best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(conv_net.state_dict(), 'best_cnn.pth')
        print(f"Checkpoint: Saving best model at epoch {epoch+1} with validation accuracy {val_acc:.4f}")

print('Finished Training CNN')

torch.save(conv_net.state_dict(), 'cnn.pth') 

# Evaluate accuracy on the test set after training
correct_cnn = 0
total_cnn = 0
conv_net.eval()
with torch.no_grad():
    for data in tqdm(testloader, desc="Evaluating Test Set"):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs_cnn = conv_net(images)
        _, predicted_cnn = torch.max(outputs_cnn, 1)
        total_cnn += labels.size(0)
        correct_cnn += (predicted_cnn == labels).sum().item()

print('Accuracy for CNN on Test Set: ', correct_cnn / total_cnn)


# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs_cnn + 1), cnn_train_losses, label='Training Loss')
plt.plot(range(1, num_epochs_cnn + 1), cnn_val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training and Validation Loss')
plt.legend()
plt.savefig('cnn_loss.png')
plt.close()

# Plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs_cnn + 1), cnn_val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Validation Accuracy')
plt.legend()
plt.savefig('cnn_accuracy.png')
plt.close()