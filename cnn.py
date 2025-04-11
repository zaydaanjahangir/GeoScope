import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.

'''


class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: [batch_size, 3, 640, 640]
        
        # Output: [batch_size, 32, 640, 640]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Output: [batch_size, 32, 320, 320]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output: [batch_size, 64, 320, 320]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Output: [batch_size, 64, 160, 160]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output: [batch_size, 128, 160, 160]
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Output: [batch_size, 128, 80, 80]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output: [batch_size, 256, 80, 80]
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # Output: [batch_size, 256, 40, 40]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output: [batch_size, 512, 40, 40]
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)        
        # Output: [batch_size, 512, 20, 20]
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output: [batch_size, 512, 20, 20]
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # Output: [batch_size, 512, 10, 10]
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 7)
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        # Block 5
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        # Block 6
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        
        # Flatten: now shape is [batch_size, 1024, 10, 10]
        x = x.view(-1, 1024 * 10 * 10)
        
        # Fully Connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x





        
