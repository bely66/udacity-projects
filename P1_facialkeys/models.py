## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,3)
        self.norm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256,512,1)
        self.norm5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2,2)
        self.drop_c = nn.Dropout(0.2)
        self.drop_fc = nn.Dropout(0.4)
        self.fc_1 = nn.Linear(512*6*6,1000)
        self.normfc1 = nn.BatchNorm1d(1000)
        self.fc_2 = nn.Linear(1000,1000)
        self.normfc2 = nn.BatchNorm1d(1000)
        self.fc_3 = nn.Linear(1000,136)
        
                               
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        x = self.pool(self.norm2(F.relu(self.conv2(x))))
        x = self.pool(self.norm3(F.relu(self.conv3(x))))
        x = (self.pool(self.norm4(F.relu(self.conv4(x)))))
        x = (self.pool(self.norm5(F.relu(self.conv5(x)))))
        
        x = x.view(x.size(0),-1)
        x = self.drop_c(self.normfc1(F.relu(self.fc_1(x))))
        x = self.drop_fc(self.normfc2(F.relu(self.fc_2(x))))
        x = self.fc_3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
