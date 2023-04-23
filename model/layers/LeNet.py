import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

'''
Credit and tutorial to https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/ for code and
descriptions
'''

class Flat(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LeNet(nn.Module):
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=480, out_features=500)
        self.relu3 = nn.ReLU()
		# initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        #self.softmax = nn.Softmax(dim=1)
    

    def forward(self, x):
        conv1_out = self.conv1(x)
        relu1_out = self.relu1(conv1_out)
        pool1_out = self.maxpool1(relu1_out)

        conv2_out = self.conv2(pool1_out)
        relu2_out = self.relu2(conv2_out)
        pool2_out = self.maxpool2(relu2_out)

        flat_out = self.flat(pool2_out)
        fc1_out = self.fc1(flat_out)
        relu3_out = self.relu3(fc1_out)

        out = self.fc2(relu3_out)
        #out = self.softmax(fc2_out)
        return out