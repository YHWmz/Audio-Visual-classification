# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:05:18 2021

@author: 10513
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class Basenet(nn.Module):
    def __init__(self):
        super(Basenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.ac1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.ac2 = nn.ReLU()
        self.max1 = nn.MaxPool2d(5,stride = 5)
        self.dropou1 = torch.nn.Dropout(p=0.3, inplace=False)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.ac3 = nn.ReLU()
        self.max2 = nn.MaxPool2d((100,4),stride = (100,4))
        self.dropou2 = torch.nn.Dropout(p=0.3, inplace=False)
        self.fc1 = nn.Linear(64, 100)
        self.ac4 = nn.ReLU()
        self.dropou3 = torch.nn.Dropout(p=0.3, inplace=False)
        self.fc2 = nn.Linear(100, 10)
        self.sm = nn.Softmax(1)
        
    def forward(self, data):
        output = self.ac1(self.bn1(self.conv1(data)))
        output = self.ac2(self.bn2(self.conv2(output)))
        output = self.dropou1(self.max1(output))
        output = self.ac3(self.bn3(self.conv3(output)))
        output = self.dropou2(self.max2(output))
        output = output.view(output.shape[0],-1)
        output = self.dropou3(self.ac4(self.fc1(output)))
        output = self.fc2(output)
        output = self.sm(output)
        return output

