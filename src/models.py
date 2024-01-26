import torch
import torch.nn as nn
from einops import rearrange, repeat

class Custom3DCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(Custom3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
 
        self.global_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(256, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 1)
        self.linear = nn.Linear(512, 1)

    def forward(self, x):
        x = repeat(x, 'batch slices width height -> batch channel slices width height', channel=1)
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.maxpool3(self.relu3(self.conv3(x)))

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.linear(x)

        return x

