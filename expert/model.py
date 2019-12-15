import torch
import torch.nn as nn


class Debug(nn.Module):
    def forward(self, x):
        print(x.shape)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_features)
        self.mpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.mpool(x)
        return x
        
    
class ExpertModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ExpertModel, self).__init__()
        self.debug = Debug()
        self.block1 = BasicBlock(in_features, 64)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 128)
        self.block4 = BasicBlock(128, 256)
        self.block5 = BasicBlock(256, 256, kernel_size=1)
        self.dp = nn.Dropout2d()
        self.fc = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.block1(x)
        # self.debug(x)
        x = self.block2(x)
        # self.debug(x)
        x = self.block3(x)
        # self.debug(x)
        x = self.block4(x)
        # self.debug(x)
        x = self.block5(x)
        # self.debug(x)
        x = self.dp(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        # self.debug(x)
        return x
