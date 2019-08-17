import torch
import torch.nn as nn
from model import common


class FASTARCNN(nn.Module):
    def __init__(self, args=None):
        super(FASTARCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,48,8,stride=2,padding=4,bias=False) #deblock for Y only, here use 48 filters only
        self.relu1= nn.PReLU()
        self.conv2 = nn.Conv2d(48,32,1,stride=1,padding=0,bias=False)
        self.relu2= nn.PReLU()
        self.conv3 = nn.Conv2d(32,32,7,stride=1,padding=3,bias=False)
        self.relu3= nn.PReLU()
        self.conv4 = nn.Conv2d(32,48,1,stride=1,padding=0,bias=False)
        self.relu4= nn.PReLU()
        self.deconv = nn.ConvTranspose2d(48,3,8,stride=2,padding=4,bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.deconv(x)
        return x

