import torch
import torch.nn as nn
import torch.nn.functional as F

class BAM(nn.Module):
    def __init__(self, channels):
        super(BAM, self).__init__()
        
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)
        
        self.sigmoid_channel = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()
        
        self.fc3 = nn.Linear(channels, 7)
    
    def forward(self, x):
        identity = x
        
        # Channel attention
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid_channel(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * x
        
        # Spatial attention
        avg = self.avg_pool(x)
        max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        out_spatial = torch.cat([avg, max], dim=1)
        out_spatial = self.conv_spatial(out_spatial)
        out_spatial = self.sigmoid_spatial(out_spatial)
        out = out * out_spatial
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        # Add a fully connected layer to output 0-6
        out = self.fc3(out.view(out.size(0), -1))
        
        return out