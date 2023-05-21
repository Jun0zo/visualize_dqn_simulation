import torch
import torch.nn as nn
import numpy as np
from .Additional_model import BottleneckAttentionModule
import os
from utils import get_highest_number
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# This class defines the DQN network structure
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, filename, model_path):
        super(DQN, self).__init__()

        self.model_path = model_path

        self.input_dim = input_dim
        channels, _, _ = input_dim

        # 3 conv layers, all with relu activations, first one with maxpool
        self.l1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            BottleneckAttentionModule(32),  # BAM added after the first convolutional layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            BottleneckAttentionModule(64),  # BAM added after the second convolutional layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculate output dimensions for linear layer
        conv_output_size = self.conv_output_dim()
        lin1_output_size = 512

        # Two fully connected layers with one relu activation
        self.l2 = nn.Sequential(
            nn.Linear(conv_output_size, lin1_output_size),
            nn.ReLU(),
            nn.Linear(lin1_output_size, output_dim)
        )

        self.feature_maps = None    # List to store feature maps
        self.attention_maps = None    # List to store attention maps

        self.l1[0].register_forward_hook(self.capture_feature_maps)
        self.l1[2].channel_attention.register_forward_hook(self.capture_attention_maps)
        self.l1[2].spatial_attention.register_forward_hook(self.capture_attention_maps)
        self.l1[3].register_forward_hook(self.capture_feature_maps)
        self.l1[5].channel_attention.register_forward_hook(self.capture_attention_maps)
        self.l1[5].spatial_attention.register_forward_hook(self.capture_attention_maps)
        self.l1[6].register_forward_hook(self.capture_feature_maps)


        # Save filename for saving model
        self.filename = filename

    # Calulates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))

    # Performs forward pass through the network, returns action values
    def forward(self, x):
        self.feature_maps = []  # Clear feature_maps list for each forward pass
        self.attention_maps = []  # Clear attention_maps list for each forward pass
        
        x = self.l1(x)
        x = x.view(x.size(0), -1)
        x = self.l2(x)
        print(x)
        return x
    
    def capture_feature_maps(self, module, input, output):
        self.feature_maps.append(output)

    def capture_attention_maps(self, module, input, output):
        self.attention_maps.append(output)
    
    def save_maps_as_images(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Save feature maps as images
        for i, feature_map in enumerate(self.feature_maps):
            feature_map = feature_map.squeeze(0).detach().cpu()
            image = TF.to_pil_image(feature_map)
            image_path = os.path.join(output_dir, f"feature_map_{i}.png")
            image.save(image_path)

        # Save attention maps as images
        for i, attention_map in enumerate(self.attention_maps):
            attention_map = attention_map.squeeze(0).detach().cpu()
            image = TF.to_pil_image(attention_map)
            image_path = os.path.join(output_dir, f"attention_map_{i}.png")
            image.save(image_path)

    # Save a model
    def save_model(self, file_idx=0):
        if not file_idx:
            file_nubmer = get_highest_number(self.model_path) + 1
            torch.save(self.state_dict(), os.path.join(self.model_path, f'{file_nubmer}.pth'))
        else:
            torch.save(self.state_dict(), os.path.join(self.model_path, f'{file_idx}.pth'))
        print(f"model loaded! : f'{file_nubmer}.pth")


    # Loads a model
    def load_model(self, file_idx=0):
        if not file_idx:
            file_nubmer = get_highest_number(self.model_path)
            model_path = os.path.join(self.model_path, f'{file_nubmer}.pth')
        else:
            model_path = os.path.join(self.model_path, f'{file_idx}.pth')
        self.load_state_dict(torch.load(model_path))
        print(f'model loaded! : {model_path}')