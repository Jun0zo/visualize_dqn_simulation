import torch
import torch.nn as nn
import numpy as np
import os

# This class defines the DQN network structure
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, filename, pretrained_model_path='./models', save_model_path='./models'):
        super(DQN, self).__init__()

        self.pretrained_model_path = pretrained_model_path
        self.save_model_path = save_model_path

        self.input_dim = input_dim
        channels, _, _ = input_dim

        # 3 conv layers, all with relu activations, first one with maxpool
        self.l1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
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

        # Save filename for saving model
        self.filename = filename

    # Calulates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))

    # Performs forward pass through the network, returns action values
    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], -1)
        actions = self.l2(x)
        # print('actions:', actions.shape)
        # print('action : ', actions.argmax().item())

        return actions
    
    # Performs forward pass through the network, returns action values and feature maps
    def forward2(self, x):
        feature_maps = []
        
        x = self.l1[0](x)
        feature_maps.append(x)
        x = self.l1[1](x)
        x = self.l1[2](x)
        feature_maps.append(x)
        x = self.l1[3](x)
        feature_maps.append(x)
        
        x = x.view(x.shape[0], -1)
        actions = self.l2(x)

        return actions, feature_maps

    # Save a model
    def save_model(self, is_idx=True):
        if is_idx:
            num_files = len(os.listdir(self.save_model_path))
            torch.save(self.state_dict(), os.path.join(self.save_model_path, f'{num_files + 1}.pth'))
        else:
            torch.save(self.state_dict(), os.path.join(self.save_model_path, f'{self.filename}.pth'))


    # Loads a model
    def load_model(self, is_idx=True):
        if is_idx:
            num_files = len(os.listdir(self.pretrained_model_path))
            model_path = os.path.join(self.pretrained_model_path, f'{num_files}.pth')
        else:
            model_path = os.path.join(self.pretrained_model_path, f'{self.filename}.pth')
        self.load_state_dict(torch.load(model_path))
        print(f'model loaded! : {model_path}')