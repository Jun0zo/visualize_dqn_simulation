import torch
import torch.nn as nn
import numpy as np
import os
from .Additional_model import BottleneckAttentionModule
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from ..utils import get_highest_number
from torchvision.transforms import ToPILImage


# This class defines the DQN network structure
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, filename, model_path, is_resiger=False):
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

        self.feature_maps = []    # List to store feature maps
        self.attention_maps = []    # List to store attention maps

        is_bam = True

        if is_resiger:
            if is_bam:
                self.l1[0].register_forward_hook(self.capture_feature_maps)
                self.l1[2].channel_attention.register_forward_hook(self.capture_attention_maps)
                self.l1[2].spatial_attention.register_forward_hook(self.capture_attention_maps)
                self.l1[3].register_forward_hook(self.capture_feature_maps)
                self.l1[5].channel_attention.register_forward_hook(self.capture_attention_maps)
                self.l1[5].spatial_attention.register_forward_hook(self.capture_attention_maps)
                self.l1[6].register_forward_hook(self.capture_feature_maps)
            else:
                self.l1[0].register_forward_hook(self.capture_feature_maps)
                self.l1[2].register_forward_hook(self.capture_feature_maps)
                self.l1[4].register_forward_hook(self.capture_feature_maps)


        # Save filename for saving model
        self.filename = filename

    # Calulates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))

    # Performs forward pass through the network, returns action values
    def forward(self, x):
        # self.feature_maps = []  # Clear feature_maps list for each forward pass
        # self.attention_maps = []  # Clear attention_maps list for each forward pass

        x = x.to('cuda:0')
        x = self.l1(x)
        x = x.view(x.size(0), -1)
        x = self.l2(x)
        return x
    
    def capture_feature_maps(self, module, input, output):
        self.feature_maps.append(output)

    def capture_attention_maps(self, module, input, output):
        self.attention_maps.append(output)
    
    def save_maps_as_images(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Save feature maps as images
        for ii, feature_map in enumerate(self.feature_maps):
            feature_map = feature_map.squeeze(0).detach().cpu()
            if feature_map.is_sparse:
                feature_map = feature_map.to_dense()
            if feature_map.dim() == 4:
                feature_map = feature_map[0]  # Remove the batch dimension

            print(feature_map.shape)
            
            num_cols = 8  # Number of columns in the subplot grid
            num_rows = feature_map.shape[0] // num_cols  # Number of rows in the subplot grid

            # Create a figure and axes for the subplots
            fig, axes = plt.subplots(num_rows, num_cols)
        
            for i in range(num_rows):
                for j in range(num_cols):
                    ffeature_map = feature_map[i * num_cols + j]  # Get the corresponding feature map
                    # Convert the feature map to a PIL image
                    image = TF.to_pil_image(ffeature_map)

                    # Plot the image in the corresponding subplot
                    ax = axes[i, j]
                    ax.imshow(image)
                    ax.axis('off')

            # Adjust the spacing between subplots
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            # Save the figure with subplots as a single image
            image_path = os.path.join(output_dir, f"combined_image_{ii}.png")
            plt.savefig(image_path)

            # Close the figure to free up memory
            plt.close(fig)

        # Save attention maps as images

    # Save a model
    def save_model(self, file_idx=0):
        if not file_idx:
            file_nubmer = get_highest_number(self.model_path) + 1
            torch.save(self.state_dict(), os.path.join(self.model_path, f'{file_nubmer}.pth'))
            print(f"model saved! : f'{file_nubmer}.pth")
        else:
            
            torch.save(self.state_dict(), os.path.join(self.model_path, f'{file_idx}.pth'))
            print(f"model saved! : f'{file_idx}.pth")
        


    # Loads a model
    def load_model(self, file_idx=0):
        print("load model!!")
        if not file_idx:
            file_nubmer = get_highest_number(self.model_path)
            model_path = os.path.join(self.model_path, f'{file_nubmer}.pth')
        else:
            model_path = os.path.join(self.model_path, f'{file_idx}.pth')
        
        self.load_state_dict(torch.load(model_path))
        print(f'model loaded! : {model_path}')