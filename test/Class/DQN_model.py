import torch
import torch.nn as nn
import numpy as np
from .Additional_model import BottleneckAttentionModule
import os
from PIL import Image
import matplotlib.pyplot as plt

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
        x = x.view(x.size(0), -1)
        x = self.l2(x)
        print(x)
        return x
    
    def forward_and_save(self, x, writer):
        fm1 = self.l1[0](x)
        att1 = self.l1[2](self.l1[1](fm1))
        fm2 = self.l1[3](att1)
        att2 = self.l1[5](self.l1[4](fm2))
        fm3 = self.l1[6](att2)

        # Save feature maps and attention maps as images
        plt.figure(figsize=(12, 6))

        # Save feature map 1
        for i in range(fm1.size(1)):
            plt.subplot(2, fm1.size(1), i + 1)
            plt.imshow(fm1[0, i].detach().cpu().numpy())
            plt.axis('off')

            # Convert the plot to a tensor
            plt.gcf().canvas.draw()
            buffer = plt.gcf().canvas.tostring_rgb()
            ncols, nrows = plt.gcf().canvas.get_width_height()
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(nrows, ncols, 3)
            # image_array_copy = np.copy(image_array)
            image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))

            # Write the tensor as an image to TensorBoard
            writer.add_image(f'Feature Map 1/{i}', image_tensor, dataformats='CHW')

        # Save attention map 1
        for i in range(att1.size(1)):
            plt.subplot(2, att1.size(1), i + 1)
            plt.imshow(att1[0, i].detach().cpu().numpy(), cmap='gray')
            plt.axis('off')

            # Convert the plot to a tensor
            plt.gcf().canvas.draw()
            buffer = plt.gcf().canvas.tostring_rgb()
            ncols, nrows = plt.gcf().canvas.get_width_height()
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(nrows, ncols, 3)
            image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))

            # Write the tensor as an image to TensorBoard
            writer.add_image(f'Attention Map 1/{i}', image_tensor, dataformats='CHW')

        # Save feature map 2
        for i in range(fm2.size(1)):
            plt.subplot(2, fm2.size(1), i + 1)
            plt.imshow(fm2[0, i].detach().cpu().numpy())
            plt.axis('off')

            # Convert the plot to a tensor
            plt.gcf().canvas.draw()
            buffer = plt.gcf().canvas.tostring_rgb()
            ncols, nrows = plt.gcf().canvas.get_width_height()
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(nrows, ncols, 3)
            image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))

            # Write the tensor as an image to TensorBoard
            writer.add_image(f'Feature Map 2/{i}', image_tensor, dataformats='CHW')

        # Save attention map 2
        for i in range(att2.size(1)):
            plt.subplot(2, att2.size(1), i + 1)
            plt.imshow(att2[0, i].detach().cpu().numpy(), cmap='gray')
            plt.axis('off')

            # Convert the plot to a tensor
            plt.gcf().canvas.draw()
            buffer = plt.gcf().canvas.tostring_rgb()
            ncols, nrows = plt.gcf().canvas.get_width_height()
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(nrows, ncols, 3)
            image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))

            # Write the tensor as an image to TensorBoard
            writer.add_image(f'Attention Map 2/{i}', image_tensor, dataformats='CHW')

        # Save feature map 3
        for i in range(fm3.size(1)):
            plt.subplot(2, fm3.size(1), i + 1)
            plt.imshow(fm3[0, i].detach().cpu().numpy())
            plt.axis('off')

            # Convert the plot to a tensor
            plt.gcf().canvas.draw()
            buffer = plt.gcf().canvas.tostring_rgb()
            ncols, nrows = plt.gcf().canvas.get_width_height()
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(nrows, ncols, 3)
            image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))

            # Write the tensor as an image to TensorBoard
            writer.add_image(f'Feature Map 3/{i}', image_tensor, dataformats='CHW')

        writer.flush()


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