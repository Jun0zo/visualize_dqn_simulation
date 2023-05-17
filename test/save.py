import torch
import numpy as np
from PIL import Image
from Class.DQN_model import DQN

model = DQN((1,256,256), 6, filename='', pretrained_model_path='./models_/curve_away_hard/', save_model_path='').to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

rgba_image = Image.open('ImageSamples.jpg').convert('RGB')
gray_image = rgba_image.convert('RGB').convert('L')

# gray_image.save('received_image.png')
image = np.asarray(gray_image)
state = image
history = np.stack((state, state, state, state), axis=0)
history = np.reshape([history], (4, 1, 256, 256))

print(history.shape)
model.forward(history)