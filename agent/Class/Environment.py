import struct
import numpy as np
from .Server import Server
from PIL import Image
from PIL import ImageFile
import io

BYTE_SIZE = 1
INT_SIZE = 4
FLOAT_SIZE = 4

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Environment:
    def __init__(self):
        self.server = Server()
        self._connect_listen()
        self.action_dict = {0: 'H', 1: 'W', 2: 'S', 3: 'A', 4: 'D', 5: 'B'}
        # self.action_dict = {0: 'H', 1: 'A', 2: 'D'}

        self.image_mem = None

        self.trace = []
    
    def _connect_listen(self):
        self.server.listen()


    def step(self, next_action):
        next_action = self.action_dict[next_action]

        data = self.server.conn.recv(2 + FLOAT_SIZE + 2 * INT_SIZE + self.server.BUFFER_SIZE)
        is_done = bool(data[0])
        reward = struct.unpack('f', data[1:5])[0]  # Assuming 'reward' is a float
        current_position = list(struct.unpack(str(2) + 'f', data[5:13]))  # Assuming 'currentPosition' is a list of two integers

        try:
            # Receive Imagebytes
            rgba_image = Image.open(io.BytesIO(data[13:]))
            
            gray_image = rgba_image.convert('RGB').convert('L')
            image = np.asarray(gray_image)
            
            self.image_mem = image
        except Exception as e:
            print('Image get Fail!', is_done, len(data), e)
            image = self.image_mem
            # self.server.conn.recv(self.server.BUFFER_SIZE)/
        
        self.server.conn.send(next_action.encode())
        
        self.trace.append(current_position)
        return is_done, reward, current_position, image

    def get_trace(self):
        return_trace = self.trace
        self.trace = []
        return return_trace

    def save_trace(self, filename):
        with open(filename, 'w') as file:
            for item in self.trace:
                file.write(','.join(map(str, item)) + '\n')
        self.trace = []

    
    def save_image(self, image):
        with open('received_image.png', 'wb') as f:
            f.write(image)
    

class TestEnvironment:
    def __init__(self):
        self.action_dict = {0: 'H', 1: 'W', 2: 'S', 3: 'A', 4: 'D', 5: 'B'}
    
    def _connect_listen(self):
        self.server.listen()
        
    def step(self, next_action):
        reward = 0
        current_position = [10,20]
        rgb_image = Image.open('received_image.jpg').convert('RGB')
        image_array = np.asarray(rgb_image)
        image = np.transpose(image_array, (2, 0, 1))
        return reward, current_position, image