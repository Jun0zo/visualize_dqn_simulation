import struct
import numpy as np
from .Server import Server
from PIL import Image
import io

BYTE_SIZE = 1
INT_SIZE = 4
FLOAT_SIZE = 4

class Environment:
    def __init__(self):
        self.server = Server()
        self._connect_listen()
        self.action_dict = {0: 'H', 1: 'W', 2: 'S', 3: 'A', 4: 'D', 5: 'B'}

        self.image_mem = None
    
    def _connect_listen(self):
        self.server.listen()


    def step(self, next_action):
        next_action = self.action_dict[next_action]

        is_done = self.server.conn.recv(1)
        is_done = bool(is_done[0])
        reward = struct.unpack('f', self.server.conn.recv(FLOAT_SIZE))[0]  # Assuming 'reward' is a float
        current_pos_data = self.server.conn.recv(2 * INT_SIZE)
        current_position = list(struct.unpack(str(2) + 'f', current_pos_data))  # Assuming 'currentPosition' is a list of two integers

        try:
        # Receive Imagebytes
            image_bytes = self.server.conn.recv(self.server.BUFFER_SIZE)
            rgba_image  = Image.open(io.BytesIO(image_bytes))
            gray_image = rgba_image.convert('RGB').convert('L')

            # gray_image.save('received_image.png')
            image = np.asarray(gray_image)

            self.image_mem = image
        except:
            print('Image get Fail!', is_done, type(is_done))
            image = self.image_mem


        self.server.conn.send(next_action.encode())
        return is_done, reward, current_position, image
    
    def save_image(self, image):
        with open('received_image.jpg', 'wb') as f:
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