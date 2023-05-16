import struct
import numpy as np
from Server import Server
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
    
    def _connect_listen(self):
        self.server.listen()


    def step(self, next_action):
        next_action = self.action_dict[next_action]

        # print(next_action)
        # Receive isDead
        is_done = self.server.conn.recv(BYTE_SIZE)
        reward = self.server.conn.recv(FLOAT_SIZE)
        # Receive currentPosition

        print(reward)
        current_pos_data = self.server.conn.recv(2 * INT_SIZE)
        current_position = list(struct.unpack(str(2) + 'i', current_pos_data))
        print(current_pos_data)

        # Receive Imagebytes
        image_bytes = self.server.conn.recv(self.server.BUFFER_SIZE)
        
        rgba_image  = Image.open(io.BytesIO(image_bytes))
        rgb_image = rgba_image.convert('RGB')
        image = np.asarray(rgb_image)


        self.server.conn.send(next_action.encode())
        return is_done, reward, current_position, image