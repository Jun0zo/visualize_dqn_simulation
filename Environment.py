import struct
import numpy as np
from Server import Server
from PIL import Image
import io

class Environment:
    def __init__(self):
        self.server = Server()
        self._connect_listen()
        self.action_dict = {0: '', 1: 'W', 2: 'S', 3: 'A', 4: 'D', 5: 'B'}
    
    def _connect_listen(self):
        self.server.listen()

    def step(self, next_action):
        next_action = self.action_dict[next_action]
        # Receive isDead
        reward = self.server.conn.recv(1)
        

        # Receive currentPosition
        current_pos_data = self.server.conn.recv(2 * 4)
        current_position = list(struct.unpack(str(2) + 'i', current_pos_data))

        # Receive Imagebytes
        image_bytes = self.server.conn.recv(self.server.BUFFER_SIZE)

        with open('received_image.jpg', 'wb') as f:
            f.write(image_bytes)
        
        image = Image.open(io.BytesIO(image_bytes))
        image = np.asarray(image)

        self.server.conn.send(next_action.encode())
        return reward, current_position, image