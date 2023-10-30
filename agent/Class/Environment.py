import struct
import numpy as np
from .Server import Server
from PIL import Image
from PIL import ImageFile
import io
import cv2

BYTE_SIZE = 1
INT_SIZE = 4
FLOAT_SIZE = 4

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Environment:
    def __init__(self, results_path):
        self.results_path = results_path
        self.server = Server()
        self._connect_listen()
        # self.action_dict_6 = {0: 'H', 1: 'W', 2: 'S', 3: 'A', 4: 'D', 5: 'B'}
        self.action_dict = {0: '-', 1: 'W', 2: 'S', 3: 'A', 4: 'D'}
        
        self.image_mem = None

        self.trace_fd = None

        self.idx = 0
        
    def position_trace_start(self):
        self.trace_fd = open(self.results_path + '\\trace.txt', 'a')
    
    def position_trace_end(self):
        self.trace_fd.close()
    
    def _connect_listen(self):
        self.server.listen()


    def step(self, next_action):
        next_action = self.action_dict[next_action]
        self.server.conn.send(next_action.encode())
        

        received_bytes = self.server.conn.recv(self.server.BUFFER_SIZE)
        packet_size = struct.unpack('i', received_bytes[:4])[0]
        is_done = struct.unpack('b', received_bytes[4:5])[0]
        reward = struct.unpack('f', received_bytes[5:9])[0]
        current_position = list(struct.unpack('2f', received_bytes[9:17]))
        image_bytes = received_bytes[17:]

        received_bytes_len = len(received_bytes)
        while received_bytes_len < packet_size:
            image_bytes_plus = self.server.conn.recv(packet_size - len(received_bytes))
            received_bytes_len += len(image_bytes_plus)
            image_bytes += image_bytes_plus
        # print(image_bytes)/


        print("info : ", packet_size, is_done, reward, current_position)
        try:
            # Receive Imagebytes
            rgba_image = Image.open(io.BytesIO(image_bytes))
            rgba_image.save(f"{self.results_path}/imgs/{self.idx}.jpg")
            self.idx += 1
            
            gray_image = rgba_image.convert('RGB').convert('L')
            # gray_image.save('./test.jpg')
            image = np.asarray(gray_image) / 255.0
            # cv2.imwrite('(recivend)image_from_bytes2.png', image)

            self.image_mem = image
        except Exception as e:
            # print('Image get Fail!', is_done, len(received_bytes), e)
            print("fail !")
            image = self.image_mem
            # self.server.conn.recv(self.server.BUFFER_SIZE)/
        
        self.trace_fd.write(', '.join(map(str, current_position)) + '\n')
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