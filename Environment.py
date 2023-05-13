class Environment:
    def __init__(self):
        self.server = Server()
        self._connect_listen()
    
    def _connect_listen(self):
        self.server.listen()

    def step(self, action):
        # Receive isDead
        reward = self.conn.recv(1)

        # Receive currentPosition
        current_pos_data = self.conn.recv(2 * 4)
        current_position = list(struct.unpack(str(2) + 'i', current_pos_data))

        # Receive Imagebytes
        image_bytes = self.conn.recv(self.BUFFER_SIZE)
        return reward, current_position, image_bytes