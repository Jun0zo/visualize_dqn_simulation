import socket
class Server:
    def __init__(self, host='localhost', port=5000, buffer_size=257*257, episodes=10):
        self.HOST = host
        self.PORT = port
        self.BUFFER_SIZE = buffer_size
        self.EPISODES = episodes
        self.conn = None
        self.addr = None

    def listen(self):
        print('Server is running!')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.HOST, self.PORT))
            s.listen()
            self.conn, self.addr = s.accept()
            print('Connected by', self.addr)