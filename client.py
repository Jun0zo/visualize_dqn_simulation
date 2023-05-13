from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
import numpy as np
import socket
import random
import struct
from collections import deque
from keras import backend as K
import tensorflow as tf

from Server import Server
from Environment import Environment

class DQNAgent:
    def __init__(self, env, action_size):
        self.env = env
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99

        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])
        
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def start(self):
        cnt = 0
        for e in range(self.EPISODES):
            done = False
            step = 0
            while True:
                try:
                    # Receive any messages from the client and print them out
                    reward, current_position, image_bytes = self.get_current_states()
                    print('Received:', reward, current_position)
                    
                    next_action = self.getRandomAction()

                    # print(next_action, next_action.encode())
                    self.conn.send(next_action.encode())
                except socket.timeout:
                    print('Timeout occurred')
                    continue
                cnt += 1

    def getRandomAction(self):
        keys = ["W", "A", "S", "D", "B"]
        probabilities = [0.2, 0.2, 0.2, 0.2, 0.1]
        return "+".join(random.choices(keys, probabilities, k=1))

    def getAction(self):
        pass

    def save_image(self):
        # with open('received_image.jpg', 'wb') as f:
        #     f.write(data)
        pass

    

if __name__ == '__main__':
    env = Environment()
    agent = DQNAgent(env)
    agent.start()
