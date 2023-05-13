import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from Environment import Environment

class Agent:
    def __init__(self, action_size):
        # 상태와 행동의 크기 정의
        self.state_size = (256, 256, 3)
        self.action_size = action_size
        self.EPISODES = 10

class DQNAgent(Agent):
    def __init__(self, env, action_size):
        super().__init__(action_size)
        self.env = env
        
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        # self.train_start = 50000
        self.train_start = 500
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
        def train(inputs, a, y):
            self.optimizer.zero_grad()
            prediction = self.model(inputs)
            a_one_hot = torch.nn.functional.one_hot(a, self.action_size)
            q_value = torch.sum(prediction * a_one_hot, dim=1)
            error = torch.abs(y - q_value)
            quadratic_part = torch.clamp(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = torch.mean(0.5 * torch.square(quadratic_part) + linear_part)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        
        return train

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = nn.Sequential(
            nn.Conv2d(self.state_size[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )
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
        self.target_model.load_state_dict(self.model.state_dict())


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
            # history[i] = np.transpose(np.float32(mini_batch[i][0] / 255.), (2, 1, 0))
            # next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model(next_history[0]).detach()

        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])

        loss = F.mse_loss(self.model(history).gather(1, action.unsqueeze(1)), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.avg_loss += loss.item()

    def start(self):
        cnt = 0
        for e in range(self.EPISODES):
            done = False
            step = 0
            dead = False

            # for _ in range(random.randint(1, self.no_op_steps)):
            #     print('gggg')
            _, _, observe = self.env.step(1)

            state = observe
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (3, 256, 256, 4))

            while True:
                # try:
                action = self.get_action(history)
                # 선택한 행동으로 환경에서 한 타임스텝 진행
                reward, current_position, observe = self.env.step(action)
                # print('Received:', reward, current_position)

                # 각 타임스텝마다 상태 전처리
                
                next_state = observe
                next_state = np.reshape([next_state], (3, 256, 256, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)
                
            
                cnt += 1

                # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
                agent.append_sample(history, action, reward, next_history, dead)

                


                print(len(agent.memory), agent.train_start)
                if len(agent.memory) >= agent.train_start:
                    print('train!!')
                    agent.train_model()

                # # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                # if global_step % agent.update_target_rate == 0:
                #     agent.update_target_model()

                if dead:
                    dead = False
                else:
                    history = next_history

                # except Exception as ee:
                #     print('Timeout occurred', ee)
                #     continue

    def getRandomAction(self):
        # keys = ["W", "A", "S", "D", "B"]
        keys = [0, 1, 2, 3, 4, 5]
        probabilities = [0.05, 0.27, 0.1, 0.27, 0.25, 0.1]
        return random.choices(keys, probabilities, k=1)[0]

    def getAction(self):
        pass

    def save_image(self):
        # with open('received_image.jpg', 'wb') as f:
        #     f.write(data)
        pass

    

if __name__ == '__main__':
    env = Environment()
    agent = DQNAgent(env, 6)
    
    agent.start()
