import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import math
from tqdm import tqdm
from collections import deque
from Class.Environment import TestEnvironment, Environment
from Class.ReplayBuffer import ReplayBuffer
from Class.transforms import Transforms
from Class.DQN_model import DQN


class Agent:
    def __init__(self, state_space=(256, 256, 3, 4), action_space=6):
        # 상태와 행동의 크기 정의
        self.state_space = state_space
        self.action_space = action_space
        self.EPISODES = 10

class DQNAgent(Agent):
    def __init__(self, env, state_space, action_space, pretrained_model_path='./models', save_model_path='./models', train_cnt=5000, replace_target_cnt=3000, gamma=0.99, eps_strt=0.1, 
                eps_end=0.001, eps_dec=5e-6, batch_size=32, lr=0.001):
        super().__init__(state_space, action_space)
        self.env = env
        
        # Set global variables
        self.env = env
        self.batch_size = batch_size
        self.GAMMA = gamma
        self.LR = lr
        self.eps = eps_strt
        self.eps_dec = eps_dec
        self.eps_end = eps_end

        # Use GPU if available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialise Replay Memory
        self.memory = deque(maxlen=100000)
        
        self.train_cnt = train_cnt

        # After how many training iterations the target network should update
        self.replace_target_cnt = replace_target_cnt
        self.learn_counter = 0

        # Initialise policy and target networks, set target network to eval mode
        self.policy_net = DQN(self.state_space, self.action_space, filename='test', pretrained_model_path=pretrained_model_path, save_model_path=save_model_path).to(self.device)
        self.target_net = DQN(self.state_space, self.action_space, filename='test_'+'target').to(self.device)
        self.target_net.eval()

        # If pretrained model of the modelname already exists, load it
        try:
            self.policy_net.load_model()
            print('loaded pretrained model')
        except Exception as e:
            print('err : ', e)
        
        # Set target net to be the same as policy net
        self.replace_target_net()

        # Set optimizer & loss function
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.loss = torch.nn.SmoothL1Loss()

    # Updates the target net to have same weights as policy net
    def replace_target_net(self):
        if self.learn_counter % self.replace_target_cnt == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('Target network replaced')

    # Returns the greedy action according to the policy net
    def greedy_action(self, history):
        history = torch.tensor(history).float().to(self.device) # torch([256, 256])
        # history = history.squeeze(0)  # torch([1, 256, 256])
        action = self.policy_net(history).argmax().item()
        return action

    # Returns an action based on epsilon greedy method
    def choose_action(self, history):
        if random.random() > self.eps:
            action = self.greedy_action(history)
        else:
            action = random.choice([x for x in range(self.action_space)])
        return action
        
    # Stores a transition into memory
    def append_sample(self, *args):
        self.memory.append(args)
        
    # Decrement epsilon 
    def dec_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_end \
                        else self.eps_end

    def sample_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        history = torch.zeros((self.batch_size, self.state_space[0],
                                self.state_space[1], self.state_space[2]))
        next_history = torch.zeros((self.batch_size, self.state_space[0],
                                    self.state_space[1], self.state_space[2]))
        target = torch.zeros((self.batch_size,))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            history_, action_, reward_, next_history_, done_ = mini_batch[i]
            history[i] = torch.FloatTensor(history_[0]/255.)
            next_history[i] = torch.FloatTensor(next_history_[0]/255.)
            action.append(action_)
            reward.append(reward_)
            done.append(done_)

        return history, torch.tensor(action), torch.tensor(reward), next_history, torch.tensor(done)

    # Samples a single batch according to batchsize and updates the policy net
    def learn(self, num_iters=100):
        for i in tqdm(range(num_iters)):

            # Sample batch
            history, action, reward, next_history, done = self.sample_batch()

            # Calculate the value of the action taken
            q_eval = self.policy_net(history).gather(1, action.unsqueeze(1))

            # Calculate best next action value from the target net and detach from graph
            q_next = self.target_net(next_history).detach().max(1)[0].unsqueeze(1)
            # Using q_next and reward, calculate q_target
            # (1-done) ensures q_target is 0 if transition is in a terminating state
            q_target = (1-done) * (reward + self.GAMMA * q_next) + (done * reward)

            # Compute the loss
            # loss = self.loss(q_target, q_eval).to(self.device)
            loss = self.loss(q_eval, q_target).to(self.device)

            # Perform backward propagation and optimization step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Increment learn_counter (for dec_eps and replace_target_net)
            self.learn_counter += 1

            # Check replace target net
            self.replace_target_net()

        # Save model & decrement epsilon
        self.dec_eps()

    # Plays num_eps amount of games, while optimizing the model after each episode
    def train(self, num_eps=1000, render=False):
        scores = []
        history = []

        print('train_start')
        for i in range(1, num_eps):
            done = False

            # Reset environment and preprocess state
            _, _, _, obs = self.env.step(1)
            state = obs
            history = np.stack((state, state, state, state), axis=0)
            history = np.reshape([history], (4, 1, 256, 256))
            
            score = 0
            cnt = 0
            
            while not done:
                # Take epsilon greedy action
                action = self.choose_action(history)
                
                done, reward, current_position, observe = self.env.step(action)

                next_state = observe
                next_state = np.reshape([next_state], (1, 1, 256, 256))
                next_history = np.append(next_state, history[:3,:, :, :], axis=0)
                
                self.append_sample(history, action, reward, next_history, int(done))

                score += reward
                cnt += 1
                
                if cnt % 1000 == 0:
                    print('cnt :', cnt)
                if len(self.memory) % self.train_cnt == 0:
                    # Train on as many transitions as there have been added in the episode
                    print(f'Learning at {i}, {len(self.memory)}, score : {score}')
                    self.learn()


            scores.append(score)
            print(f'Episode {i}/{num_eps}: \n\tScore: {score}\n\t \n\tEpsilon: {self.eps}')
            self.policy_net.save_model()
            done = False
            

    def getRandomAction(self):
        # keys = ["W", "A", "S", "D", "B"]
        keys = [0, 1, 2, 3, 4, 5]
        probabilities = [0.05, 0.27, 0.1, 0.27, 0.25, 0.1]
        return random.choices(keys, probabilities, k=1)[0]

    def getAction(self):
        pass

    

    

if __name__ == '__main__':
    env = Environment()
    agent = DQNAgent(env, state_space=(1, 256, 256), pretrained_model_path='./models_/curve_away_hard/', save_model_path='./models_/curve_away_hard/', action_space=6)
    
    agent.train()
