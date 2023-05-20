import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import math
from tqdm import tqdm
from collections import deque
from Class.Environment import TestEnvironment, Environment
from Class.ReplayBuffer import ReplayBuffer
from Class.transforms import Transforms
from Class.DQN_model import DQN
import os
import matplotlib.pyplot as plt

def get_highest_number(folder_path):
    files = os.listdir(folder_path)
    numbers = []

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.isdigit():
            numbers.append(int(file_name))

    if numbers:
        highest_number = max(numbers)
        return highest_number
    else:
        return 0

class Agent:
    def __init__(self, state_space=(256, 256, 3, 4), action_space=6):
        # 상태와 행동의 크기 정의
        self.state_space = state_space
        self.action_space = action_space
        self.EPISODES = 10
        

class DQNAgent(Agent):
    def __init__(self, 
                env, 
                state_space, 
                action_space, 
                tensor_board_path='',
                pretrained_model_path='./models', 
                save_model_path='./models', 
                train_cnt=5000, 
                replace_target_cnt=3000, 
                gamma=0.99, 
                eps_strt=0.5, 
                eps_end=0.001, 
                eps_dec=5e-6, 
                batch_size=32, 
                lr=0.001
                ):
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

        self.start_episode = get_highest_number(pretrained_model_path)
        print('biggest num : ', self.start_episode)

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

        if tensor_board_path:
            self.writer = SummaryWriter(tensor_board_path)
        else:
            self.writer = SummaryWriter()

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
    def greedy_action(self, history, save_maps):
        history = torch.tensor(history).float().to(self.device) # torch([256, 256])
        # history = history.squeeze(0)  # torch([1, 256, 256])
        res = self.policy_net(history)
        if save_maps:
            os.makedirs('./maps', exist_ok=True)
            num_files = len(os.listdir('./maps'))
            
            self.policy_net.forward_and_save(history, self.writer)
        action = res.argmax().item() % 3
        return action

    # Returns an action based on epsilon greedy method
    def choose_action(self, history, save_maps=False):
        if random.random() > self.eps:
            action = self.greedy_action(history, save_maps)
            
        else:
            action = random.choice([x for x in range(self.action_space)])
            print('=====================gridy', self.eps)
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
    def learn(self, num_iters=3):
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

            self.writer.add_scalar('Loss', loss.item(), i)


            # Increment learn_counter (for dec_eps and replace_target_net)
            self.learn_counter += 1

            # Check replace target net
            self.replace_target_net()

        # Save model & decrement epsilon
        self.dec_eps()

        self.writer.flush()

    # Plays num_eps amount of games, while optimizing the model after each episode
    def train(self, num_episode=1000):
        scores = []
        history = []
        global_cnt = 0
        learn_cnt = 0

        print('train_start')
        for episode_idx in range(self.start_episode, self.start_episode + num_episode):
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
                if global_cnt > 0 and global_cnt % 10000 == 0:
                    print('=========save===========')
                    action = self.choose_action(history, save_maps=True)
                    # action = self.choose_action(history)
                else:
                    action = self.choose_action(history)
                
                done, reward, current_position, observe = self.env.step(action)

                # self.writer.add_scalars(f'car/{episode_idx}/position', {'x': current_position[0], 'y': current_position[1]}, global_step=episode_idx)
                
                print(done, reward, current_position, 'action : ', action)
                next_state = observe
                next_state = np.reshape([next_state], (1, 1, 256, 256))
                next_history = np.append(next_state, history[:3,:, :, :], axis=0)
                
                self.append_sample(history, action, reward, next_history, int(done))

                score += reward
                cnt += 1
                global_cnt += 1

                history = next_history
                
                if len(self.memory) % 1000 == 0:
                    print('cnt :', cnt)


                if len(self.memory) % self.train_cnt == 0:
                    # Train on as many transitions as there have been added in the episode
                    print(f'Learning at {episode_idx}, {len(self.memory)}, score : {score}')
                    self.learn()
                    learn_cnt += 1

            if done == True:
                print('done true ======= ')
                print(len(self.memory))

            if cnt < 5:
                print('==============')
                continue

            # scores.append(score)
            print('Sum of Reward', score)
            print('Mean of Reward', score / cnt)
            self.writer.add_scalar("Sum of Reward", score, episode_idx)
            self.writer.add_scalar("Mean of Reward", score / cnt, episode_idx)
            print(f'Episode {episode_idx}/{num_episode}: \n\tScore: {score}\n\t \n\tEpsilon: {self.eps}')
            self.policy_net.save_model()
            self.env.save_trace(f'./traces/eps{episode_idx}-trace.txt')
            done = False
            

        self.writer.close()
    

if __name__ == '__main__':
    env = Environment()
    agent = DQNAgent(env,
                     state_space=(1, 256, 256), 
                     action_space=6,
                     tensor_board_path='',
                     pretrained_model_path='./models_/new_curve_mid/', 
                     save_model_path='./models_/new_curve_mid/')
    
    agent.train(num_episode=1000)
    
