import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import math
from Class.Environment import TestEnvironment
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
    def __init__(self, env, state_space, action_space, train_cnt=10000, replace_target_cnt=10000, gamma=0.99, eps_strt=0.1, 
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
        self.memory = ReplayBuffer()
        
        self.train_cnt = train_cnt

        # After how many training iterations the target network should update
        self.replace_target_cnt = replace_target_cnt
        self.learn_counter = 0

        # Initialise policy and target networks, set target network to eval mode
        self.policy_net = DQN(self.state_space, self.action_space, filename='test').to(self.device)
        self.target_net = DQN(self.state_space, self.action_space, filename='test_'+'target').to(self.device)
        self.target_net.eval()

        # If pretrained model of the modelname already exists, load it
        try:
            self.policy_net.load_model()
            print('loaded pretrained model')
        except:
            pass
        
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
    def greedy_action(self, obs):
        obs = torch.tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        action = self.policy_net(obs).argmax().item()
        return action

    # Returns an action based on epsilon greedy method
    def choose_action(self, obs):
        if random.random() > self.eps:
            action = self.greedy_action(obs)
        else:
            action = random.choice([x for x in range(self.action_space)])
        return action
        
    # Stores a transition into memory
    def store_transition(self, *args):
        self.memory.add_transition(*args)
        
    # Decrement epsilon 
    def dec_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_end \
                        else self.eps_end

    def sample_batch(self):
        batch = self.memory.sample_batch(self.batch_size)
        state_shape = batch.state[0].shape

        # Convert to tensors with correct dimensions
        state = torch.tensor(batch.state).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        action = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch.reward).float().unsqueeze(1).to(self.device)
        state_ = torch.tensor(batch.state_).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        done = torch.tensor(batch.done).float().unsqueeze(1).to(self.device)

        return state, action, reward, state_, done

    # Samples a single batch according to batchsize and updates the policy net
    def learn(self, num_iters=1):
        print('pointer :', self.memory.pointer)
        if self.memory.pointer < self.batch_size:
            return

        for i in range(num_iters):

            # Sample batch
            state, action, reward, state_, done = self.sample_batch()

            # Calculate the value of the action taken
            q_eval = self.policy_net(state).gather(1, action)

            # Calculate best next action value from the target net and detach from graph
            q_next = self.target_net(state_).detach().max(1)[0].unsqueeze(1)
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
        # self.policy_net.save_model()
        self.dec_eps()

    # Plays num_eps amount of games, while optimizing the model after each episode
    def train(self, num_eps=100, render=False):
        scores = []

        max_score = 0

        for i in range(num_eps):
            done = False

            # Reset environment and preprocess state
            _, _, obs = self.env.step(1)
            state = obs
            
            score = 0
            cnt = 0
            while not done:
                # Take epsilon greedy action
                action = self.choose_action(state)
                reward, current_position, obs_ = self.env.step(action)

                # Preprocess next state and store transition
                state_ = obs_
                self.store_transition(state, action, reward, state_, int(done), obs)

                score += reward
                obs = obs_
                state = state_
                cnt += 1
                
                print(cnt)
                if cnt == 5000:
                    done = True
                    
                if cnt % 1000 == 0:
                    # Train on as many transitions as there have been added in the episode
                    print(f'Learning x{math.ceil(cnt/self.batch_size)}')
                    self.learn(math.ceil(cnt/self.batch_size))

            # Maintain record of the max score achieved so far
            if score > max_score:
                max_score = score

            scores.append(score)
            print(f'Episode {i}/{num_eps}: \n\tScore: {score}\n\tAvg score (past 100):\
                \n\tEpsilon: {self.eps}\n\tTransitions added: {cnt}')
            
            

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
    env = TestEnvironment()
    agent = DQNAgent(env, state_space=(3, 256, 256), action_space=6)
    
    agent.train()
