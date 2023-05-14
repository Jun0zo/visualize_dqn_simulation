import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

action_size = 6
epsilon = 1.
epsilon_start, epsilon_end = 1.0, 0.1
exploration_steps = 1000000.
epsilon_decay_step = (epsilon_start - epsilon_end) \
                            / exploration_steps
batch_size = 32
# train_start = 50000
train_start = 500
update_target_rate = 10000
discount_factor = 0.99

state_size = (256, 256, 3)

# 리플레이 메모리, 최대 크기 400000
memory = deque(maxlen=400000)
no_op_steps = 30
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def build_model(input_shape, num_actions):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 512),
        nn.ReLU(),
        nn.Linear(512, num_actions)
    )
    return model

def update_target_model(model, target_model):
    target_model.load_state_dict(model.state_dict())

def train_model(model, target_model, memory):
    # Sample random minibatch of transitions from memory
    minibatch = random.sample(memory, 32)
    
    history = np.zeros((32, 256, 256, 3))
    next_history = np.zeros((32, 256, 256, 3))

    # Unpack minibatch
    state_batch, action_batch, reward_batch, next_state_batch, dead_batch = zip(*minibatch)
    
    # Compute Q values for current state and target values for next state
    q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (0.99 * next_q_values * (1 - dead_batch))

    # Compute loss and update model
    loss = F.smooth_l1_loss(q_values, expected_q_values)
    optimizer = optim.Adam(model.parameters(), lr=0.00025)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


model = build_model()
target_model = build_model()

history = 
action = 
reward = 0
next_history = 
dead = 0

memory = []