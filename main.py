from agent.DQN_agent import DQNAgent
from agent.Class.Environment import Environment

if __name__ == '__main__':
    env = Environment()
    agent = DQNAgent(env,
                     state_space=(1, 84, 84), 
                     action_space=6,
                     results_path='.\\results\\resized_3')
    
    agent.train(num_episode=100000)
    