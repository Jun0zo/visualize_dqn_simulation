from agent.DQN_agent import DQNAgent
from agent.Class.Environment import Environment

if __name__ == '__main__':
    result_path = './results/new_bam'
    env = Environment(results_path=result_path)
    agent = DQNAgent(env,
                     state_space=(1, 84, 84), 
                     action_space=5,
                     results_path=result_path, train_mode=True)
    
    # agent.play()
    agent.train(num_episode=1000)