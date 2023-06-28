import os
import site

import gymnasium as gym
import torch.optim as optim

site.addsitedir('../src/')
from model import DSNN, H1RateCodedDSNN
from datetime import date
from dqn_agent import Agent


# create environment
env_name = 'CartPole-v1'
env = gym.make(env_name)

# SNN parameters
alpha = 0.8
beta = 0.8
threshold = 0.5
time_steps = 10
learn_beta = False
learn_threshold = False
reset_mechanism = 'subtract' # options are 'subtract' 'zero' and 'none'

# number of input neurons is 2x observation space because of two-neuron encoding
architecture = [env.observation_space.shape[0]*2, 64, 64, 2]

# DQN parameters
batch_size = 128
gamma = 0.99995
eps_start = 1.0
eps_end = 0.05
eps_decay = 0.999
update_every = 4
target_update_frequency = 10
learning_rate = 0.001
memory_size = 4*10**4

n_runs = 1
n_evaluations = 100
num_episodes = 1000
seed = 0

#policy_net = H1RateCodedDSNN(architecture, alpha, beta, threshold, reset_mechanism, time_steps, seed)
#target_net = H1RateCodedDSNN(architecture, alpha, beta, threshold, reset_mechanism, time_steps, seed)
policy_net = DSNN(architecture, beta, threshold, reset_mechanism, time_steps, seed)
target_net = DSNN(architecture, beta, threshold, reset_mechanism, time_steps, seed)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# create a time-stamped results directory
dirs = os.listdir('.')
if not any('result' in d for d in dirs):
    result_id = 1
else:
    results = [d for d in dirs if 'result' in d]
    result_id = len(results) + 1

# Get today's date and add it to the results directory
d = date.today()
result_dir = 'dqn_result_' + str(result_id) + '_{}'.format(
    str(d.year) + str(d.month) + str(d.day))
os.mkdir(result_dir)
print('Created Directory {} to store the results in'.format(result_dir))

agent = Agent(env_name, policy_net, target_net, batch_size, memory_size, gamma, eps_start, eps_end,
              eps_decay, update_every, target_update_frequency, optimizer, learning_rate,
              num_episodes, env._max_episode_steps, 0, result_dir, seed, snn_time_steps=time_steps,
              two_neuron=True)

smoothed_scores, scores, best_average, best_average_after = agent.train_agent()
