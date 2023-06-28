import os
import site
import torch

import gymnasium as gym
import numpy as np

site.addsitedir('../src/')
from model import DSNN
from utils import *
from datetime import date
from snntorch import spikegen


# create environment
env_name = 'CartPole-v1'
env = gym.make(env_name)

# SNN parameters
beta = 0.5
threshold = 0.1
snn_time_steps = 5
learn_beta = False
learn_threshold = False
reset_mechanism = 'subtract' # other options are 'zero' and 'none'
two_neuron = True

# number of input neurons is 2x observation space because of two-neuron encoding
architecture = [env.observation_space.shape[0]*2, 64, 64, 2]

# DQN parameters
batch_size = 128
gamma = 0.999
eps_start = 1.0
eps = 0.05
update_every = 4
target_update_frequency = 100
learning_rate = 0.001
memory_size = 4*10**4

n_runs = 3
n_evaluations = 100
seed = 0

result_dir = 'dqn_result_37_2023515'

policy_net = DSNN(architecture, beta, threshold, reset_mechanism, snn_time_steps)
model = torch.load(result_dir + '/checkpoint_DQN_0.pt')
policy_net.load_state_dict(model)

def select_action(state, eps):
    spike_data = spikegen.rate(state, num_steps=snn_time_steps)
    with torch.no_grad():
        spikes, membrane_potential = policy_net.forward(spike_data)
    total_spikes.append(spikes)
    return np.argmax(membrane_potential[-1]).item(), total_spikes

state = env.reset(seed=seed)
total_spikes = []
if two_neuron:
    state = two_neuron_encoding(state[0])
    score = 0
    done = False
    while not done:
        action, spikes = select_action(state, eps)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        score += reward
        if done:
            break
print(score)