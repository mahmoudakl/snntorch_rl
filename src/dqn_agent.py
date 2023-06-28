import torch
import random

import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from utils import *
from snntorch import spikegen
from collections import deque, namedtuple


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward",
                                                                "next_state", "done"])
        random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).\
            float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).\
            long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).\
            float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).
                                 astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent:
    def __init__(self, env, policy_net, target_net, batch_size, memory_size, gamma, eps_start,
                 eps_end, eps_decay, update_every, target_update_frequency, optimizer,
                 learning_rate, num_episodes, max_steps, i_run, result_dir, seed, snn=True,
                 snn_time_steps=10, quantization=False, two_neuron=False):

        self.env = gym.make(env)
        #self.max_observation = self.env.observation_space.high
        self.max_observation = [4.8, 3.2965348 , 0.41887903, 2.508906]
        self.adjust_max_obs()
        self.outputs = []
        self.losses = []

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.policy_net = policy_net
        self.target_net = target_net

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.update_every = update_every
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.i_run = i_run
        self.seed = seed
        self.result_dir = result_dir
        self.snn = snn
        self.snn_time_steps = snn_time_steps
        self.quantization = quantization
        self.two_neuron = two_neuron

        self.threshold_update_frequency = 1
        self.scores_window = deque(maxlen=100)
        self.previous_reward = 0
        self.i_episode = 0

        # Initialize Replay Memory
        self.memory = ReplayBuffer(self.memory_size, self.batch_size, seed)

        # Initialize time step
        self.t_step = 0
        self.t_step_total = 0

    def adjust_max_obs(self):
        self.inf_observations = []
        for i in range(len(self.max_observation)):
            if self.max_observation[i] > 100:
                self.max_observation[i] = 1
                self.inf_observations.append(i)

    def update_max_observations(self, obs):
        """
        update max_observation of input space dims that are not bounded
        :param obs:
        :return:
        """
        for i in self.inf_observations:
            if obs[i] > self.max_observation[i]:
                self.max_observation[i] = obs[i]

    def normalize_state(self, state):
        if self.two_neuron:
            two_neuron_max_obs = np.array([val for val in self.max_observation for _ in (0, 1)])
            return torch.tensor(state / two_neuron_max_obs, dtype=torch.float)

        return torch.tensor(state / self.max_observation, dtype=torch.float)

    def select_action(self, state, eps=0.):
        #state = torch.from_numpy(state).unsqueeze(0).to(device)

        if random.random() > eps:
            # encode observations in spike trains in case of SNN
            if self.snn:
                state = self.normalize_state(state)
                #spike_data = spikegen.rate(state, num_steps=self.snn_time_steps)
            with torch.no_grad():
                if self.snn:
                    spikes, membrane_potential = self.policy_net.forward(state)
                    self.outputs.append(membrane_potential[-1])
                    return np.argmax(membrane_potential[-1]).item()
                else:
                    return np.argmax(self.policy_net.forward(state.float())[0].cpu().data.numpy())
        else:
            return random.choice(np.arange(self.policy_net.architecture[-1]))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.optimize_model(experiences)

    def soft_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.snn:
            next_states = self.normalize_state(next_states)
            #spike_data = spikegen.rate(next_states, num_steps=self.snn_time_steps)
            spikes, membrane_potential = self.target_net.forward(next_states)
            Q_targets_next = membrane_potential[-1].max(1)[0].detach().unsqueeze(1)
        else:
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next*(1 - dones))

        # Get expected Q values from local model
        if self.snn:
            states = self.normalize_state(states)
            #spike_data = spikegen.rate(states, num_steps=self.snn_time_steps)
            spikes, membrane_potential = self.policy_net.forward(states)
            Q_expected = membrane_potential[-1].gather(1, actions)
        else:
            Q_expected = self.policy_net.forward(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        #self.losses.append(loss.item())
        # Minimize the loss
        self.optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward()
        self.optimizer.step()

        if self.t_step_total % self.target_update_frequency == 0:
            self.soft_update()

    def train_agent(self):
        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        self.scores_window = deque(maxlen=100)
        eps = self.eps_start

        for self.i_episode in range(1, self.num_episodes + 1):
            state = self.env.reset(seed=self.seed)
            self.update_max_observations(state[0])
            if self.two_neuron:
                state = two_neuron_encoding(state[0])
            score = 0
            done = False
            while not done:
                self.t_step_total += 1
                action = self.select_action(state, eps)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.update_max_observations(next_state)
                if self.two_neuron:
                    next_state = two_neuron_encoding(next_state)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                eps = max(self.eps_end, self.eps_decay * eps)
                if done:
                    break
            self.scores_window.append(score)
            scores.append(score)
            smoothed_scores.append(np.mean(self.scores_window))

            # save model if smoothed average reward is higher than before
            if smoothed_scores[-1] > best_average:
                best_average = smoothed_scores[-1]
                best_average_after = self.i_episode

                torch.save(self.policy_net.state_dict(),
                           self.result_dir + '/checkpoint_DQN_{}.pt'.format(self.i_run))

            print("Episode {}\tAverage Score: {:.2f}\t Epsilon: {:.2f}\t Threshold: {}".
                  format(self.i_episode, np.mean(self.scores_window), eps,
                         self.policy_net.threshold),
                  end='\r')

            if self.i_episode % 100 == 0:
                print("\rEpisode {}\tAverage Score: {:.2f}".
                      format(self.i_episode, np.mean(self.scores_window)))

        print('Best 100 episode average: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        return smoothed_scores, scores, best_average, best_average_after
