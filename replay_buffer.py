import numpy as np
import random
from collections import deque


class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, mstate, goal, m_value, policy, w_value_int, w_value_ext, reward_ext, done, action):
        self.memory.append([observation, mstate, goal, m_value, policy, w_value_int, w_value_ext, reward_ext, done, action])

    def sample(self):
        batch = list(self.memory)
        observations, mstates, goals, m_values, policies, w_values_int, w_values_ext, rewards_ext, dones, actions = zip(* batch)
        return observations, mstates, goals, m_values, policies, w_values_int, w_values_ext, rewards_ext, dones, actions

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()