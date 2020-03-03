import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class percept(nn.Module):
    def __init__(self, observation_dim, feature_dim, conv=False):
        super(percept, self).__init__()
        self.observation_dim = observation_dim
        self.feature_dim = feature_dim
        self.conv = conv

        if not self.conv:
            self.feature = nn.Sequential(
                nn.Linear(self.observation_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.feature_dim),
                nn.ReLU()
            )
        else:
            self.feature = nn.Sequential(
                nn.Conv2d(self.observation_dim[0], 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1)
            )
            self.fc = nn.Linear(self.feature_size(), self.feature_dim)

    def feature_size(self):
        tmp = torch.zeros(1, * self.observation_dim)
        return self.feature(tmp).view(1, -1).size(1)

    def forward(self, observation):
        feature = self.feature(observation)
        if self.conv:
            feature = F.relu(self.fc(feature.view(feature.size(0), -1)))
        return feature


class manager(nn.Module):
    def __init__(self, feature_dim, dilation):
        super(manager, self).__init__()
        self.feature_dim = feature_dim
        self.dilation = dilation

        self.mspace = nn.Linear(self.feature_dim, self.feature_dim)

        self.mrnn = nn.LSTMCell(self.feature_dim, self.feature_dim)

        self.h = [torch.zeros([1, self.feature_dim]) for _ in range(self.dilation)]
        self.c = [torch.zeros([1, self.feature_dim]) for _ in range(self.dilation)]

        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_mstate(self, feature):
        return self.mspace(feature)

    def get_goal(self, feature, hidden, count):
        mstate = self.get_mstate(feature)

        # * the implement of the dilation LSTM (dLSTM)
        h_t, c_t = hidden
        h_t_1 = self.h[count % self.dilation]
        c_t_1 = self.c[count % self.dilation]

        self.h[count % self.dilation] = h_t
        self.c[count % self.dilation] = c_t

        h, c = self.mrnn(mstate, (h_t_1, c_t_1))

        # * choose the h as the goal
        goal = h
        goal_norm = torch.norm(goal, p=2, dim=1).detach()
        goal = goal / goal_norm
        return goal, (h, c)

    def get_value(self, feature):
        return self.critic(self.get_mstate(feature))


class worker(nn.Module):
    def __init__(self, feature_dim, action_dim, k_dim):
        super(worker, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.k_dim = k_dim

        self.wrnn = nn.LSTMCell(self.feature_dim, self.k_dim * self.action_dim)

        self.phi = nn.Linear(self.feature_dim, self.k_dim, bias=False)

        # * two critics: respectively intricate and external reward
        self.critic_int =  nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.critic_ext = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_policy(self, feature, goals, hidden):
        goal_sum = goals.sum(0, keepdim=True)
        w = self.phi(goal_sum)
        w = w.unsqueeze(2)

        h, c = hidden

        h_new, c_new = self.wrnn(feature, (h, c))

        # * choose the h as the U
        u = h_new
        u = u.view(u.size(0), self.action_dim, self.k_dim)

        # * batch matrix multiply
        policy = torch.bmm(u, w).squeeze(2)
        policy = torch.softmax(policy, dim=1)
        return policy, (h_new, c_new)

    def get_value(self, feature):
        value_int = self.critic_int(feature)
        value_ext = self.critic_ext(feature)
        return value_int, value_ext


class feudal_networks(nn.Module):
    def __init__(self, observation_dim, feature_dim, k_dim, action_dim, dilation, horizon_c, conv=False):
        super(feudal_networks, self).__init__()
        self.feature_dim = feature_dim
        self.observation_dim = observation_dim
        self.k_dim = k_dim
        self.action_dim = action_dim
        self.dilation = dilation
        self.horizon_c = horizon_c
        self.conv = conv

        self.percept = percept(self.observation_dim, self.feature_dim, self.conv)
        self.manager = manager(self.feature_dim, dilation)
        self.worker = worker(self.feature_dim, self.action_dim, self.k_dim)
        self.goal_horizon = deque(maxlen=self.horizon_c)
        for _ in range(self.horizon_c):
            self.goal_horizon.append(torch.zeros([1, self.feature_dim]))

    def get_goal(self, observation, m_hidden, count):
        z = self.percept.forward(observation)

        mstate = self.manager.get_mstate(z)
        goal, m_hidden_new = self.manager.get_goal(z, m_hidden, count)
        m_value = self.manager.get_value(z)

        return mstate, goal, m_hidden_new, m_value

    def get_policy(self, observation, w_hidden):
        z = self.percept.forward(observation)

        goals = torch.cat(list(self.goal_horizon), dim=0)
        policy, w_hidden_new = self.worker.get_policy(z, goals, w_hidden)
        w_value_int, w_value_ext = self.worker.get_value(z)
        return policy, w_hidden_new, w_value_int, w_value_ext

    def store_goal(self, goal):
        self.goal_horizon.append(goal)