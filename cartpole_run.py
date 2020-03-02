import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from net import feudal_networks
from replay_buffer import replay_buffer
import gym


class feudal_model(object):
    def __init__(self, env, capacity, update_freq, episode, batch_size, feature_dim, k_dim, dilation, horizon_c, learning_rate, alpha, gamma, entropy_weight, render):
        # * feature_dim >> k_dim
        # * dilation = horizon_c
        # * capacity <= update_freq
        self.env = env
        self.capacity = capacity
        self.update_freq = update_freq
        self.episode = episode
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.k_dim = k_dim
        self.dilation = dilation
        self.horizon_c = horizon_c
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.render = render

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = feudal_networks(self.observation_dim, self.feature_dim, self.k_dim, self.action_dim, self.dilation, self.horizon_c)
        self.buffer = replay_buffer(self.capacity)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.h_m = torch.zeros([1, self.feature_dim])
        self.c_m = torch.zeros([1, self.feature_dim])
        self.h_w = torch.zeros([1, self.action_dim * self.k_dim])
        self.c_w = torch.zeros([1, self.action_dim * self.k_dim])
        self.count = 0
        self.weight_reward = None

    def get_returns(self, rewards, dones, values):
        returns = []
        run_return = values[-1]
        for i in reversed(range(rewards.size(0))):
            run_return = rewards[i] + self.gamma * run_return * (1. - dones[i])
            returns.append(run_return)
        returns = list(reversed(returns))
        returns = torch.cat(returns, dim=0).unsqueeze(1)
        return returns

    def train(self):
        # * Need to notice that the valid range of samples is [horizon_c: - horizon_c]
        observations, mstates, goals, m_values, policies, w_values_int, w_values_ext, rewards_ext, dones, actions = self.buffer.sample(self.batch_size)

        actions = torch.LongTensor(actions)
        observations = torch.FloatTensor(np.vstack(observations))
        dones = torch.FloatTensor(dones)
        rewards_ext = torch.FloatTensor(rewards_ext)
        m_values = torch.cat(m_values, 0)
        policies = torch.cat(policies, 0)
        w_values_ext = torch.cat(w_values_ext, 0)
        w_values_int = torch.cat(w_values_int, 0)

        rewards_int = []
        for i in range(self.horizon_c, observations.size(0)):
            s = mstates[i]
            reward_int = 0
            for j in range(self.horizon_c):
                s_ = mstates[i - j - 1]
                g_ = goals[i - j - 1]
                reward_int += F.cosine_similarity(s - s_, g_)
            reward_int = reward_int / self.horizon_c
            rewards_int.append(reward_int)
        rewards_int = torch.cat(rewards_int, 0).unsqueeze(1)

        m_returns = self.get_returns(rewards_ext, dones, m_values)
        w_returns_ext = self.get_returns(rewards_ext, dones, w_values_ext)
        w_returns_int = self.get_returns(rewards_int, dones[self.horizon_c:], w_values_int[self.horizon_c:])

        m_adv = m_returns - m_values
        w_ext_adv = w_returns_ext - w_values_ext
        w_int_adv = w_returns_int[: -self.horizon_c, :] - w_values_int[self.horizon_c: -self.horizon_c, :]

        m_loss = []
        for i in range(0, observations.size(0) - self.horizon_c):
            s_ = mstates[i]
            s = mstates[i + self.horizon_c]
            g = goals[i]
            cos_sim = F.cosine_similarity(s - s_, g)
            m_loss.append(- m_adv[i].detach() * cos_sim)
        m_loss = torch.cat(m_loss, 0).unsqueeze(1)

        dists = torch.distributions.Categorical(policies)
        log_probs = dists.log_prob(actions)
        w_loss = - (w_ext_adv[self.horizon_c: -self.horizon_c, :] + self.alpha * w_int_adv).detach() * log_probs.unsqueeze(1)[self.horizon_c: -self.horizon_c, :] - self.entropy_weight * dists.entropy().unsqueeze(1)[self.horizon_c: -self.horizon_c]

        m_returns = m_returns[self.horizon_c: -self.horizon_c, :]
        m_values = m_values[self.horizon_c: -self.horizon_c, :]
        w_returns_ext = w_returns_ext[self.horizon_c: -self.horizon_c, :]
        w_values_ext = w_values_ext[self.horizon_c: -self.horizon_c, :]
        w_returns_int = w_returns_int[: -self.horizon_c, :]
        w_values_int = w_values_int[self.horizon_c: -self.horizon_c, :]
        m_loss = m_loss[self.horizon_c:, :]

        m_critic_loss = (m_returns.detach() - m_values).pow(2)
        w_critic_ext_loss = (w_returns_ext.detach() - w_values_ext).pow(2)
        w_critic_int_loss = (w_returns_int.detach() - w_values_int).pow(2)

        loss = m_loss + m_critic_loss + w_loss + w_critic_ext_loss + w_critic_int_loss
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                # * manager change the goal every horizon_c steps
                if self.count % self.horizon_c == 0:
                    mstate, goal, m_hidden_new, m_value = self.net.get_goal(torch.FloatTensor(np.expand_dims(obs, 0)), (self.h_m, self.c_m), self.count)
                    self.net.store_goal(goal)
                policy, w_hidden_new, w_value_int, w_value_ext = self.net.get_policy(torch.FloatTensor(np.expand_dims(obs, 0)), (self.h_w, self.c_w))
                self.h_m, self.c_m = m_hidden_new
                self.h_w, self.c_w = w_hidden_new
                dist = torch.distributions.Categorical(policy)
                action = dist.sample().detach().item()
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                self.count += 1
                self.buffer.store(obs, mstate, goal, m_value, policy, w_value_int, w_value_ext, reward, done, action)
                obs = next_obs

                if self.count % self.update_freq == 0:
                    self.train()

                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    print('episode: {}  reward: {}  weight_reward: {:.2f}'.format(i + 1, total_reward, self.weight_reward))
                    break


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    test = feudal_model(
        env=env,
        capacity=200,
        update_freq=200,
        episode=10000,
        batch_size=100,
        feature_dim=256,
        k_dim=16,
        dilation=10,
        horizon_c=10,
        learning_rate=1e-4,
        alpha=0.5,
        gamma=0.99,
        entropy_weight=1e-4,
        render=False
    )
    test.run()