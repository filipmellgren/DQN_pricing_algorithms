#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:05:24 2019

@author: filip

Agent
"""

import numpy as np
import torch
import collections
from cont_bertrand import ContBertrand
env = ContBertrand()
from config import ECON_PARAMS
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
MIN_PRICE = ECON_PARAMS[4]
price_range = ECON_PARAMS[5]
MAX_PRICE = MIN_PRICE + price_range
NASH_PROFIT = ECON_PARAMS[6]
NASH_PROFIT = NASH_PROFIT[0]
MONOPOLY_PROFIT = ECON_PARAMS[7]

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0
    
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        
        if  np.random.random() < epsilon: 
            action = env.action_space.sample()
        else:            
            state_v = torch.Tensor(self.state)
            q_vals_v = net(state_v)
            #_, act_v = torch.max(q_vals_v, dim=1) I changed dim to 0, correct? #TODO
            _, act_v = torch.max(q_vals_v, dim=0) 
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

class Agent1:
    # TODO: must have its own Q-table,
    # must also make it observe only parts of it (max a slice of it, i.e. condition on variables it sees?)
    # First, value updating may have to happen here instead of WHERE
    def __init__(self, env, exp_buffer, net, tgt_net, optimizer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.net = net
        self.tgt_net = tgt_net
        self.optimizer = optimizer
        self.reset()
        
     #   self.values = collections.defaultdict(float) # The Q-table?! #TODO: maybe not necessary given initial_Q()
    def reset(self): #, nS, nA, gamma, c, ai, aj, a0, mu, price_range, min_price):
        self.best_action = 0
        self.length_opt_act = 0
        self.total_rewards = []
        self.best_mean_reward = None
       # self.initial_Q(nS, nA, gamma, c, ai, aj, a0, mu, price_range, min_price)
        self.state = self.env.reset(NASH_PROFIT, MIN_PRICE, MONOPOLY_PROFIT, MAX_PRICE) #?
        return
    
    def act(self, net, state,eps, device = "cpu"):
        done_reward = None
        if np.random.uniform() < eps: # eps goes from 0 to 1 over iterations
            action = self.env.single_action_space.sample()
        else:
            state_v = torch.Tensor(state)
            q_vals_v = net(state_v)
            #_, action = self.max_value_action()
            _, act_v = torch.max(q_vals_v, dim=0) # TODO: correct diumension?
            action = int(act_v.item())
            self.time_same_best_action(action) # from my old agent
            self.best_action = action # also from my old agent
        return action
    
    # TODO: step happens in the environment after combining the two actions

    
# =============================================================================
#     def max_value_action(self):
#         # works a bit like argmax
#         # "max_value_argmax_action"
#         max_value, best_action = None, None
#         for action in range(self.env.action_space.n):
#             action_value  = self.values[(self.state, action)]
#             if max_value is None or max_value < action_value:
#                 max_value = action_value
#                 best_action = action
#         return max_value, best_action
# =============================================================================
    
    def time_same_best_action(self, action):
        if action == self.best_action:
            self.length_opt_act += 1
        else:
            self.length_opt_act = 0
        return
    
    def value_update(self, s, a, r, next_s, alpha, gamma):
        self.state = next_s # Correct to update states here?
        best_v, _ = self.max_value_action()
        new_val = r + gamma * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-alpha) + new_val * alpha
        #self.state = next_s


    def initial_Q(self, nS, nA, gamma, c, ai, aj, a0, mu, price_range, min_price): # Initialize the Q-table.
        for s in range(nS):
            for a in range(nA):
                action = a + np.zeros((nA))
                action_other = np.arange(0.,nA) # Opponent randomizes uniformly
                actions = np.vstack([action, action_other])
                profit = profit_n(actions.transpose(), nA, c, ai, aj, a0, mu, price_range, min_price) # TODO: check if transpose necessary
                self.values[s, a] = (sum(profit[:,0])) / ((1-gamma) * nA)
        return
