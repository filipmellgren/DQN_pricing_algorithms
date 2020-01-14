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


class Agent1:
    def __init__(self, env, exp_buffer, net, tgt_net, optimizer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.net = net
        self.tgt_net = tgt_net
        self.optimizer = optimizer
        self.reset()
        
    def reset(self): 
        self.best_action = 0
        self.length_opt_act = 0
        self.total_pg = []
        self.best_mean_pg = None
        self.state = self.env.reset()
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
            self.time_same_best_action(action)
        return action
    
    def time_same_best_action(self, action):
        if action == self.best_action:
            self.length_opt_act += 1
        else:
            self.length_opt_act = 0
            self.best_action = action
        return
    