#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:05:24 2019

@author: filip

Agent
"""
import numpy as np
import torch
#from cont_bertrand import ContBertrand
#env = ContBertrand()


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
        #self.state = self.env.reset()
        return
    # TODO: why not use the agents net attribute instead of passing it as an argument?
    def act(self, net, state,eps, device = "cpu"):
       # done_reward = 0 # TODO: should this be the PV of future cashflows?
        if np.random.uniform() < eps: # eps goes from 0 to 1 over iterations
            action = self.env.single_action_space.sample()
        else:
            state_v = torch.Tensor(state).to(device)
            q_vals_v = net(state_v)
            #_, action = self.max_value_action()
            _, act_v = torch.max(q_vals_v, dim=0) 
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
    