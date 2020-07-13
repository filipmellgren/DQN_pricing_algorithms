#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:05:24 2019

@author: filip

Agent
"""
import numpy as np
import torch

class Agent1:
    def __init__(self, env, exp_buffer, net, tgt_net, optimizer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.net = net
        self.tgt_net = tgt_net
        self.optimizer = optimizer
        self.reset()
        print("hej")
        
    def reset(self): 
        self.best_action = 0
        self.length_opt_act = 0
        self.total_pg = []
        self.best_mean_pg = None

        return
    def act(self, state,eps, device = "cpu"):
        '''
        act selects an action for the agent.
        With probability eps, the action is randomized uniformly from the 
        state space. Else, the action taken is the value with the corresponing
        highest Q-value.
        INPUT
        # TODO: Why not use agent.net?
        net......is a neural network which can give Q values based on state
        state....is the current state in the environment
        eps......probability to select a random action
        device...whether the processor is a CPU or GPU
        OUTPUT
        action...The agent's action
        '''
        if np.random.uniform() < eps: # eps goes from 0 to 1 over iterations
            action = self.env.single_action_space.sample()
        else:
            state_v = torch.Tensor(state).to(device)
            q_vals_v = self.net(state_v) # TODO: used to be simply net(state_v)
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
    