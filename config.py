5#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:38:21 2020

@author: filip

Configuration of parameters
"""
import torch
from torch import nn
from calc_nash_monopoly import act_to_price, actions_dict
import numpy as np
# Hyperparameters
HYPERPARAMS = {
        'full_obs_NB': {
                'gamma': 0.95, 
                'batch_size': 2**9, # In Leibo, they use 2**10 and up. Bigger means more complex behaviour is possible but takes longer time
                'replay_size': 500_000,
                'replay_start_size': 100_000,
                'learning_rate': 0.001,
                'sync_target_frames': 10_000, 
                'epsilon_decay_last_frame': 500_000,
                'epsilon_start': 1,
                'epsilon_final': 0.01,
                'nA': 7, 
                'dO': 6,
                'dO_a': 2,
                'frames': 3_500_000,
                'seed': 1,
                'path': "checkpoint.pt",
                'nodes': 8, # For neural network (its hidden layers has same no. nodes)
                'p_end' : 0, # Episodes never end when nothing changes in env
                }
        }

nA = HYPERPARAMS['full_obs_NB']['nA']
GAMMA = HYPERPARAMS['full_obs_NB']['gamma']

# Economic parameters
MEAN_C = 1
MEAN_Q = 2

# Train with this one on.
FIRMLIST = []
for c in [MEAN_C]:
    for q in [MEAN_Q]:
        q = round(q, 2)
        FIRMLIST.append({'cost': c, 'quality': q})

A0 = 1
MU = 1/2
grid = nA # Higher values gives better approximation of nash/monopoly-profits
NASH_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "nash")
MONOPOLY_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "monopoly")

ECONPARAMS = {
        'base_case': {
                'firmlist': FIRMLIST,
                'a0': A0,
                'mu': MU,
                'nash_actions': NASH_ACTIONS,
                'monopoly_actions': MONOPOLY_ACTIONS
                }}

# Functions
def calc_loss(batch, net, tgt_net, device="cpu", double = True):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device).float()
    next_states_v = torch.tensor(next_states).to(device).float()
    actions_v = torch.tensor(actions).to(device)
    actions_v = actions_v.long()
    
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones).to(device) #torch.from_numpy() ByteTensor

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0 # if episode has ended,future value is 0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def profit_gain(reward, nash_profit):
    '''
    avg_profit_gain() gives an index of collusion.
    Note, it assumes that monopoly profit >= Nash profit, which is not
    necessarily true.
    INPUT
    reward......scalar. Mean profit above baseline over episodes. (baseline is nash_profit)
    OUTPUT
    pg..........normalised value of the scalar
    '''
    pg = (reward) / (nash_profit) # prev: (monopoly_profit - nash_profit)
    return pg

def normalize_state(state):
    '''
    State: np.array([action[0], action[1], vert0, vert1, b]) 
    '''
    state[2] = (state[2] - MEAN_Q)
    state[3] = (state[3] - MEAN_Q)
    return(state)