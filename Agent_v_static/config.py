#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:38:21 2020

@author: filip

Configuration of parameters
"""
import torch
from torch import nn
from calc_nash_monopoly import act_to_price, profit, nash_action, monopoly_action, max_profit, min_profit, actions_dict

# Hyperparameters
HYPERPARAMS = {
        'full_obs_NB': {
                'gamma': 0.95, 
                'batch_size': 16, 
                'replay_size': 50_000,
                'replay_start_size': 50_000,
                'learning_rate': 0.01,
                'sync_target_frames': 50_000,
                'epsilon_decay_last_frame': 500_000,
                'epsilon_start': 1,
                'epsilon_final': 0.1,
                'nA': 50,
                'dO': 6,
                'dO_a': 6,
                'frames': 1_000_000,
                'seed': 1,
                'path': "checkpoint.pt",
                'nodes': 32 # For neural network (its hidden layers has same no. nodes)
                },
        'deepmind2015': {
                'gamma': 0.99,
                'batch_size': 32,
                'replay_size': 1_000_000,
                'replay_start_size': 50_000,
                'learning_rate': 0.00025, # Note, also included momentum
                'sync_target_frames': 10_000, # They don't save only when a record has been reached
                'epsilon_decay_last_frame': 1_000_000,
                'epsilon_start': 1,
                'epsilon_final': 0.1,
                'nA': 20,
                'dO': 6,
                'dO_a': 4,
                'frames': 5_000,
                'seed': 1,
                'path': "checkpoint.pt"
                }
        }

nA = HYPERPARAMS['full_obs_NB']['nA']
GAMMA = HYPERPARAMS['full_obs_NB']['gamma']

# Economic parameters
FIRMLIST = []
for c in [0.9, 1, 1.1]:
    for q in [0.9, 1, 1.1]:
        FIRMLIST.append({'cost': c, 'quality': q})

A0 = 1
MU = 1/2
grid = nA # Higher values gives better approximation of nash/monopoly-profits

NASH_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "nash")
MONOPOLY_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "monopoly")
COLAB_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "colab")

#NASH_ACTION = nash_action(grid, A0, MU, firm0, firm1)
#NASH_PRICE = act_to_price(NASH_ACTION, grid)
#NASH_PROFIT = profit(NASH_ACTION, A0, MU, firm0, firm1, grid)

#MONOPOLY_ACTION = monopoly_action(grid, A0, MU, firm0, firm1) # TODO: Weird results here. Likely because of the way the monopoly profit is defined
#MONOPOLY_PRICE = act_to_price(MONOPOLY_ACTION, grid)
#MONOPOLY_PROFIT = profit(MONOPOLY_ACTION, A0, MU, firm0, firm1, grid) 

#MIN_PROFIT = min_profit(nA, A0, MU, firm0, firm1)
#MAX_PROFIT = max_profit(nA, A0, MU, firm0, firm1)
MIN_PRICE = act_to_price(0, nA)
MAX_PRICE = act_to_price(nA, nA)



# TODO: what in econparams is being used?
ECONPARAMS = {
        'base_case': {
                #'firm0': firm0,
                #'firm1': firm1,
                'firmlist': FIRMLIST,
                'a0': A0,
                'mu': MU,
                #'nash_profit': NASH_PROFIT,
                #'monopoly_profit': MONOPOLY_PROFIT, 
                #'min_price': MIN_PRICE,
                #'max_price': MAX_PRICE,
                #'monopoly_action': MONOPOLY_ACTION,
                #'nash_action': NASH_ACTION,
                #'min_profit': MIN_PROFIT,
                #'max_profit': MAX_PROFIT,
                'nash_actions': NASH_ACTIONS,
                'monopoly_actions': MONOPOLY_ACTIONS,
                'colab_actions': COLAB_ACTIONS}}

# Functions
def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device).float()
    next_states_v = torch.tensor(next_states).to(device).float()
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.from_numpy(dones).to(device) #torch.from_numpy() ByteTensor

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def profit_gain(reward, nash_profit, monopoly_profit):
    '''
    avg_profit_gain() gives an index of collusion
    INPUT
    reward......scalar. Mean profit over episodes.
    OUTPUT
    pg..........normalised value of the scalar
    '''
    pg = (reward - nash_profit) / (monopoly_profit - nash_profit)
    return pg

