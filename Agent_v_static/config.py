#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:38:21 2020

@author: filip

Configuration of parameters
"""
import torch
from torch import nn
from calc_nash_monopoly import act_to_price, actions_dict

# Hyperparameters
HYPERPARAMS = {
        'full_obs_NB': {
                'gamma': 0.95, 
                'batch_size': 64,
                'replay_size': 400_000,
                'replay_start_size': 100_000,
                'learning_rate': 0.0001,
                'sync_target_frames': 10_000, 
                'epsilon_decay_last_frame': 1_000_000,
                'epsilon_start': 1,
                'epsilon_final': 0.01,
                'nA': 30,
                'dO': 6,
                'dO_a': 4,
                'frames': 2_000_000,
                'seed': 1,
                'path': "checkpoint.pt",
                'nodes': 8, # For neural network (its hidden layers has same no. nodes)
                'p_end' : 0.001,
                'punishlen': 1
                }
        }

nA = HYPERPARAMS['full_obs_NB']['nA']
GAMMA = HYPERPARAMS['full_obs_NB']['gamma']

# Economic parameters
MEAN_C = 1
MEAN_Q = 2
FIRMLIST = []
for c in [MEAN_C]:
    for q in [1.9, 1.95, MEAN_Q, 2.05, 2.1]:
        FIRMLIST.append({'cost': c, 'quality': q})

A0 = 1
MU = 1/2
grid = nA # Higher values gives better approximation of nash/monopoly-profits

NASH_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "nash")
MONOPOLY_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "monopoly")
COLAB_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "colab")

MIN_PRICE = act_to_price(0, nA)
MAX_PRICE = act_to_price(nA, nA)



# TODO: what in econparams is being used?
ECONPARAMS = {
        'base_case': {
                'firmlist': FIRMLIST,
                'a0': A0,
                'mu': MU,
                'nash_actions': NASH_ACTIONS,
                'monopoly_actions': MONOPOLY_ACTIONS,
                'colab_actions': COLAB_ACTIONS}}

# Functions
def calc_loss(batch, net, tgt_net, device="cpu", double = True):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device).float()
    next_states_v = torch.tensor(next_states).to(device).float()
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.from_numpy(dones).to(device) #torch.from_numpy() ByteTensor

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

def profit_gain(reward, nash_profit, monopoly_profit):
    '''
    avg_profit_gain() gives an index of collusion.
    Note, it assumes that monopoly profit >= Nash profit, which is not
    necessarily true.
    INPUT
    reward......scalar. Mean profit above baseline over episodes. (baseline is nash_profit)
    OUTPUT
    pg..........normalised value of the scalar
    '''
    pg = (reward) / (monopoly_profit - nash_profit)
    return pg

def normalize_state(state):
    '''
    State: np.array([reward[0], action[0], reward[1], action[1],
    cost0, cost1, vert0, vert1, b]) 
    '''
    state[4] = (state[4] - MEAN_C)
    state[5] = (state[5] - MEAN_C)
    state[6] = (state[6] - MEAN_Q)
    state[7] = (state[7] - MEAN_Q)
    return(state)
    
    
    
# =============================================================================
# firm1 = {'cost': 1, 'quality': 1.9}
# firm0 = {'cost': 1, 'quality': 2.1}
# 
# col = colab_action(nA, a0, mu, firm0, firm1, tol = 0.00)
# mon = monopoly_action(nA, a0, mu, firm0, firm1)
# nash = nash_action(nA, a0, mu, firm0, firm1)
# cheat = np.array([10, 10])
# pcheat = profit(cheat, a0, mu, firm0, firm1, nA)
# pnash = profit(nash, a0, mu, firm0, firm1, nA)
# pmon = profit(mon, a0, mu, firm0, firm1, nA)
# pcheat - pnash
# pmon - pnash
# (pcheat - pnash )/ (pmon - pnash)
# =============================================================================
