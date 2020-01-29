#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:38:21 2020

@author: filip

Configuration of parameters
"""
import torch
from torch import nn
from calc_nash_monopoly import act_to_price, profit, nash_action, monopoly_action 

# Hyperparameters
HYPERPARAMS = {
        'full_obs_NB': {
                'gamma': 0.995,
                'batch_size': 750,
                'replay_size': 100_000,
                'replay_start_size': 50_000,
                'learning_rate': 0.00025,
                'sync_target_frames': 10_000,
                'epsilon_decay_last_frame': 1_000_000,
                'epsilon_start': 1,
                'epsilon_final': 0.1,
                'nA': 20,
                'dO': 6,
                'dO_a': 4,
                'frames': 20_000,
                'seed': 1,
                'path': "checkpoint.pt"
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
C = 1
A = 2
A0 = 1
MU = 1/2
grid = 500 # Higher values gives better approximation of nash/monopoly-profits
NASH_ACTION = nash_action(grid, A0, A, A, MU, C)
NASH_PRICE = act_to_price(NASH_ACTION, grid)
NASH_PROFIT = profit(NASH_ACTION, A0, A, A, MU, C, grid)

MONOPOLY_ACTION = monopoly_action(grid, A0, A, A, MU, C)
MONOPOLY_PRICE = act_to_price(MONOPOLY_ACTION, grid)
MONOPOLY_PROFIT = profit(MONOPOLY_ACTION, A0, A, A, MU, C, grid) 
MIN_PRICE = 0.9 * NASH_PRICE
MAX_PRICE = 1.1 * MONOPOLY_PRICE

ECONPARAMS = {
        'base_case': {
                'c': C,
                'a':A,
                'a0': A0,
                'mu': MU,
                'nash_profit': NASH_PROFIT,
                'monopoly_profit': MONOPOLY_PROFIT,
                'min_price': MIN_PRICE,
                'max_price': MAX_PRICE}}

#ENV = gym.make("CartPole-v1")
#ENV = ContBertrand()

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

def avg_profit_gain(avg_profit, nash_profit = NASH_PROFIT[0], monopoly_profit = MONOPOLY_PROFIT[0]):
    '''
    avg_profit_gain() gives an index of collusion
    INPUT
    avg_profit......scalar. Mean profit over episodes.
    OUTPUT
    apg.............normalised value of the scalar
    '''
    apg = (avg_profit - nash_profit) / (monopoly_profit - nash_profit)
    return apg

# =============================================================================
# def profit_n(action_n, nA = nA, c = C, ai = A, aj = A, a0 = A0, mu = MU, price_range = MAX_PRICE - MIN_PRICE, min_price = MIN_PRICE):
#     '''
#     profit_n gives profits in the market after taking prices as argument
#     INPUT
#     action_n.....an np.array([]) containing two prices
#     OUTPUT
#     profit.......profit, an np.array([]) containing profits
#     '''
#     a = np.array([ai, aj])
#     a_not = np.flip(a) # to obtain the other firm's a
#       
#     p = (price_range * action_n/(nA-1)) + min_price 
#     p_not = np.flip(p) # to obtain the other firm's p
#     num = np.exp((a - p)/mu)
#     denom = np.exp((a - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
#     quantity_n = num / denom
#           
#     profit = quantity_n * (p-c)
#     return(profit)
# =============================================================================

