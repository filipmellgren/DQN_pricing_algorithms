#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:38:21 2020

@author: filip

Configuration of parameters
"""
import gym
import numpy as np
import torch
from torch import nn
import argparse
from calc_nash_monopoly import act_to_price, demand, profit, nash_action, monopoly_action 

# Hyperparameters
HYPERPARAMS = {
        'full_obs_NB': {
                'gamma': 0.99,
                'batch_size': 1000,
                'replay_size': 25_000,
                'replay_start_size': 25_000,
                'learning_rate': 0.01,
                'sync_target_frames': 50_000,
                'epsilon_decay_last_frame': 400_000,
                'epsilon_start': 1,
                'epsilon_final': 0.02,
                'nA': 10,
                'dO': 6,
                'dO_a': 4,
                'frames': 5_000
                },
        }

nA = HYPERPARAMS['full_obs_NB']['nA']
GAMMA = HYPERPARAMS['full_obs_NB']['gamma']

# Economic parameters
C = 1
A = 2
A0 = 1
MU = 1/2

NASH_ACTION = nash_action(nA, A0, A, A, MU)
NASH_PRICE = act_to_price(NASH_ACTION)
NASH_PROFIT = profit(NASH_PRICE, A0, A, A, MU)

MONOPOLY_ACTION = monopoly_action(nA, A0, A, A, MU)
MONOPOLY_PRICE = act_to_price(MONOPOLY_ACTION)
MONOPOLY_PROFIT = profit(MONOPOLY_PRICE, A0, A, A, MU) # Sum and divide by two?

# MAX AND MIN
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

parser = argparse.ArgumentParser(description='Pricing algorithms')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args() # TODO: load just this guy for minimalism


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

def avg_profit_gain(avg_profit, nash_profit = NASH_PROFIT[0], monopoly_profit = MONOPOLY_PROFIT):
    '''
    avg_profit_gain() gives an index of collusion
    INPUT
    avg_profit......scalar. Mean profit over episodes.
    OUTPUT
    apg.............normalised value of the scalar
    '''
    apg = (avg_profit - nash_profit) / (monopoly_profit - nash_profit)
    return apg

def profit_n(action_n, nA = nA, c = C, ai = A, aj = A, a0 = A0, mu = MU, price_range = MAX_PRICE - MIN_PRICE, min_price = MIN_PRICE):
    '''
    profit_n gives profits in the market after taking prices as argument
    INPUT
    action_n.....an np.array([]) containing two prices
    OUTPUT
    profit.......profit, an np.array([]) containing profits
    '''
    a = np.array([ai, aj])
    a_not = np.flip(a) # to obtain the other firm's a
      
    p = (price_range * action_n/(nA-1)) + min_price 
    p_not = np.flip(p) # to obtain the other firm's p
    num = np.exp((a - p)/mu)
    denom = np.exp((a - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
    quantity_n = num / denom
          
    profit = quantity_n * (p-c)
    return(profit)

