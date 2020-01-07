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
#from cont_bertrand import ContBertrand


# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32*2
REPLAY_SIZE = 10_000
REPLAY_START_SIZE = 10_000
LEARNING_RATE = 0.001*5
SYNC_TARGET_FRAMES = 1000
EPSILON_DECAY_LAST_FRAME = 100_000 - 20000
EPSILON_START =  1.0
EPSILON_FINAL = 0.02
MEAN_REWARD_BOUND = 195 # TODO: adapt!
nA = 15 # Number of actions
#?

PARAMS = np.array([GAMMA, BATCH_SIZE, REPLAY_SIZE, REPLAY_START_SIZE, 
                   LEARNING_RATE,SYNC_TARGET_FRAMES, EPSILON_DECAY_LAST_FRAME, 
                   EPSILON_START, EPSILON_FINAL,  MEAN_REWARD_BOUND, nA])

C = 1
A = 2
A0 = 1
MU = 1/2
min_price_tmp = 0
price_range_tmp = 2

#ENV = gym.make("CartPole-v1")
#ENV = ContBertrand()

    
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


def profit_n(action_n, nA, c, ai, aj, a0, mu, price_range, min_price):
    '''
    profit_n gives profits in the market after taking prices as argument
    INPUT
    action_n.....an np.array([]) containing two prices
    OUTPUT
    profit.......profit, an np.array([]) containing profits
    '''
    a = np.array([ai, aj])
    a_not = np.flip(a) # to obtain the other firm's a
      
    p = (price_range * action_n/nA) + min_price # minus 1 ? was a comment I left in the other file
    p_not = np.flip(p) # to obtain the other firm's p
    num = np.exp((a - p)/mu)
    denom = np.exp((a - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
    quantity_n = num / denom
          
    profit = quantity_n * (p-c)
    return(profit)

# Calculate Nash and monopoly profits using the following logic: 
profits = np.zeros((nA, nA))

for p1 in range(nA):
    for p2 in range(nA):
        profits[p1][p2] = profit_n(np.array([p1,p2]), nA, C, A, A, A0, MU, price_range_tmp, min_price_tmp)[0]

best_response = np.zeros((nA))
for p2 in range(nA):
    best_response[p2] = np.argmax(profits[:, p2])

best_response = np.vstack((best_response, np.arange(nA))).transpose()

Nash = best_response[:,0] == best_response[:,1]

NASH_ACTION = np.argmax(Nash)
NASH_PRICE = (price_range_tmp * NASH_ACTION/(nA-1)) + min_price_tmp # minus 1?
NASH_PROFIT = profit_n(np.array((NASH_ACTION, NASH_ACTION)), nA, C, A, A, A0, MU, price_range_tmp, min_price_tmp)
MIN_PROFIT = np.min(profits)
MAX_PROFIT = np.max(profits)
MAX_ACTION = np.argmax(best_response, axis = 0)[0] # highest action ever rational to take
MAX_PRICE = (price_range_tmp * MAX_ACTION/(nA-1)) + min_price_tmp # Circular?
MONOPOLY_ACTION = np.argmax(np.diag(profits))
MONOPOLY_PRICE = (price_range_tmp * MONOPOLY_ACTION/(nA-1)) + min_price_tmp
MONOPOLY_PROFIT = np.max(np.diag(profits)) # max profit w. constraint there's 1 price
#PROFIT_NASH = 0.1133853
#PROFIT_MONOPOLY = 0.1157638717582288
MIN_PRICE = 0.9 * NASH_PRICE
MAX_PRICE = 1.1 * MONOPOLY_PRICE
price_range = MAX_PRICE- MIN_PRICE
NREWS = 15

ECON_PARAMS = np.array([C, A, A0, MU, MIN_PRICE, price_range,
                        NASH_PROFIT, MONOPOLY_PROFIT, MAX_PROFIT, MIN_PROFIT,
                        NREWS])
    
def avg_profit_gain(avg_profit):
    '''
    avg_profit_gain() gives an index of collusion
    INPUT
    avg_profit......scalar. Mean profit over episodes.
    OUTPUT
    apg.............normalised value of the scalar
    '''
    apg = (avg_profit - NASH_PROFIT) / (MONOPOLY_PROFIT - NASH_PROFIT)
    return apg

def rew_to_int(reward):
    rewrange = MAX_PROFIT - MIN_PROFIT
    rewint = np.round(NREWS * (reward-MIN_PROFIT)/rewrange).astype(int)
    return(rewint)

def to_s(act, reward):
    '''

    '''
    return(act*NREWS + reward)


