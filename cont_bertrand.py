#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:27:25 2020

@author: filip
"""

# TODO: turn into a continuous pbservation space based on cartpole env

# Discrete case
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from config import ECON_PARAMS
from config import PARAMS
from config import profit_n

C = ECON_PARAMS[0]
AI = ECON_PARAMS[1]
AJ = ECON_PARAMS[1]
A0 = ECON_PARAMS[2]
MU = ECON_PARAMS[3]
MIN_PRICE = ECON_PARAMS[4]
PRICE_RANGE = ECON_PARAMS[5]
MAX_REWARD = ECON_PARAMS[8]
MIN_REWARD = ECON_PARAMS[9]
NREWS = ECON_PARAMS[10]

MIN_PRICE = ECON_PARAMS[4]
price_range = ECON_PARAMS[5]
MAX_PRICE = MIN_PRICE + price_range
NASH_PROFIT = ECON_PARAMS[6]
NASH_PROFIT = NASH_PROFIT[0]
MONOPOLY_PROFIT = ECON_PARAMS[7]

nA = PARAMS[10].astype(int) # Number of unique prices

class ContBertrand(gym.Env):
    metadata = {'render.modes': ['human']} #?
    
    # useful blog post:
    # https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
    """
    This environment represents a discrete world with two agents. The agents 
    may set prices which generates profits depending on the joint behaviour.
      
    In principle, the environment is similar to FrozenLake with the difference 
    that rows and columns (the state) are prices in the previous period.
      
    Inherits from discrete.Discrete which:
          
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
    P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self):
        # self.variables go here
        
        # States are defined as the profits and prices of both agents
        high_state = np.array([
            MONOPOLY_PROFIT*1.5,
            MAX_PRICE,
            MONOPOLY_PROFIT*1.5,
            MAX_PRICE])
    
        low_state = np.array([
                0,
                MIN_PRICE,
                0,
                MIN_PRICE])
        self.single_action_space = spaces.Discrete(nA) # Need this unconventional space to sample single agent actions from
        self.action_space = spaces.Discrete(nA*nA) # actually, number of combinations
        self.observation_space = spaces.Box(low_state, high_state,
                                            dtype=np.float32)
        self.seed()
        self.reset(NASH_PROFIT, MIN_PRICE, MONOPOLY_PROFIT, MAX_PRICE)
        
        def to_a(act1, act2, nA):
            '''
            Takes two discrete actions and turns it into a meta action
            '''
            action = act1*nA + act2
            return(action)
        
        
# =============================================================================
#         nrow = nA # Number of own possible actions, last state
#         ncol = nA
#         nS = nrow * nA
#         isd = np.zeros(nS)
#         P = {s : {a : [] for a in range(nA*nA)} for s in range(nS)} # Takes both actions into account
# =============================================================================

# =============================================================================
#         def to_s(row, col):
#             '''
#             enumerates row, col combo to a state. the row and col can be thought
#             of as action and profit in the last period in this application.
#             '''
#             return(row*ncol + col)
# 
#         for s in range(nS): # TODO: Correct?
#             for action1 in range(nA): # TODO: can this be simplified?
#                 for action2 in range(nA):
#                     a = to_s(action1, action2)
#                     action_n = np.array([action1, action2])
#                     li = P[s][a]
#                     reward_n = profit_n(action_n, nA, C, AI, AJ, A0, MU, PRICE_RANGE, MIN_PRICE)
#                     newstate = to_s(action1, action2) # new env state is determined by what they did in the last period. TODO: is it even important whatexaclty it is?
#                     done = False # No need to update done at init – my stopping rule does not depend on state
#                     # Here, P[s][a] is not updated
#                     li.append((1.0, newstate, reward_n, done)) # Why does it not need "P[s][a].append"?
# =============================================================================
                    # Here, P[s][a] is updated
       #? super(DiscrBertrand, self).__init__(nS, nA, P, isd)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action0, action1):
        # action made by the "meta agent", i.e. all market participants' joint action
        # Or, two actions packed into a vector
        action = np.array([action0, action1])
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action)) # TODO: make this work (threw assertionerror for valid actions)
        #state = self.state
        reward = profit_n(action)
        self.state = np.array([reward[0], action[0], reward[1], action[1]])
        done = bool(False) # TODO: how to define this? Might want to include a counter in the environment
        return self.state, reward, done, {}
    
    def reset(self, nash_profit, nash_price, monopoly_profit, monopoly_price):
        profit0 = self.np_random.uniform(low=nash_profit, high=monopoly_profit)
        price0 = self.np_random.uniform(low=nash_price, high=monopoly_price)
        profit1 = self.np_random.uniform(low=nash_profit, high=monopoly_profit)
        price1 = self.np_random.uniform(low=nash_price, high=monopoly_price)
        self.state = np.array([profit0, price0, profit1, price1]) 
        return np.array(self.state)

