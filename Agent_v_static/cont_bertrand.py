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

from config import HYPERPARAMS
from config import ECONPARAMS
from calc_nash_monopoly import profit
params = HYPERPARAMS['full_obs_NB']
eparams = ECONPARAMS['base_case']

#MIN_PRICE = eparams['min_price']
#MAX_PRICE = eparams['max_price']
#MIN_PROFIT = eparams['min_profit']
#MAX_PROFIT = eparams['max_profit']

#firm0 = eparams['firm0']
#firm1 = eparams['firm1']
A0 = eparams['a0']
MU = eparams['mu']
GAMMA = params['gamma']

nA = params['nA']
FRAMES = params['frames']

class ContBertrand(gym.Env):
    metadata = {'render.modes': ['human']} #?
    
    # useful blog post:
    # https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
    """
    This environment represents a continuous world with two agents. The agents 
    may set discrete prices which generates profits depending on the joint 
    behaviour.
      
    In principle, the environment is similar to CartPole with the difference 
    that rows and columns (the state) are prices in the previous period.
    
    The state is defined as:
    state = np.array([reward[0], action[0], reward[1], action[1], eps, frame,
    cost0, cost1, vert0, vert1])      
    """
    def __init__(self, firm0, firm1):
        # self.variables go here
        # States are defined as the profits, prices, and competition params
        # of both agents
        high_state = np.array([
            10,
            10,
            10,
            10,
            1,
            FRAMES,
            10,
            10,
            10,
            10])
    
        low_state = np.array([
                -10,
                0,
                -10,
                0,
                0,
                0,
                0,
                0,
                0,
                0])
    
        self.single_action_space = spaces.Discrete(nA) # Need this unconventional space to sample single agent actions from
        self.action_space = spaces.Discrete(nA*nA) # actually, number of combinations
        self.observation_space = spaces.Box(low_state, high_state,
                                            dtype=np.float32)
        self.seed()
        self.reset(firm0, firm1)
        
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action0, action1):
        # action made by the "meta agent", i.e. all market participants' joint action
        # Or, two actions packed into a vector
        action = np.array([action0, action1])
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action)) # TODO: make this work (threw assertionerror for valid actions)
        reward = profit(action, a0 = A0, mu = MU, firm0 = self.firm0, firm1 = self.firm1, nA = nA)
        self.state = np.array([reward[0], action[0], reward[1], action[1], self.cost0, self.cost1, self.vert0, self.vert1])
        if np.random.random() > GAMMA:
            done = True
        else:
            done = False
        return self.state, reward, done, {}
    
    def reset(self, firm0, firm1):
        # TODO: define better
        min_profit = 0
        min_price = 0
        max_profit = 0.3
        max_price = 2
        
        profit0 = self.np_random.uniform(low=min_profit, high=max_profit)
        price0 = self.np_random.uniform(low=min_price, high=max_price)
        profit1 = self.np_random.uniform(low=min_profit, high=max_profit)
        price1 = self.np_random.uniform(low=min_price, high=max_price)
        self.cost0 = firm0['cost']
        self.cost1 = firm1['cost']
        self.vert0 = firm0['quality']
        self.vert1 = firm1['quality']
        
        self.firm0 = firm0
        self.firm1 = firm1
        self.state = np.array([profit0, price0, profit1, price1, self.cost0, self.cost1, self.vert0, self.vert1]) 
        return np.array(self.state)

