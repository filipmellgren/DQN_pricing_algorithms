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
from config import avg_profit_gain
params = HYPERPARAMS['full_obs_NB']
eparams = ECONPARAMS['base_case']


MIN_PRICE = eparams['min_price']
MAX_PRICE = eparams['max_price']
MIN_PROFIT = eparams['min_profit']
MAX_PROFIT = eparams['max_profit']
NASH_PROFIT = eparams['nash_profit'][0]
MONOPOLY_PROFIT = eparams['monopoly_profit'][0]

firm0 = eparams['firm0']
firm1 = eparams['firm1']
A0 = eparams['a0']
MU = eparams['mu']

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
    state = np.array([reward[0], action[0], reward[1], action[1], eps, frame])      
    """
    def __init__(self):
        # self.variables go here
        # States are defined as the profits and prices of both agents
        high_state = np.array([
            MAX_PROFIT,
            MAX_PRICE,
            MAX_PROFIT,
            MAX_PRICE,
            1,
            FRAMES])
    
        low_state = np.array([
                MIN_PROFIT,
                MIN_PRICE,
                MIN_PROFIT,
                MIN_PRICE,
                0,
                0])
    
        self.single_action_space = spaces.Discrete(nA) # Need this unconventional space to sample single agent actions from
        self.action_space = spaces.Discrete(nA*nA) # actually, number of combinations
        self.observation_space = spaces.Box(low_state, high_state,
                                            dtype=np.float32)
        self.seed()
        self.reset()
        
        def to_a(act1, act2, nA): # TODO: unused
            '''
            Takes two discrete actions and turns it into a meta action
            '''
            action = act1*nA + act2
            return(action)
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action0, action1, eps, frame):
        # action made by the "meta agent", i.e. all market participants' joint action
        # Or, two actions packed into a vector
        action = np.array([action0, action1])
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action)) # TODO: make this work (threw assertionerror for valid actions)
        #state = self.state
        reward = profit(action, a0 = A0, mu = MU, firm0 = firm0, firm1 = firm1, nA = nA)
        #reward = avg_profit_gain(reward)
        self.state = np.array([reward[0], action[0], reward[1], action[1], eps, frame])
        done = bool(False) # TODO: how to define this? Might want to include a counter in the environment
        return self.state, reward, done, {}
    
    def reset(self, nash_profit = NASH_PROFIT, nash_price = MIN_PRICE, monopoly_profit = MONOPOLY_PROFIT, monopoly_price = MAX_PRICE):
        profit0 = self.np_random.uniform(low=nash_profit, high=monopoly_profit)
        price0 = self.np_random.uniform(low=nash_price, high=monopoly_price)
        profit1 = self.np_random.uniform(low=nash_profit, high=monopoly_profit)
        price1 = self.np_random.uniform(low=nash_price, high=monopoly_price)
        eps = 1
        idx = 0
        self.state = np.array([profit0, price0, profit1, price1, eps, idx]) 
        return np.array(self.state)

