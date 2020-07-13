#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:27:25 2020

@author: filip
"""

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

p_end = params['p_end']
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
    state = np.array([action[0], action[1]])      
    """
    def __init__(self, firm0, firm1, eparams):
        # self.variables go here
        # States are defined as the profits, prices, and competition params
        # of both agents
        high_state = np.array([
            nA,
            nA
            ])
    
        low_state = np.array([
                0,
                0
                ])
    
        self.single_action_space = spaces.Discrete(nA) # Need this unconventional space to sample single agent actions from
        self.action_space = spaces.Discrete(nA*nA)
        self.observation_space = spaces.Box(low_state, high_state,
                                            dtype=np.float32)
        self.seed()
        self.reset(firm0, firm1, eparams)
        
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action0, action1):
        # action made by the "meta agent", i.e. all market participants' joint action
        action = np.array([action0, action1])
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action)) # TODO: make this work (threw assertionerror for valid actions)
        a0 = A0 + np.random.normal(0, 1) * self.randomness
        reward = profit(action, a0 = a0, mu = MU, firm0 = self.firm0, firm1 = self.firm1, nA = nA)
        #subtract baseline reward and add randomness
        reward = reward - self.baseline
        # new state
        self.state = np.array([action[0], action[1]])
        if np.random.random() > (1-p_end):
            done = True
        else:
            done = False
        return self.state, reward, done, {}
    
    def reset(self, firm0, firm1, eparams):
        action0 = self.np_random.uniform(low=0, high=nA)
        action1 = self.np_random.uniform(low=0, high=nA)
        #self.vert0 = firm0['quality']
        #self.vert1 = firm1['quality']
        self.firm0 = firm0
        self.firm1 = firm1
        self.nash_act = eparams['nash_actions'][str((self.firm0, self.firm1))]
        self.baseline = profit(self.nash_act, A0, MU, self.firm0, self.firm1, nA)
        self.randomness = eparams['randomness'] * self.baseline 
        self.state = np.array([action0, action1]) 
        return np.array(self.state)

