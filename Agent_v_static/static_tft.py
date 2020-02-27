#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:51:55 2020

@author: Filip Mellgren
"""
import numpy as np
from calc_nash_monopoly import profit
from config import HYPERPARAMS, ECONPARAMS
params = HYPERPARAMS['full_obs_NB']
eparams = ECONPARAMS['base_case']
nA = params['nA']
a0 = eparams['a0']
mu = eparams['mu']
punishlen = params['punishlen']
nash_actions = eparams['nash_actions']
monopoly_actions = eparams['monopoly_actions']
colab_actions = eparams['colab_actions']

class Amtft:
    '''
    Approximate Markov for Tit For Tat (amTFT)
    This agent is based on Lerer & Peysakhovic (2018)*. It is designed to
    fullfill a number of key properties of an agent facing social dilemmas.    
    
    Every time the other firm cheats, the amTFT agent takes record and bases
    its actions on this record. A cheating opponent gets punished whereas a 
    collaborative opponent is treated with cooperative actions.
    
    * "Maintaining cooperation in complex social dilemmas using deep 
    reinforcement learning"
    '''
    def __init__(self, nA, a0, mu, firm0, firm1, gamma):
        self.reset(nA, a0, mu, firm0, firm1, gamma)
        
    def reset(self, nA, a0, mu, firm0, firm1, gamma):
        self.b = 0 # begins in a cooperative phase
        self.W = 0 # Payoff balance initiates at 0
    
        self.c_act = colab_actions[str((firm0, firm1))][0]
        self.c_act1 = colab_actions[str((firm0, firm1))][1]
        self.d_act = nash_actions[str((firm0, firm1))][0] 
        self.d_act1 = nash_actions[str((firm0, firm1))][1] 
        
        self.a0 = a0
        self.mu = mu
        self.firm0 = firm0
        self.firm1 = firm1
        self.nA = nA
        self.total_pg = []
        act_colab = np.array([self.c_act, self.c_act1])
        cheat_act = np.array([self.c_act, self.d_act1])
        self.profit1_cc = profit(act_colab, self.a0, self.mu, self.firm0, self.firm1, self.nA)[1]
        self.Q1_cc = gamma * self.profit1_cc / (1 - gamma) # infinite sum of discounted p_cc starting next period
        
        self.profit1_cd = profit(cheat_act, self.a0, self.mu, self.firm0, self.firm1, self.nA)[1]
        self.T = 0.1*(self.profit1_cd - self.profit1_cc) # Threshold for how much profit gain rival can gather before amTFT punishes
        self.punishlen = punishlen
        return
    
    def act(self, act1):
        meta_act = np.array([self.c_act, act1])
        r1 = profit(meta_act, self.a0, self.mu, self.firm0, self.firm1, self.nA)[1]
        
        deviation = r1 + self.Q1_cc - self.profit1_cc - self.Q1_cc
        deviation = max(deviation, 0) # no negative credits will be rewarded
        
        if self.b == 0:
            act = self.c_act
            self.W = self.W + deviation
        else:
            act = self.d_act
            self.b = self.b - 1
            
        if self.W > self.T:
            self.b = self.punishlen
            self.W = 0
        return(act)