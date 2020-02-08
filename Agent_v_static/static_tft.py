#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:51:55 2020

@author: Filip Mellgren
"""
import numpy as np
from calc_nash_monopoly import best_response, profit
from config import HYPERPARAMS, ECONPARAMS
params = HYPERPARAMS['full_obs_NB']
eparams = ECONPARAMS['base_case']
nA = params['nA']
a0 = eparams['a0']
mu = eparams['mu']
#firm0 = eparams['firm0']
#firm1 = eparams['firm1']
#rival_act_c = eparams['monopoly_action'][0] # Rival's monopoly action # redundant
#rival_act_n = eparams['nash_action'][0] # Rival's nash action # redundant
nash_actions = eparams['nash_actions']
monopoly_actions = eparams['monopoly_actions']
colab_actions = eparams['colab_actions']

# Static Tit-For-Tat player
# TODO: update this so that it is built as a firm and then calculate collaborative and punishing actions
class Tft:
    '''
    Models a Tit-for-Tat playing agent.
    ATTRIBUTES
    self.last_profit....numeric value indicating last period profit
    self.action_c.......action integer indicating collaborative action
    self.action_p.......action integer indicating punishing action
    self.profit_c.......threshold for collaborative profit
    self.prob_cheat.....Probability to play BR against last observation.
    '''
    def __init__(self, profit_c, action_c, action_p, prob_cheat = 0):
        self.reset(profit_c, action_c, action_p, prob_cheat)
        
    def reset(self, profit_c, action_c, action_p, prob_cheat): 
        self.last_profit = profit_c # redundant
        self.action_c = action_c # redundant
        self.action_p = action_p# redundant
        self.profit_c = profit_c # redundant
        self.prob_cheat = prob_cheat # redundant
        self.total_pg = []
        return
    
    def act(self, rival_action, firm0, firm1):
        '''
        Returns an action of the static player based on the rival's action and chance.
        It is punishing if the rival cheated. Otherwise, with a certain
        probability it will try to cheat. If it doesn't cheat, it plays nice and
        takes the average action between best response and monopoly action
        INPUT: 
            rival_action....action of the rival observed in state
        OUTPUT:
            action..........an action 
        '''
        br = best_response(rival_action, nA, a0, mu, firm0, firm1)
        rival_colab_action = monopoly_actions[str((firm0, firm1))][0]
        own_colab_action = monopoly_actions[str((firm0, firm1))][1]
        
        if rival_action < rival_colab_action: # Punish.
            dist = rival_colab_action - rival_action # severeness of cheat
            action = min(own_colab_action, max(own_colab_action - dist + 1, br)) 
            return(action)
        else: # Colaborate
            return(own_colab_action)
            
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
    def __init__(self, nA, a0, mu, firm0, firm1, T, gamma):
        self.reset(nA, a0, mu, firm0, firm1, T, gamma)
        
    def reset(self, nA, a0, mu, firm0, firm1, T, gamma):
        self.b = 0 # begins in a cooperative phase
        self.W = 0 # Payoff balance initiates at 0
        self.T = T # Threshold for how much profit gain rival can gather before punishment
    
        self.c_act = colab_actions[str((firm0, firm1))][0]
        self.c_act1 = colab_actions[str((firm0, firm1))][1]
        self.d_act = nash_actions[str((firm0, firm1))][0] #best_response(self.c_act1, nA, a0, mu, firm0, firm1)
       
        self.a0 = a0
        self.mu = mu
        self.firm0 = firm0
        self.firm1 = firm1
        self.nA = nA
        #self.alpha
        self.total_pg = []
        act_colab = np.array([self.c_act, self.c_act1])
        self.profit1_cc = profit(act_colab, self.a0, self.mu, self.firm0, self.firm1, self.nA)[1]
        self.Q1_cc = gamma * self.profit1_cc / (1 - gamma) # infinite sum of discounted p_cc starting next period
        return
    
    def act(self, act1):
        meta_act = np.array([self.c_act, act1])
        r1 = profit(meta_act, self.a0, self.mu, self.firm0, self.firm1, self.nA)[1]
        
        deviation = r1 + self.Q1_cc - self.profit1_cc - self.Q1_cc
        
        if self.b == 0:
            act = self.c_act
            self.W = self.W + deviation
        else:
            act = self.d_act
            self.b = self.b - 1
            
        if self.W > self.T:
            self.b = 4#self.len_defect() # length of punishment period
            self.W = 0
        return(act)
            
    def len_defect():
        '''
        Calculate length of a punishment phase.
        
        profits under punishment - profits that would have accrued under collab
        
        has to be greater than 
        '''
        #TODO develop this
        return(4)
            
        
        
            
