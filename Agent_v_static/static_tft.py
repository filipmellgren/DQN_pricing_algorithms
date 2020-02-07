#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:51:55 2020

@author: Filip Mellgren
"""
import numpy as np
from calc_nash_monopoly import best_response
from config import HYPERPARAMS, ECONPARAMS
params = HYPERPARAMS['full_obs_NB']
eparams = ECONPARAMS['base_case']
nA = params['nA']
a0 = eparams['a0']
mu = eparams['mu']
firm0 = eparams['firm0']
firm1 = eparams['firm1']
rival_act_c = eparams['monopoly_action'][0] # Rival's monopoly action # redundant
rival_act_n = eparams['nash_action'][0] # Rival's nash action # redundant
nash_actions = eparams['nash_actions']
monopoly_actions = eparams['monopoly_actions']


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
            
