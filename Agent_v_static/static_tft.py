#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:51:55 2020

@author: Filip Mellgren
"""
import numpy as np
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
        self.last_profit = profit_c
        self.action_c = action_c
        self.action_p = action_p
        self.profit_c = profit_c
        self.prob_cheat = prob_cheat
        return
    
    def act(self, rival_action):
        '''
        INPUT: 
            rival_action....action of the rival
        '''
        if self.prob_cheat < np.random.random():
            return(0) # TODO: add this 
        elif self.last_profit >= 0.95*self.profit_c:
            return(self.action_c)
        else:
            action = round((self.action_c + rival_action)/2) # try to change to Nash
            return(action)