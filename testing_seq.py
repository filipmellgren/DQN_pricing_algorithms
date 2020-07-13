#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:19:49 2020

@author: filip
"""

import pandas as pd
import numpy as np
from calc_nash_monopoly import profit, actions_dict
from cont_bertrand import ContBertrand
import itertools

def testing(firmlist, agent, params, eparams,saveto, cheat):
    '''
    testing tests a given agent and saves a dataframe to a file
    An instance of class Agent is tested in an economic environment determined
    by the firmlist, params, and eparams. Testng occurs by exposing the agent
    to all combinations of possible firms present in the firmlist argument and 
    results are measured by recording the profit, as well as important values
    used to evaluate how good this profit is. The recorded values are appended
    to a dictionary that is then turned into a pandas data frame that I export
    to a .csv file specified by the argument saveto
    INPUT:
        firmlist...a list of combinations of firms
        agent......object of class Agent that has an already trained network
        params.....dict of hyperparameters relating to code mechanics
        eparams....dict of hyperparameters relating to the economic env
        saveto.....string specifying to where dataframe should be saved.
        cheat......boolean. True if studying irf of a cheat. 
    OUTPUT:
        Function does not return anything; it does save a pandas data frame
        to file 'saveto' though. This file can be read R and be used to 
        produced graphical output.
    
    '''

    nA = params['nA']
    A0 = eparams['a0']
    MU = eparams['mu']
    
    nash_actions = actions_dict(nA, A0, MU, firmlist, firmlist, "nash")
    mon_actions = actions_dict(nA, A0, MU, firmlist, firmlist, "monopoly")
    
    firmprod = itertools.product(firmlist, firmlist)
    firmlist_cart = []
    for element in firmprod:
        firmlist_cart.append(element)
    

    firm0 = firmlist_cart[0][0]
    firm1 = firmlist_cart[0][0]
    env = ContBertrand(firm0, firm1, eparams)
    agent.net.eval()
    df = []
    
    # Make econ variables
    dict_key = str((firm0, firm1))
    nash_action = nash_actions[dict_key]
    monopoly_action = mon_actions[dict_key]
    nash_profit = profit(nash_action, A0, MU, firm0, firm1, nA)
    monopoly_profit = profit(monopoly_action, A0, MU, firm0, firm1, nA)
    
    # Initiate new env and amTFT agent
    s_next = env.reset(firm0, firm1, eparams)
    done = False
    frame_idx = 0
    firm_idx = 0
    epsilon = 0
    obs_firm0 = np.array([0,1])
    obs_firm1 = np.array([1,0])
    
    # For sequentiality, I need to initiate action1
    action1 = agent.act(s_next[obs_firm1], 0)
    action0 = agent.act(s_next[obs_firm0], 0)
    for t in range(1, (len(firmlist_cart)+1)*1000):
        if done:
            # Save episodal reward
            firm0 = firmlist_cart[firm_idx][0]
            firm1 = firmlist_cart[firm_idx][1]
            firm_idx += 1
            # Make econ variables
            dict_key = str((firm0, firm1))
            nash_action = nash_actions[dict_key]
            monopoly_action = mon_actions[dict_key]
            nash_profit = profit(nash_action, A0, MU, firm0, firm1, nA)
            monopoly_profit = profit(monopoly_action, A0, MU, firm0, firm1, nA)
            
            # Initiate new env
            s_next = env.reset(firm0, firm1, eparams)
            done = False
            
        frame_idx += 1
        s = s_next
        
        # Sequentiality of action choices
        if frame_idx % 2 == 0:
            # Update only agent0's action
            action0 = agent.act(s[obs_firm0], epsilon)
        else:
            # Update only agent1's action
            action1 = agent.act(s[obs_firm1], epsilon)
        
        if cheat and frame_idx == 100:
            action0 = 1
        
        s_next, reward_n, done, _ = env.step(action0, action1)
        done = False # Overwrite whatever may have come out before
        if frame_idx % 1000 == 0:
          done = True
        
        if reward_n is not None:
            df.append({'vertdiff0': firm0['quality'], 'vertdiff1': firm1['quality'], 
                   'reward0': reward_n[0], 'reward1': reward_n[1], 
                   'nash0': nash_profit[0], 'nash1': nash_profit[1], 
                   'mon0': monopoly_profit[0], 'mon1': monopoly_profit[1],
                   'index': frame_idx, 'firm_index': firm_idx})
    
    df = pd.DataFrame(df)
    df.to_csv(saveto)
    return
