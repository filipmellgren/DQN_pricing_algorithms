#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:19:49 2020

@author: filip
"""

import pandas as pd
import numpy as np
from calc_nash_monopoly import profit
from cont_bertrand import ContBertrand

def testing(firmlist, agent, params, eparams,saveto):
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
    OUTPUT:
        Function does not return anything; it does save a pandas data frame
        to file 'saveto' though. This file can be read R and be used to 
        produced graphical output.
    
    '''
    nA = params['nA']
    A0 = eparams['a0']
    MU = eparams['mu']
    firm0 = firmlist[0]
    firm1 = firmlist[0]
    env = ContBertrand(firm0, firm1)
    agent.net.eval()
    df = []
    
    # Make econ variables
    dict_key = str((firm0, firm1))
    nash_action = eparams['nash_actions'][dict_key]
    monopoly_action = eparams['monopoly_actions'][dict_key]
    nash_profit = profit(nash_action, A0, MU, firm0, firm1, nA)
    monopoly_profit = profit(monopoly_action, A0, MU, firm0, firm1, nA)
    
    # Initiate new env and amTFT agent
    s_next = env.reset(firm0, firm1)
    done = False
    frame_idx = 0
    firm_ix = 0
    for t in range(1, (len(firmlist)+1)*1000):
        if done:
            # Save episodal reward
            # TODO: expand set of firms
            firm0 = firmlist[firm_ix][0]
            firm1 = firmlist[firm_ix][1]
            firm_ix += 1
            # Make econ variables
            dict_key = str((firm0, firm1))
            nash_action = eparams['nash_actions'][dict_key]
            monopoly_action = eparams['monopoly_actions'][dict_key]
            nash_profit = profit(nash_action, A0, MU, firm0, firm1, nA)
            monopoly_profit = profit(monopoly_action, A0, MU, firm0, firm1, nA)
            
            # Initiate new env and amTFT agent
            s_next = env.reset(firm0, firm1)
            done = False
            
        frame_idx += 1
        epsilon = 0
        s = s_next
        
        action0 = agent.act(s[np.array([0,1,4,5])], epsilon)
        action1 = agent.act(s[np.array([2,3,4,5])], epsilon)
        s_next, reward_n, done, _ = env.step(action0, action1)
        done = False # Overwrite whatever may have come out before
        if frame_idx % 1000 == 0:
          done = True
        
        if reward_n is not None:
            reward = reward_n[0]
            pg = reward
            #pg = profit_gain(reward, nash_profit, colab_profit)[0] # important to index here
            agent.total_pg.append(pg)
            df.append({'vertdiff0': firm0['quality'], 'vertdiff1': firm1['quality'], 
                   'reward0': reward_n[0], 'reward1': reward_n[1], 'nash0': nash_profit[0],
                   'nash1': nash_profit[1], 'mon0': monopoly_profit[0], 'mon1': monopoly_profit[1]})
    
    df = pd.DataFrame(df)
    df.to_csv(saveto)
    return
