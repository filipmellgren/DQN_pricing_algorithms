#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:38:21 2020

@author: filip

Configuration of parameters

Nash price: price from which each agent has no incentive to deviate from.
    No agent can be made better off by a unilateral change in price from a N.E.

Monopoly price: price that maximises the joint profits.
    Because of symmetry, the price is the same for both firms. 
    If not symmetric, the two firms maximise joint profit and split it equally.
"""
import numpy as np
# Hyperparameters
#nA = 500 # Number of actions, only used to gain a fine accuracy of the values
MIN_PRICE = 1.5
PRICE_RANGE = 0.3

# Functions
def act_to_price(action_n, nA):
    '''
    Converts discrete actions into prices
    This is currently wrong. Need to know
    '''
    price_n = (PRICE_RANGE * action_n/(nA-1)) + MIN_PRICE 

    return(price_n)

def demand(price_n, a0, ai, aj, mu):
    '''
    Calculates a demand for each firm, given prices
    '''
    p = price_n
    a = np.array([ai, aj])
    a_not = np.flip(a) # to obtain the other firm's a
    p_not = np.flip(p) # to obtain the other firm's p
    num = np.exp((a - p)/mu)
    denom = np.exp((a - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
    quantity_n = num / denom
    return(quantity_n)
    
def profit(action_n, a0, ai, aj, mu, c, nA):
    '''
    profit_n gives profits in the market after taking prices as argument
    INPUT
    action_n.....an np.array([]) containing two prices
    OUTPUT
    profit_n.......profit, an np.array([]) containing profits
    '''
    price_n = act_to_price(action_n, nA)
    quantity_n = demand(price_n, a0, ai, aj, mu)          
    profit_n = quantity_n * (price_n-c)
    return(profit_n)
    
# NASH 
def nash_action(nA, a0, ai, aj, mu, c):
    '''
    Calculates a nash action.
    Does not assume symmetry.
    First, the function calculates profits given all action combinations.
    Then, the function calculates the best response for all values of the rival's
    prices.
    Finally, the function finds the Nash equilibrium by checking whether the
    best response to an action is the same as the action.
    INPUT
    nA...........Number of actions. Higher value gives more accurate output
    OUTPUT
    nash_action..Actions that correspond to a Nash equilibrium
    '''
    # Profits
    profits = np.zeros((nA, nA))
    for a1 in range(nA):
        for a2 in range(nA):
            action = np.array([a1, a2])
            profits[a1][a2] = profit(action, a0, ai, aj, mu, c, nA)[0]
    # Best response firm 1 to any price of firm 2
    br = np.zeros((nA))
    for a2 in range(nA):
        br[a2] = np.argmax(profits[:, a2])
    # Best reponse firm 2 to any best response of firm 1
    br1 = np.zeros((nA))
    for a1 in range(nA):
        br1[a1] = np.argmax(profits[:, a1])
    
    br = np.vstack((br,br1, np.arange(nA))).transpose()
    # NE  action is the first firm's BR when the BR of the rival corresponds to 
    # the action the first firm reacted to.
    is_nash = br[:,1] == br[:,2]
    nash_action1 = np.argmax(is_nash)
    nash_action2 = int(br1[nash_action1])
    nash_action_n = np.array([nash_action1, nash_action2])
    return(nash_action_n)

# MONOPOLY
def monopoly_action(nA, a0, ai, aj, mu, c):
    '''
    Calculates the fully collusive actions.
    Assumes symmetry.
    Monopoly actions are defined as the pair of actions that jointly maximise
    total profits. 
    INPUT:
        nA........ Number of actions, higher for higher accuracy
    OUTPUT:
        action_n.. The collusive actions. np array
    '''
    profits = np.zeros((nA, nA))
    for a1 in range(nA):
       for a2 in range(nA):
           action = np.array([a1, a2])
           profits[a1][a2] = profit(action, a0, ai, aj, mu, c, nA)[0]
    # Add profits and t(profits) to get sum of an action pair
    profits = profits + profits.transpose()
    act1,act2 = np.where(profits==profits.max())
    act1 = int(act1)
    act2 = int(act2)
    action_n = np.array([act1,act2])
    return(action_n)