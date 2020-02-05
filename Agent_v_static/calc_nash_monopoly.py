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

def demand(price_n, a_n, a0, mu):
    '''
    Calculates a demand for each firm, given prices
    '''
    p = price_n
    a_not = np.flip(a_n) # to obtain the other firm's a
    p_not = np.flip(p) # to obtain the other firm's p
    num = np.exp((a_n - p)/mu)
    denom = np.exp((a_n - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
    quantity_n = num / denom
    return(quantity_n)
 # TODO: does profit take action or price?
    
def profit(action_n, a0, mu, firm0, firm1, nA):
    '''
    profit_n gives profits in the market after taking prices as argument
    INPUT
    action_n.....an np.array([]) containing two actions (to be converted to prices)
    firmX, .dictionaries with cost and quality information
    OUTPUT
    profit_n.......profit, an np.array([]) containing profits
    '''
    c_n = np.array([firm0['cost'], firm1['cost']])
    a_n = np.array([firm0['quality'], firm1['quality']])
    price_n = act_to_price(action_n, nA)
    quantity_n = demand(price_n, a_n, a0, mu)          
    profit_n = quantity_n * (price_n-c_n)
    return(profit_n)

def profit_matrix(nA, a0, mu, firm0, firm1):
    '''
    Returns a matrix of possible profits in the game.
    
    INPUT
    nA..............number of actions available
    a0..............outside option value
    mu..............an economic parameter
    firmX...........a dictionary with firm info (cost and quality)
    OUTPUT 
    A matrix of potential profits of a firm.
    '''
    profits = np.zeros((nA, nA))
    for a1 in range(nA):
       for a2 in range(nA):
           action = np.array([a1, a2])
           profits[a1][a2] = profit(action, a0, mu, firm0, firm1, nA)[0]
    return(profits)

# NASH     
def best_response(rival_action, nA, a0, mu, firm0, firm1):
    '''
    Gives best response to an action
    It does this by looping over all possible responses/actions and selects the
    action that gives the highest profit.
    
    INPUT
    rival_action....the action to respond to
    nA..............number of actions available
    a0..............outside option value
    mu..............an economic parameter
    firmX...........a dictionary with firm info (cost and quality)
    '''
    # Profits
    profits = np.zeros((nA, 1))
    for action in range(nA):
        meta_action = np.array([int(action), int(rival_action)])
        profits[int(action)] = profit(meta_action, a0, mu, firm0, firm1, nA)[0]
    # Best response
    return(np.argmax(profits))

def nash_action(nA, a0, mu, firm0, firm1):
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
    # Best response firm 1 to any price of firm 2
    br0 = np.zeros((nA))
    for a2 in range(nA):
        br0[a2] = best_response(a2, nA, a0, mu, firm0, firm1)
    # Best reponse firm 2 to any best response of firm 1
    br1 = np.zeros((nA))
    for a1 in range(nA):
        br1[a1] = best_response(a1, nA, a0, mu, firm1, firm0) # Note they are flipped
    
    # NE  action is the first firm's BR when the BR of the rival corresponds to 
    # the action the first firm reacted to.
    # Iteration: 
    #   Initiate player 0's action at 0 (arbitrary assuming unique equilibrium)
    #   Look at best response to action 0 of player 1
    #   Let player 0 play best response to best response of player 1
    #   Break loop if player 0's best response is the same as the action
    #   player 1 responded to. Otherwise continue and update player 0's action
    #   to the best response player 1 choose.
    
    action0 = 0
    action0_n = 1
    while action0 != action0_n:
        action0 = action0_n
        action1 = int(br1[action0])
        action0_n = int(br0[action1])
        
    nash_action_n = np.array([action0, action1])
    return(nash_action_n)

# MONOPOLY
def monopoly_action(nA, a0, mu, firm0, firm1):
    '''
    Calculates the fully collusive actions.
    Allows assymetric firms.
    Monopoly action is defined as the pair of actions that jointly maximise
    total profits. 
    INPUT:
        nA........ Number of actions, higher for higher accuracy
    OUTPUT:
        action_n.. The collusive actions. np array
    '''
    profits0 = profit_matrix(nA, a0, mu, firm0, firm1)
    profits1 = profit_matrix(nA, a0, mu, firm1, firm0) # Note the swapped position of the firms
    # Add profits to get sum of overall profits
    profits = profits0 + profits1
    act1,act2 = np.where(profits==profits.max())
    act1 = int(act1)
    act2 = int(act2)
    action_n = np.array([act1,act2])
    return(action_n)

# Max and mins
def max_profit(nA, a0, mu, firm0, firm1):
    max0 = np.amax(profit_matrix(nA, a0, mu, firm0, firm1))
    max1 = np.amax(profit_matrix(nA, a0, mu, firm1, firm0))
    max_profit = max(max0, max1)
    return(max_profit)
    
def min_profit(nA, a0, mu, firm0, firm1):
    min0 = np.amin(profit_matrix(nA, a0, mu, firm0, firm1))
    min1 = np.amin(profit_matrix(nA, a0, mu, firm1, firm0))
    min_profit = min(min0, min1)
    return(min_profit)