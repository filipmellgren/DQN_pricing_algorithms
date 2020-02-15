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
import itertools
# Hyperparameters
MIN_PRICE = 1.4 # Prev: 1.5
PRICE_RANGE = 0.5 # Prev 0.3

# Functions
def act_to_price(action_n, nA):
    '''
    Converts discrete actions into prices
    This is currently wrong. Need to know 
    '''
    # TODO: currently wrong?
    price_n = (PRICE_RANGE * action_n/(nA-1)) + MIN_PRICE 

    return(price_n)

def demand(price_n, a_n, a0, mu):
    '''
    Calculates a demand for each firm, given prices.
    INPUT
    price_n.....a vector of prices
    a_n.........vector of product quality indices (vertical differentiation)
    a0..........scalar of inverse index of aggregate demand
    mu..........scalar of horizontal differentiation
    '''
    p = price_n
    a_not = np.flip(a_n) # to obtain the other firm's a
    p_not = np.flip(p) # to obtain the other firm's p
    num = np.exp((a_n - p)/mu)
    denom = np.exp((a_n - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
    quantity_n = num / denom
    return(quantity_n)
    
def profit(action_n, a0, mu, firm0, firm1, nA):
    '''
    profit_n gives profits in the market after taking prices as argument
    INPUT
    action_n.....an np.array([]) containing two actions (to be converted to prices)
    firmX, ......dictionaries with cost and quality information
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
    
    OUTPUT
    argmax(profits).the action that yields the highest profit given rivals action.
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
    # TODO: is this correct? I've found larger profits
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
    profits = profits0 + profits1.transpose() # Transpose because actions have to correspond
    act1,act2 = np.where(profits==profits.max())
    act1 = int(act1)
    act2 = int(act2)
    action_n = np.array([act1,act2])
    return(action_n)

def actions_dict(nA, a0, mu, firmlist1, firmlist2, action_type):
    '''
    Returns a dictionary of actions given two lists to be combined.
    The dictionary can be used as a reference, so given the dictionary and two
    firms, the corresponding actions (in a monopoly or equilibrium) can quickly
    be accessed.
    
    INPUT
    nA
    a0
    mu
    firmlistX......cartesian product of all available firms.
    action_type....either 'monopoly' or 'nash', what type of actions to be gen.
    '''
    actions_dict = {}
    
    if action_type == "monopoly":
        for element in itertools.product(firmlist1,firmlist2):
            firm0 = element[0]
            firm1 = element[1]
            action_array = monopoly_action(nA, a0, mu, firm0, firm1)
            actions_dict[str(element)] = action_array
    elif action_type == "nash":
        for element in itertools.product(firmlist1,firmlist2):
            firm0 = element[0]
            firm1 = element[1]
            action_array = nash_action(nA, a0, mu, firm0, firm1)
            actions_dict[str(element)] = action_array
    elif action_type == "colab":
        for element in itertools.product(firmlist1,firmlist2):
            firm0 = element[0]
            firm1 = element[1]
            action_array = colab_action(nA, a0, mu, firm0, firm1)
            actions_dict[str(element)] = action_array
    else:
        raise ValueError('action_type has to be either "monopoly", "nash", or "colab".')
    
    return(actions_dict)

# Max and mins
def max_profit(nA, a0, mu, firm0, firm1):
    '''
    Returns the highest achievable profit of a single firm that can be gained.
    '''
    max0 = np.amax(profit_matrix(nA, a0, mu, firm0, firm1))
    max1 = np.amax(profit_matrix(nA, a0, mu, firm1, firm0))
    max_profit = max(max0, max1)
    return(max_profit)
    
def min_profit(nA, a0, mu, firm0, firm1):
    '''
    Returns the lowest achievable profit of a single firm that can be gained.
    '''
    min0 = np.amin(profit_matrix(nA, a0, mu, firm0, firm1))
    min1 = np.amin(profit_matrix(nA, a0, mu, firm1, firm0))
    min_profit = min(min0, min1)
    return(min_profit)

# Colaboration action
def colab_action(nA, a0, mu, firm0, firm1, tol = 0.00):
    # TODO: How to define the collusive action?
    # cant be too high – then they could find a higher joint outcome
    # must lead to profits at least as large as Nash profits for both firms
    # TODO: take solution and test whether +1 for both leads to a profitable outcome
    '''
    Returns a colaborative action. An action is deemed colaborative if there is
    no higher action that could lead to a higher profit for both when the rival
    plays its colaborative action.
    '''
    action = nash_action(nA, a0, mu, firm0, firm1)
    profit_prev = profit(action, a0, mu, firm0, firm1, nA)
    profit_next = profit(action, a0, mu, firm0, firm1, nA)
    while (profit_next - profit_prev >= 0).all() and (action <= nA).all():
        profit_prev = profit_next
        action = action + 1
        profit_next = profit(action, a0, mu, firm0, firm1, nA)
    action = action - 1    
    return(action)

