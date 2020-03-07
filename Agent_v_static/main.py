#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 09:44:07 2020

@author: filip
"""
import numpy as np
from config import HYPERPARAMS, ECONPARAMS
from training import training
from calc_nash_monopoly import actions_dict
from testing import testing
params = HYPERPARAMS['full_obs_NB']
eparams = ECONPARAMS['base_case']

# Economic parameters
MEAN_C = 1
MEAN_Q = 2

A0 = 1
MU = 1/2
nA = params['nA']
grid = nA # Higher values gives better approximation of nash/monopoly-profits

# TRAINING SETUP #####################################################
FIRMLIST = []
diff_range = np.linspace(MEAN_Q*0.9,MEAN_Q*1.1,5)

for c in [MEAN_C]:
    for q in diff_range:
        q = round(q, 2)
        FIRMLIST.append({'cost': c, 'quality': q})

NASH_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "nash")
MONOPOLY_ACTIONS = actions_dict(nA, A0, MU, FIRMLIST, FIRMLIST, "monopoly")

eparams = {
        'firmlist': FIRMLIST,
        'a0': A0,
        'mu': MU,
        'nash_actions': NASH_ACTIONS,
        'monopoly_actions': MONOPOLY_ACTIONS
        }

##### Takes a long time to run #######################################
agent8 = training(params, eparams, 8)
agent16 = training(params, eparams, 16)
agent32 = training(params, eparams, 32)
######################################################################

# TESTING ############################################################
# Same environment
testing(FIRMLIST, agent8, params, eparams,'ag_v_ag_netsize8.csv')
testing(FIRMLIST, agent16, params, eparams,'ag_v_ag_netsize16.csv')
testing(FIRMLIST, agent32, params, eparams,'ag_v_ag_netsize32.csv')

# Changed firmlist      
FIRMLIST_test = []
diff_range = np.linspace(MEAN_Q*0.85,MEAN_Q*1.15,15)
for c in [MEAN_C]:
    for q in diff_range:
        q = round(q, 2)
        FIRMLIST_test.append({'cost': c, 'quality': q})

