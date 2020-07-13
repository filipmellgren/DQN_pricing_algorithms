#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:58:09 2020

@author: filip
"""
import numpy as np
from config import ECONPARAMS
econ_params = ECONPARAMS['base_case']
from calc_nash_monopoly import profit, nash_action, monopoly_action
import itertools
import pandas as pd

a0 = econ_params['a0']
mu = econ_params['mu']
nA = 7
firm0 = {'cost': 1, 'quality': 2}
firm1 = {'cost': 1, 'quality': 2}

acts = itertools.product(np.arange(nA), np.arange(nA))
profit_df = np.zeros((nA*nA,5))
ix = 0
nash_act = nash_action(nA, a0, mu, firm0, firm1)
monopoly_act = monopoly_action(nA, a0, mu, firm0, firm1)
nash_profit = profit(nash_act, a0, mu, firm0, firm1, nA)[0]
monopoly_profit = profit(monopoly_act, a0, mu, firm0, firm1, nA)[0]

for act in acts:
    action = np.array(act)
    profit_0 = profit(action, a0, mu, firm0, firm1, nA)[0]
    profit_df[ix][0] = action[0]
    profit_df[ix][1] = action[1]
    profit_df[ix][2] = profit_0
    profit_df[ix][3] = nash_profit
    profit_df[ix][4] = monopoly_profit
    ix = ix + 1

df = pd.DataFrame(profit_df)
df.to_csv("actact_profit.csv")