#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:05:26 2020

@author: filip
"""
from dqn_model import Net
import pandas as pd
import torch
import numpy as np

def q_values(params, filepath, state):
    '''
    q_values returns Q values of each action in a df, given a state and network
    '''
    nA = 7 #params['nA']
    dO_a = 2#params['dO_a']
    nodes = params['nodes']
    
    net = Net(dO_a,nA, nodes).to("cpu")
    net.load_state_dict(torch.load(filepath)['agent_state_dict'])
    state_v = torch.Tensor(state).to("cpu")
    q_vals_v = net(state_v)
    qlist = []
    for q in q_vals_v:
        qlist.append(q.data.numpy())
    #
    return(qlist)
    
from config import HYPERPARAMS
params = HYPERPARAMS['full_obs_NB']
# =============================================================================
# action0 = 1
# action1 = 1
# state = np.array([action0, action1])
# qlist = q_values(params,"3499993_0checkpoint.pt",state)
# df = pd.DataFrame(qlist)
# df.to_csv("Q_values.csv")
# =============================================================================

import itertools
nA = 7
acts = itertools.product(np.arange(nA), np.arange(nA))
q_value_mat = np.zeros([nA*nA,nA+2])
ix = 0
for act in acts:
    state = np.array(act)
    qlist = q_values(params,"3499993_0checkpoint.pt",state)
    qlist.append(state[0])
    qlist.append(state[1])
    q_value_mat[ix] = qlist
    ix = ix + 1
    
df = pd.DataFrame(q_value_mat)
df.to_csv("Q_values.csv")

##############################
import itertools
nA = 7
cont_acts = np.random.uniform(0,6,10_000)
acts = itertools.product(np.arange(nA), cont_acts)
q_value_mat = np.zeros([nA*len(cont_acts),nA+2])
ix = 0
for act in acts:
    state = np.array(act)
    qlist = q_values(params,"3499993_0checkpoint.pt",state)
    qlist.append(state[0])
    qlist.append(state[1])
    q_value_mat[ix] = qlist
    ix = ix + 1
    
df = pd.DataFrame(q_value_mat)
df.to_csv("Q_values_cont.csv")