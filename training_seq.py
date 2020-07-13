#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:45:43 2020

@author: filip
"""

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from dqn_model import Net
from agent import Agent1 as Agent
from experience_buffer import ExperienceBuffer
from config import calc_loss, profit_gain
from calc_nash_monopoly import profit
from cont_bertrand import ContBertrand
import collections
#import os.path
#from static_tft import Amtft
import itertools
#import pandas as pd

    #params = HYPERPARAMS['full_obs_NB']
   # eparams = ECONPARAMS['base_case']  

def training(params, eparams, netsize):
    '''
    training is the main method used to train an agent
    Before running a main training loop, several objects need to be initialised
    First, a neural network is initialised based on imports from the dgn_model
    file. Then, a single agent and an environment are initialised. In this 
    training code, the agent plays against itself, but is unaware of this fact.
    
    In the training loop, the agent selects optimal actions with probability 
    epsilon (element in params), and randomises otherwise. It then observes
    transition dynamics and receives rewards for these actions. These 
    transitions are learned by the neural network provided to the agent.
    
    INPUT:
        params....A dictionary of hyperparameters
        eparams...Another dictionary of parameters determining the economic env
        netsize...the size of the two hidden layers in the neural network
            this is an important parameter that I want to present results for
            when it is being varied
    OUTPUT:
        agent...an instance of class Agent that is equipped with a trained
        neural network. agent can then be tested by calling the file testing.py
    
    '''
    Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
    BATCH_SIZE = params['batch_size']
    REPLAY_SIZE = params['replay_size']
    REPLAY_START_SIZE = params['replay_start_size']
    LEARNING_RATE = params['learning_rate']
    SYNC_TARGET_FRAMES = params['sync_target_frames']
    EPSILON_DECAY_LAST_FRAME = params['epsilon_decay_last_frame']
    EPSILON_START = params['epsilon_start']
    EPSILON_FINAL = params['epsilon_final']
    nA = params['nA']
    dO_a = params['dO_a']
    FRAMES = params['frames']
    NODES = params['nodes']
    SEED = params['seed']
    PATH = params['path']
    GAMMA = params['gamma']
    A0 = eparams['a0']
    MU = eparams['mu']
    FIRMLIST = eparams['firmlist']
    
    NODES = netsize
    
    RANDOMNESS = str(eparams['randomness'])
    
    print(params)
    # PyTorch setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu') # CPU when GPU is not available <-> device agnostic
    
    torch.manual_seed(SEED)
    if use_cuda:
      torch.cuda.manual_seed(SEED)
    
    # Neural network model:
    net = Net(dO_a,nA, NODES).to(device)
    print(net)
    tgt_net = Net(dO_a,nA, NODES).to(device) # Prediction target.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) 
    buffer = ExperienceBuffer(REPLAY_SIZE)
    
    # Reinforcement learning environment
    firm0 = random.sample(FIRMLIST, 1)[0]
    firm1 = random.sample(FIRMLIST, 1)[0]
    env = ContBertrand(firm0, firm1, eparams)
    
    # RL agent
    agent = Agent(env, buffer, net, tgt_net, optimizer)
    
    # Write output statistics to tensorboard
    writer = SummaryWriter(comment = "-")
    
    # Initialize variables
    env.seed(SEED) # TODO; is this used?
    torch.manual_seed(SEED)
    frame_idx = 0
    ep_idx = 0
    epsilon = EPSILON_START
    
    firmlist_cartesian = itertools.product(FIRMLIST, FIRMLIST)
    firmlist = []
    for element in firmlist_cartesian:
      firmlist.append(element)
      
    # Training – Main loop
    firm0 = random.sample(FIRMLIST, 1)[0]
    firm1 = random.sample(FIRMLIST, 1)[0]
    
    obs_firm0 = np.array([0,1])
    obs_firm1 = np.array([1,0])
    
    # Make econ variables
    #dict_key = str((firm0, firm1))
    #nash_action = eparams['nash_actions'][dict_key]
    #monopoly_action = eparams['monopoly_actions'][dict_key]
    #colab_action = eparams['colab_actions'][dict_key]
    #nash_profit = profit(nash_action, A0, MU, firm0, firm1, nA)
    #monopoly_profit = profit(monopoly_action, A0, MU, firm0, firm1, nA)
    #colab_profit = profit(colab_action, A0, MU, firm0, firm1, nA)
    
    # Initiate new env and amTFT agent
    s_next = env.reset(firm0, firm1, eparams)
    done = False
    total_pg0 = []
    total_pg1 = []
    # For sequentiality, I need to initiate action1
    action1 = agent.act(s_next[obs_firm1], 1, device = device.type)
    action0 = agent.act(s_next[obs_firm0], 1, device = device.type)
    for t in range(1, FRAMES):
        if frame_idx%1000 == 0:            
            # Save average reward
            #mean_pg = np.mean(agent.total_pg)
            mean_pg0 = np.mean(total_pg0)
            mean_pg1 = np.mean(total_pg1)
            writer.add_scalar("Agent0_avg_profit", mean_pg0, ep_idx)
            writer.add_scalar("Agent1_avg_profit", mean_pg1, ep_idx)
            #agent.total_pg = []
            total_pg0 = []
            total_pg1 = []
            ep_idx += 1
            
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        s = s_next
        
        # Sequentiality of action choices
        if frame_idx % 2 == 0:
            # Update only agent0's action
            action0 = agent.act(s[obs_firm0], epsilon, device = device.type)
        else:
            # Update only agent1's action
            action1 = agent.act(s[obs_firm1], epsilon, device = device.type) 
            
        s_next, reward_n, done, _ = env.step(action0, action1)
        exp = Experience(s[obs_firm0], action0, reward_n[0], done, s_next[obs_firm0])
        exp1 = Experience(s[obs_firm1], action1, reward_n[1], done, s_next[obs_firm1])
        agent.exp_buffer.append(exp)
        agent.exp_buffer.append(exp1)
        
        if reward_n is not None:
          # TODO: add for both firms
            reward = reward_n[0]
            pg = reward
            #pg = profit_gain(reward, nash_profit, colab_profit)[0] # important to index here
            agent.total_pg.append(pg)
            total_pg0.append(reward_n[0])
            total_pg1.append(reward_n[1])
    
        if len(agent.exp_buffer) < REPLAY_START_SIZE: 
            continue
    
        if frame_idx % SYNC_TARGET_FRAMES == 0: # Update target network
            agent.tgt_net.load_state_dict(agent.net.state_dict())
            
        batch = agent.exp_buffer.sample(BATCH_SIZE)
        agent.optimizer.zero_grad()
        loss_t = calc_loss(batch, agent.net, agent.tgt_net, device = device)
        loss_t.backward()
        # Gradient clipping
        for param in agent.net.parameters():
            param.grad.clamp_(-1, 1)
        agent.optimizer.step()
        writer.add_scalar("loss", loss_t, frame_idx)
        
        if frame_idx % 100_000 == 0:
          print(frame_idx)
    
        if frame_idx % (500_000-1) == 0:
            print(frame_idx)
            torch.save({
                'agent_state_dict': agent.net.state_dict(),
                'optimizer_dict': agent.optimizer.state_dict(),
                'epsilon': epsilon,
                'frame_idx': frame_idx,
                'env_state': s_next 
                },  str(frame_idx) + "_" + RANDOMNESS + PATH)
    writer.close()
    return(agent)