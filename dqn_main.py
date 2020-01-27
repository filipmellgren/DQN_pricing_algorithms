#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:47:03 2019

@author: filip

!echo "# DQN_pricing_algorithms" >> README.md
!git init
!git add README.md
!git add cont_bertrand.py
!git add config.py
!git add calc_nash_monopoly.py
!git add dqn_main.py
!git add dqn_model.py
!git add agent.py
!git add experience_buffer.py
!git commit -m "updated to allow for use of a gpu"
!git remote add origin https://github.com/filipmellgren/DQN_pricing_algorithms.git
!git push -u origin master

Grokking pytorch: https://github.com/Kaixhin/grokking-pytorch/blob/master/README.md?fbclid=IwAR1h7QPpTi8-v7ij6fJvVLRm-weI5HzohiUtIG7CDB0jZGr0cE0lEcNApXE
Cartpole with 4d input on stackoverflow: https://stackoverflow.com/questions/56964657/cartpole-v0-loss-increasing-using-dqn

From the correct directory in the terminal write:
    !tensorboard --logdir runs --host localhost
Then go to:
    http://localhost:6006/
in the browser

"""
# Imports I might need:
#from torch.nn import functional as F
#from torch.utils.data import DataLoader
#import gym
# import experience buffer
#from config import ENV

# TODO: do I have to "turn off" the network somehow?
# I know that adding many actions don't increase the time by much. But for now, it results in wrong numbers. 
# Wacky transformation of price might explain why I get weird results

# TODO: another idea in foerster is to simply use a shorter, more recent experience replay buffers

# TODO: reset agents
    # Add to this the "warm" initialisation of net and optimizer

# TODO: Let synchronizatin be closer related to learnng rate
# TODO: how often to sync network?
# TODO: sequential actions


import torch
from torch import nn, optim
import numpy as np
from tensorboardX import SummaryWriter
from dqn_model import Net
import time
from agent import Agent1 as Agent
from experience_buffer import ExperienceBuffer
from config import HYPERPARAMS
from config import calc_loss
from cont_bertrand import ContBertrand
from config import avg_profit_gain
ENV = ContBertrand()
import collections
import os.path

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

params = HYPERPARAMS['full_obs_NB']
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
SEED = params['seed']
PATH = params['path']

# PyTorch setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu') # CPU when GPU is not available <-> device agnostic
print(device)

torch.manual_seed(SEED)
if use_cuda:
  torch.cuda.manual_seed(SEED)

# Neural network model:
net0 = Net(dO_a,nA).to(device)
net1 = Net(dO_a,nA).to(device)
tgt_net0 = Net(dO_a,nA).to(device) # Prediction target. TODO: do I maybe want just one target to speed up training? Might work because of symmetry and that there might be an objective functon
tgt_net1 = Net(dO_a,nA).to(device)
criterion = nn.MSELoss()
optimizer0 = optim.Adam(net0.parameters(), lr=LEARNING_RATE) 
optimizer1 = optim.Adam(net1.parameters(), lr=LEARNING_RATE)
buffer0 = ExperienceBuffer(REPLAY_SIZE)
buffer1 = ExperienceBuffer(REPLAY_SIZE)

# Reinforcement learning environment
env = ENV
agent0 = Agent(env, buffer0, net0, tgt_net0, optimizer0)
agent1 = Agent(env, buffer1, net1, tgt_net1, optimizer1)

# Write output statistics to tensorboard
writer = SummaryWriter(comment = "-")

# Initialize variables
env.seed(SEED) # TODO; is this used
torch.manual_seed(SEED)

frame_idx = 0
ts_frame = 0
ts = time.time()

s_next = env.reset()
epsilon = EPSILON_START

# Initialize nets and optimizers:
if os.path.exists(PATH):
    checkpoint = torch.load(PATH)
    agent0.net.load_state_dict(checkpoint['agent0_state_dict'])
    agent1.net.load_state_dict(checkpoint['agent1_state_dict'])
    agent0.optimizer.load_state_dict(checkpoint['optimizer0_dict'])
    agent1.optimizer.load_state_dict(checkpoint['optimizer1_dict'])
    frame_idx = checkpoint['frame_idx']
    epsilon = checkpoint['epsilon']
    s_next = checkpoint['env_state']


# Training – Main loop

FRAMES = 5_000 # Update manually here for experimentatino purposes
time_start = time.time()

for t in range(1, FRAMES):
    frame_idx += 1
    #epsilon = np.exp(-BETA*frame_idx) + 0.01
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    s = s_next
    action0 = agent0.act(net0, s[np.array([0,2,4,5])], epsilon, device = device.type)
    action1 = agent1.act(net1, s[np.array([0,2,4,5])], epsilon, device = device.type)
    s_next, reward_n, done, _ = env.step(action0, action1, epsilon, frame_idx)
        
    exp0 = Experience(s_next[np.array([0,2,4,5])], action0, reward_n[0], done, s[np.array([0,2,4,5])])
    agent0.exp_buffer.append(exp0)
    exp1 = Experience(s_next[np.array([0,2,4,5])], action1, reward_n[1], done, s[np.array([0,2,4,5])])
    agent1.exp_buffer.append(exp1)
    
    if reward_n is not None:
        a = 0
        for agent in [agent0, agent1]:
            reward = reward_n[a]
            pg = avg_profit_gain(reward)
            #agent.total_rewards.append(reward)
            agent.total_pg.append(pg)
            #speed = (frame_idx - ts_frame) / (time.time() - ts)
            #ts_frame  = frame_idx
            #ts = time.time()
            
            mean_pg = np.mean(agent.total_pg[-10000:])
        #writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar(str(a), mean_pg, frame_idx)
            if agent.best_mean_pg is None or agent.best_mean_pg < mean_pg or frame_idx % (SYNC_TARGET_FRAMES) == 0:
                torch.save(agent.net.state_dict(),  "-best.dat")  
                if agent.best_mean_pg is not None:
                    print("Best mean profit gain updated, %.1f: %.3f -> %.3f, model saved. Iteration: %.1f" % (a, agent.best_mean_pg, mean_pg, frame_idx))
                agent.best_mean_pg = mean_pg
            if agent.length_opt_act > 25000: # TODO: Don't hardcode
                print("Solved in %d frames!" % frame_idx)
                print(agent.length_opt_act)
                break
            a += 1
            
    if len(agent0.exp_buffer) < REPLAY_START_SIZE: 
        continue

    for agent in [agent0, agent1]:
        if frame_idx % SYNC_TARGET_FRAMES == 0: # Update target network
            agent.tgt_net.load_state_dict(agent.net.state_dict())
        batch = agent.exp_buffer.sample(BATCH_SIZE)
        agent.optimizer.zero_grad()
        loss_t = calc_loss(batch, agent.net, agent.tgt_net, device = device) # does the agent ever value update? Where?
        loss_t.backward()
        agent.optimizer.step()
    writer.add_scalar(str(a) + "loss", loss_t, frame_idx)
    
    if frame_idx % 1_000_000:
        torch.save({
            'agent0_state_dict': agent0.net.state_dict(),
            'agent1_state_dict': agent1.net.state_dict(),
            'optimizer0_dict': agent0.optimizer.state_dict(),
            'optimizer1_dict': agent1.optimizer.state_dict(),
            'epsilon': epsilon,
            'frame_idx': frame_idx,
            'env_state': s_next
            }, PATH)
    
writer.close()
time_stop = time.time()
print(time_start - time_stop) # 100 000 frames on CPU: 1817 seconds, on GPU in colab: 952
# 50 000 on CPU: 1566 (?) on GPU: 
