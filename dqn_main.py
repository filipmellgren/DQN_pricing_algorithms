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
!git add dqn_main.py
!git add dqn_model.py
!git add agent.py
!git add experience_buffer.py
!git commit -m "This version runs w.o. problems. From here, I can tidy it up and ensure everything is 100% correct"
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
# TODO: one algorithm thrives, the other sucks. Why is this?
# I know that adding many actions don't increase the time by much. But for now, it results in wrong numbers. 
# Wacky transformation of price might explain why I get weird results

import torch
from torch import nn, optim
import numpy as np
from tensorboardX import SummaryWriter
from dqn_model import Net
import time
from agent import Agent1 as Agent
from experience_buffer import ExperienceBuffer
from config import PARAMS
from config import calc_loss
from cont_bertrand import ContBertrand
from config import args #args includes a few arguments used throughout. TODO: should all arguments be passed via this guy?
from config import ECON_PARAMS
from config import avg_profit_gain
ENV = ContBertrand()
import collections

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

GAMMA = PARAMS[1]
BATCH_SIZE = int(PARAMS[1])
REPLAY_SIZE = int(PARAMS[2])
REPLAY_START_SIZE = PARAMS[3]
LEARNING_RATE = PARAMS[4]
SYNC_TARGET_FRAMES = PARAMS[5]
EPSILON_DECAY_LAST_FRAME = PARAMS[6]
EPSILON_START =  PARAMS[7]
EPSILON_FINAL = PARAMS[8]
MEAN_REWARD_BOUND = PARAMS[9]
epsilon= EPSILON_START
nA = PARAMS[10].astype(int)
dO = PARAMS[11].astype(int)

NASH_PROFIT = ECON_PARAMS[6]
NASH_PROFIT = NASH_PROFIT[0] # 0.11278
MONOPOLY_PROFIT = ECON_PARAMS[7] # 0.11576

# PyTorch setup
use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu') # CPU when GPU is not available <-> device agnostic
print(device)

torch.manual_seed(args.seed)
if use_cuda:
  torch.cuda.manual_seed(args.seed)

# Neural network model:
net0 = Net(2,nA).to(device) # might need layers to depend on sizes of env: env.observation_space.shape, env.action_space.n).to(device)
print(net0)
net1 = Net(2,nA).to(device)
tgt_net0 = Net(2,nA).to(device) # Prediction target. TODO: do I maybe want just one target to speed up training? Might work because of symmetry and that there might be an objective functon
tgt_net1 = Net(2,nA).to(device)
criterion = nn.MSELoss()
optimizer0 = optim.Adam(net0.parameters(), lr=LEARNING_RATE)
optimizer1 = optim.Adam(net1.parameters(), lr=LEARNING_RATE)
buffer0 = ExperienceBuffer(REPLAY_SIZE) # buffer from which to sample data
buffer1 = ExperienceBuffer(REPLAY_SIZE) # buffer from which to sample data

# Reinforcement learning environment
env = ENV
agent0 = Agent(env, buffer0, net0, tgt_net0, optimizer0)
agent1 = Agent(env, buffer1, net1, tgt_net1, optimizer1)

# Write output statistics to tensorboard
writer = SummaryWriter(comment = "-")

# Initialize variables
env.seed(args.seed)
torch.manual_seed(args.seed)

frame_idx = 0
ts_frame = 0
ts = time.time()
# Training – Main loop
# TODO: repeat training loop multiple times for main results
#state = env.reset()
BETA = 1.7*10**-5
s_next = np.array([0,0,0,0]) # TODO: this is a bit too hacky. reset env
for t in range(1, 250_000):
    frame_idx += 1
   # epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME) # TODO: according to book. Looks wrong
    epsilon = np.exp(-BETA*frame_idx)
    
    s = s_next
    # TODO: have two independent epsilons
    action0 = agent0.act(net0, s[0:2], epsilon)
    action1 = agent1.act(net1, s[2:5], epsilon)
    s_next, reward_n, done, _ = env.step(action0, action1) # TODO somehow new state is 4 dim
        
    exp0 = Experience(s_next[0:2], action0, reward_n[0], done, s[0:2])
    agent0.exp_buffer.append(exp0)
    exp1 = Experience(s_next[2:5], action1, reward_n[1], done, s[2:5])
    agent1.exp_buffer.append(exp1)
    
    
    if reward_n is not None: # TODO: now I'm doing this for just one agent
        a = 0
        for agent in [agent0, agent1]:
            reward = reward_n[a]
            agent.total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame  = frame_idx
            ts = time.time()
            mean_reward = np.mean(agent.total_rewards[-100:])
            apg = avg_profit_gain(reward)
            #writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar(str(a), apg, frame_idx)
            if agent.best_mean_reward is None or agent.best_mean_reward < mean_reward:
                torch.save(agent.net.state_dict(),  "-best.dat")
                if agent.best_mean_reward is not None:
                    print("Best mean reward updated, %.3f: %.3f -> %.3f, model saved" % (a, agent.best_mean_reward, mean_reward))
                agent.best_mean_reward = mean_reward
            if mean_reward > args.reward: # TODO: this is where I check for the convergence criterion
                print("Solved in %d frames!" % frame_idx)
                break # TODO: does it break both loops or just one?
            a += 1
            
    if len(agent0.exp_buffer) < REPLAY_START_SIZE: # TODO: this guy never gets appended -.-
        continue

    for agent in [agent0, agent1]:
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            agent.tgt_net.load_state_dict(agent.net.state_dict())
        batch = agent.exp_buffer.sample(BATCH_SIZE)
        agent.optimizer.zero_grad()
        loss_t = calc_loss(batch, agent.net, agent.tgt_net, device = device) # does the agent ever value update? Where?
        loss_t.backward()
        agent.optimizer.step()
    
writer.close()
# TODO: reset agents
# TODO: change mean_reward to avg_profit_gain AND duration of same optimal action
# 20:25, started a 500k iteration step long journey
# Had ended 22:23
# Lead to non convergence. Increase learnign rate


# 08:56, learning rate 10 times as large -> 09:39
# changed # actions to 50 09:40 -> 10:22
