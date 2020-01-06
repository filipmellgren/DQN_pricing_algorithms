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
!git commit -m "commit"
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
import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import gym
import numpy as np
# import experience buffer
from tensorboardX import SummaryWriter
from dqn_model import Net
import time
from agent import Agent
from experience_buffer import ExperienceBuffer
from config import PARAMS
from config import calc_loss
from config import ENV


GAMMA = PARAMS[0]
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


# PyTorch setup
parser = argparse.ArgumentParser(description='PyTorch CartPole Example')

parser.add_argument('--lr', type=float, default=LEARNING_RATE, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
args = parser.parse_args() # TODO: load just this guy for minimalism

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu') # CPU when GPU is not available <-> device agnostic
print(device)

torch.manual_seed(args.seed)
if use_cuda:
  torch.cuda.manual_seed(args.seed)


# Neural network model:
net = Net(4,2).to(device) # might need layers to depend on sizes of env: env.observation_space.shape, env.action_space.n).to(device)
print(net)
tgt_net = Net(4,2).to(device) # Prediction target
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
buffer = ExperienceBuffer(REPLAY_SIZE) # buffer from which to sample data



# Not sure about this:
# =============================================================================
# if args.resume:
#   model.load_state_dict(torch.load('model.pth'))
#   optimizer.load_state_dict(torch.load('optimiser.pth'))
# 
# =============================================================================
# Reinforcement learning environment
env = ENV
agent = Agent(env, buffer) # TODO: make similar to Agent environment I had before

# Write output statistics to tensorboard
writer = SummaryWriter(comment = "-")

# Initialize variables
env.seed(args.seed)
torch.manual_seed(args.seed)

total_rewards = []
frame_idx = 0
ts_frame = 0
ts = time.time()
best_mean_reward = None

# Training – Main loop
# =============================================================================
# Have to think about whether I want to reset it 1000 times as Calvano does. Might become overtly massive.
#     state = env.reset()
#     ep_reward = 0
# =============================================================================
#for episode in range(1): # Or is it epochs? 
state = env.reset()
for t in range(1, 100_000):
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME) # TODO: according to book. Looks wrong
# =============================================================================
#         # select action from policy
#         action = select_action(state)
#         # take the action
#         state, reward, done, _ = env.step(action)
# =============================================================================
    reward = agent.play_step(net, epsilon, device=device)
    state = np.array([agent.state], copy=False)
    if reward is not None:
        total_rewards.append(reward)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame  = frame_idx
        ts = time.time()
        mean_reward = np.mean(total_rewards[-100:])
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        # TODO: ADD SOMETHING ABOUT NET p 147 Or what is just print something?
        if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(),  "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
        if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break
    if len(buffer) < REPLAY_START_SIZE:
        continue
    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())


    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t  = calc_loss(batch, net, tgt_net, device = device)
    loss_t.backward()
    optimizer.step()
writer.close()

# LR *2: 141.83
# LR *10: 155.1
# LR *100: Quit, no progress shown
# LR *5: (0.001*5): 167.4


