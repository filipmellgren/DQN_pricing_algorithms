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
!git commit -m "moved two classes"
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
# Think that the code will run. Test this upon next session with small number of iterations.
import argparse
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
from agent import Agent1 as Agent
from experience_buffer import ExperienceBuffer
from config import PARAMS
from config import calc_loss
#from config import ENV
from cont_bertrand import ContBertrand
ENV = ContBertrand()

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
net0 = Net(4,2).to(device) # might need layers to depend on sizes of env: env.observation_space.shape, env.action_space.n).to(device)
print(net0)
net1 = Net(4,2).to(device)
tgt_net0 = Net(4,2).to(device) # Prediction target. TODO: do I maybe want just one target to speed up training? Might work because of symmetry and that there might be an objective functon
tgt_net1 = Net(4,2).to(device)
criterion = nn.MSELoss()
optimizer0 = optim.Adam(net0.parameters(), lr=LEARNING_RATE)
optimizer1 = optim.Adam(net1.parameters(), lr=LEARNING_RATE)
buffer0 = ExperienceBuffer(REPLAY_SIZE) # buffer from which to sample data
buffer1 = ExperienceBuffer(REPLAY_SIZE) # buffer from which to sample data


# Not sure about this:
# =============================================================================
# if args.resume:
#   model.load_state_dict(torch.load('model.pth'))
#   optimizer.load_state_dict(torch.load('optimiser.pth'))
# 
# =============================================================================
# Reinforcement learning environment
env = ENV
agent0 = Agent(env, buffer0) 
agent1 = Agent(env, buffer1)

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
#state = env.reset()
for t in range(1, 1_000):
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME) # TODO: according to book. Looks wrong
    # TODO: have two independent epsilons
    action0 = agent0.act(net0, epsilon) 
    action1 = agent1.act(net1, epsilon)
    state_n, reward_n, done, _ = env.step(action0, action1)
    
    if reward_n is not None: # TODO: now I'm doing this for just one agent
        reward = reward_n[0]
        total_rewards.append(reward)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame  = frame_idx
        ts = time.time()
        mean_reward = np.mean(total_rewards[-100:])
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net0.state_dict(),  "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
        if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break
    if len(buffer0) < REPLAY_START_SIZE and len(buffer1) < REPLAY_START_SIZE: # The two should have the same length
        continue
    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net0.load_state_dict(net0.state_dict())
        tgt_net1.load_state_dict(net1.state_dict())


    batch0 = buffer0.sample(BATCH_SIZE) # TODO: same or separate buffers? Probably two separate buffers
    optimizer0.zero_grad()
    loss_t0  = calc_loss(batch0, net0, tgt_net0, device = device)
    loss_t0.backward()
    optimizer0.step()
    
    batch1 = buffer1.sample(BATCH_SIZE)
    optimizer1.zero_grad()
    loss_t1  = calc_loss(batch1, net1, tgt_net1, device = device)
    loss_t1.backward()
    optimizer1.step()
    
writer.close()


