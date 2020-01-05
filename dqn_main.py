#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:47:03 2019

@author: filip

!echo "# DQN_pricing_algorithms" >> README.md
!git init
!git add README.md
!git add dqn_main.py
!git commit -m "first commit"
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
import collections
# import experience buffer
from tensorboardX import SummaryWriter
from dqn_model import Net
import time

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10_000
REPLAY_START_SIZE = 10_000
LEARNING_RATE = 0.001
SYNC_TARGET_FRAMES = 1000
EPSILON_DECAY_LAST_FRAME = 100_000
EPSILON_START =  1.0
EPSILON_FINAL = 0.02
epsilon = EPSILON_START
MEAN_REWARD_BOUND = 195 # TODO: adapt!
#?
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.bool), np.array(next_states) #Change to bool from uint8

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0
    
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        
        if  np.random.random() < epsilon: 
            action = env.action_space.sample()
        else:            
            state_v = torch.Tensor(self.state)
            q_vals_v = net(state_v)
            #_, act_v = torch.max(q_vals_v, dim=1) I changed dim to 0, correct? #TODO
            _, act_v = torch.max(q_vals_v, dim=0) 
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
    
def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device).float()
    next_states_v = torch.tensor(next_states).to(device).float()
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.from_numpy(dones).to(device) #torch.from_numpy() ByteTensor

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

# PyTorch setup
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--batch-size', type=int, default=64, metavar='N', #?
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', #?
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=LEARNING_RATE, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', #?
                    help='SGD momentum (default: 0.5)')
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
env = gym.make("CartPole-v1")
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
for t in range(1, 250_000):
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
    # update cumulative reward. Where is this happening? Maybe in calc_loss
#   running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t  = calc_loss(batch, net, tgt_net, device = device)
    loss_t.backward()
    optimizer.step()
            # Does:
# =============================================================================
#                 # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels) # M.Lapan uses hiw own function here
#         loss.backward()
#         optimizer.step()
# =============================================================================
writer.close()




