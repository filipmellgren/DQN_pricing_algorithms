#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:03:48 2019

DQN Model
p 137 DRL hands on

This file defines a neural network

"""

import torch.nn.functional as F
import torch.nn as nn



# =============================================================================
# class DQN(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(DQN, self).__init__()
# 
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU()
#         )
#         
#         conv_out_size = self._get_conv_out(input_shape)
#         self.fc = nn.Sequential(
#             nn.Linear(conv_out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, n_actions)
#         )
# 
#     def _get_conv_out(self, shape):
#         print(torch.zeros(1, *shape))
#         o = self.conv(torch.zeros(1, *shape))
# 
#         return int(np.prod(o.size()))
# 
#     def forward(self, x):
#         conv_out = self.conv(x).view(x.size()[0], -1)
#         return self.fc(conv_out)
# =============================================================================
# TODO: simplify the net
# TOOD: understand it
# feedforward fullly connected with ReLus according to Goodfellow (good thing it already had reLus)

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_dim, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, output_dim)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)




