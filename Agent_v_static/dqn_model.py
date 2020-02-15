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

# feedforward fullly connected with ReLus according to Goodfellow
# Net similar to apple gatherer/wolfpack paper: https://arxiv.org/pdf/1702.03037.pdf?utm_source=datafloq&utm_medium=ref&utm_campaign=datafloq

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, nodes):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_dim, nodes) # Hidden layer
        self.hidden2 = nn.Linear(nodes, nodes) # Hidden layer
        self.output = nn.Linear(nodes, output_dim) # Output layer

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        return x

# =============================================================================
# class Net(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Net, self).__init__()
#         self.hidden = nn.Linear(input_dim, 32) # Hidden layer
#         self.output = nn.Linear(32, output_dim) # Output layer
# 
#     def forward(self, x):
#         x = self.hidden(x)
#         x = F.relu(x)
#         x = self.output(x)
#         return x
#     # Think it was 32 when first worked. Testing with 16.
# =============================================================================
