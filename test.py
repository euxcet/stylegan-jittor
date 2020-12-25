'''
import torch
from torch import nn

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        self.weight = nn.Parameter(torch.zeros((1, 10, 1, 1)))

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.predict(x)
        return x

net = Net(2, 5, 3)
for num, para in enumerate(list(net.parameters())):
    print(num, para)


'''

import jittor as jt
import numpy as np
from jittor import nn, Module, init

class Net(jt.nn.Module):
    def __init__(self):
        self.weight = jt.float32([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        self.weight.requires_grad = False
        self.layer1 = nn.Linear(1, 10)

    def execute(self, x):
        x = self.layer1(x)
        return x

net = Net()
for num, para in enumerate(list(net.parameters())):
    print(num, para)