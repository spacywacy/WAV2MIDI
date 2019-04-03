import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import random


class FC_NET(nn.Module):

	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(4375,10000)
		self.fc2 = nn.Linear(10000,2000)
		self.fc3 = nn.Linear(2000, 88)
		self.name = 'simply_fc'

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return x

class CONV_NET_1D_0(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv1d(1, 3, 100)
		self.conv2 = nn.Conv1d(3, 6, 10)
		self.fc1 = nn.Linear(246, 128)
		self.fc2 = nn.Linear(128, 88)
		self.name = 'conv_net_1d_0'

	def forward(self, x):
		x = F.max_pool1d(F.relu(self.conv1(x)), 10)
		x = F.max_pool1d(F.relu(self.conv2(x)), 10)
		x = x.view(-1, self.num_flat_feature(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x

	def num_flat_feature(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features



if __name__ == '__main__':
	net = FC_NET()
	p = net.parameters()
	for item in p:
		print(item.shape)




























