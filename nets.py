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

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return x































