import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import random
import matplotlib.pyplot as plt
import nets


def read_batch(data_dir, batch_size):
	i_row = 1
	batch = []

	with open(data_dir, 'r') as f:
		for line in f:
			if i_row%batch_size == 0 and i_row!=1:
				clean_batch = True
				yield batch
				batch = []

			try:
				data_row = [float(x) for x in line[:-1].split(',')]
			except Exception as e:
				print('Error when appending row({}): {}'.format(str(i_row), str(e)))

			batch.append(data_row)
			i_row+=1

def single_epoch(i_epoch=0):
	i_batch = 0
	for batch in read_batch(data_dir, batch_size):
		if i_batch >= n_batch:
			break

		Xs = torch.tensor([x[:-label_len] for x in batch], dtype=torch.float)
		y = torch.tensor([x[-label_len:] for x in batch], dtype=torch.float)
		
		optimizer.zero_grad()
		output = net(Xs)
		loss = criterion(output, y)
		cost_history.append(loss)
		print('epoch: {}, batch: {}, loss: {}'.format(str(i_epoch), str(i_batch), str(float(loss.data))))
		loss.backward()
		optimizer.step()
		i_batch+=1

def train_loop():
	for i_epoch in range(n_epoch):
		single_epoch(i_epoch)

def plot_cost(cost_history):
	xs = list(range(5, len(cost_history)))
	plt.plot(xs, cost_history[5:])
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.show()


if __name__ == '__main__':

	data_dir = 'data/dataset_train.csv'
	batch_size = 1000
	n_batch = 50
	n_epoch = 10
	label_len = 88
	net = nets.FC_NET()
	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr=0.000000001)
	cost_history = []
	train_loop()
	plot_cost(cost_history)
























