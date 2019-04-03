import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import random
import matplotlib.pyplot as plt
import nets
import pickle
from time import time


class Test():

	def __init__(self, data_dir, model, loss=None, n_batch=None):
		#io
		self.data_dir = data_dir
		self.model_bin = 'model_bin'

		#net
		if type(model) == str:
			model_full_dir = os.path.join(self.model_bin, model)
			self.net = self.load_model(model_full_dir)
		else:
			self.net = model

		#loss
		if loss:
			self.criterion = loss
		else:
			self.criterion = nn.MSELoss()

		#utilities
		self.label_len = 88
		self.n_batch = n_batch
		self.channel_dim = True

		#results
		self.batch_losses = []
		self.loss = 0.0


	def load_model(self, fname):
		with open(fname, 'rb') as f:
			obj = pickle.load(f)
		return obj

	def run_test(self):
		if self.n_batch:
			self.test_w_batch()
		else:
			self.test_all_data()

		return self.loss

	def test_all_data(self):
		with open(self.data_dir, 'r') as f:
			all_data = []
			for line in f:
				data_row = [float(x) for x in line[:-1].split(',')]
				all_data.append(data_row)

			Xs = torch.tensor([x[:-self.label_len] for x in all_data], dtype=torch.float)
			y = torch.tensor([x[-self.label_len:] for x in all_data], dtype=torch.float)
			if self.channel_dim:
				Xs = Xs.view(Xs.shape[0], 1, Xs.shape[1])

			output = self.net(Xs)
			self.loss = self.criterion(output, y)
			print('{} on all data: {}'.format(str(self.criterion), str(self.loss)))

	def test_w_batch(self):
		i_batch = 0
		for batch in self.load_batch():
			if i_batch >= self.n_batch:
				break

			Xs = torch.tensor([x[:-self.label_len] for x in batch], dtype=torch.float)
			y = torch.tensor([x[-self.label_len:] for x in batch], dtype=torch.float)
			if self.channel_dim:
				Xs = Xs.view(Xs.shape[0], 1, Xs.shape[1])
			
			output = self.net(Xs)
			self.batch_losses.append(self.criterion(output, y))
			i_batch += 1

		self.loss = sum(self.batch_losses)/len(self.batch_losses)
		print('Average {} over {} batches: {}'.format(str(self.criterion), str(self.n_batch), str(self.loss)))

	def load_batch(self):
		i_row = 1
		batch = []

		with open(self.data_dir, 'r') as f:
			for line in f:
				if i_row % self.batch_size == 0 and i_row!=1:
					yield batch
					batch = []

				try:
					data_row = [float(x) for x in line[:-1].split(',')]
				except Exception as e:
					print('Error when appending row({}): {}'.format(str(i_row), str(e)))

				batch.append(data_row)
				i_row += 1



def test_Test():
	data_dir = 'data/dataset_validation.csv'
	model = 'simply_fc_1554228850.pickle'
	test_obj = Test(data_dir, model)
	test_obj.run_test()


if __name__ == '__main__':
	test_Test()































