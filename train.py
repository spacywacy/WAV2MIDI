import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import random
import matplotlib.pyplot as plt
import nets
from test import Test
import pickle
from time import time


class Train():

	def __init__(self):
		#io
		self.data_dir = 'data/dataset_train.csv'
		self.model_bin = 'model_bin'
		if not os.path.exists(self.model_bin):
			os.makedirs(self.model_bin)

		#net & loss
		#self.net = nets.FC_NET()
		self.net = nets.CONV_NET_1D_0()
		self.criterion = nn.MSELoss()
		self.learning_rate = 0.000000001
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9)
		#self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.learning_rate)
		#self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
		self.channel_dim = True

		#training specifics
		self.batch_size = 1024 #size of a single batch
		self.n_batch = 60 #number of batches to load from dataset
		self.n_epoch = 8 #number of epochs
		
		#utilities
		self.label_len = 88 #length of the label vector
		self.cost_history = [] #losses over batches
		self.cost_plot_offset = 30 #ignore cost first n number of batch during plotting
		self.n_rows_used = 0 #number of rows from the training set
		
		#validation
		self.if_validate = True
		self.if_train_error = False
		self.vali_score = np.inf
		self.vali_data_dir = 'data/dataset_validation.csv'


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

		return i_row

	def single_epoch(self, i_epoch=0):
		i_batch = 0
		for batch in self.load_batch():
			if i_batch >= self.n_batch:
				break

			Xs = torch.tensor([x[:-self.label_len] for x in batch], dtype=torch.float)
			y = torch.tensor([x[-self.label_len:] for x in batch], dtype=torch.float)

			if self.channel_dim:
				Xs = Xs.view(Xs.shape[0], 1, Xs.shape[1])
			
			self.optimizer.zero_grad()
			output = self.net(Xs)
			loss = self.criterion(output, y)
			self.cost_history.append(float(loss))
			print('epoch: {}, batch: {}, loss: {}'.format(str(i_epoch), str(i_batch), str(float(loss.data))))
			loss.backward()
			self.optimizer.step()
			i_batch += 1

	def train_loop(self):
		#training
		for i_epoch in range(self.n_epoch):
			n_rows = self.single_epoch(i_epoch)
			if i_epoch == 0:
				self.n_rows_used = n_rows
		print('Done training')

		#validation
		if self.if_validate:
			vali_wrapper = Test(self.vali_data_dir,
								self.net,
								loss=self.criterion)
			self.vali_score = float(vali_wrapper.run_test())
		print('Done validating')

		#closing
		self.close_train()
		print('Dumped files')

	def plot_cost(self, plt_dir=None):
		xs = list(range(self.cost_plot_offset, len(self.cost_history)))
		plt.plot(xs, self.cost_history[self.cost_plot_offset:])
		plt.xlabel('Iterations')
		plt.ylabel('Cost')
		if plt_dir:
			plt.savefig(plt_dir)
		else:
			plt.show()

	def pickle_dump(self, fname, obj_):
		with open(fname, 'wb') as f:
			pickle.dump(obj_, f)

	def pickle_load(self, fname):
		with open(fname, 'rb') as f:
			obj = pickle.load(f)
		return obj

	def write_model_desc(self, fname):
		with open(fname, 'w') as f:
			f.write('Net name: {}\n'.format(self.net.name))
			f.write('Loss: {}\n'.format(str(self.criterion)))
			f.write('Learning rate: {}\n'.format(self.learning_rate))
			f.write('Batch size: {}\n'.format(self.batch_size))
			f.write('Total batches: {}\n'.format(self.n_batch))
			f.write('Total epochs: {}\n'.format(self.n_epoch))
			f.write('Validation error: {}\n'.format(self.vali_score))
			f.write('Graph offset: {}\n'.format(self.cost_plot_offset))
			f.write('Total rows of data used: {}\n'.format(self.n_rows_used))

	def close_train(self):
		model_name = '{}_{}'.format(self.net.name, str(int(time())))
		model_pickle_dir = os.path.join(self.model_bin, model_name + '.pickle')
		model_desc_dir = os.path.join(self.model_bin, model_name + '.txt')
		model_graph_dir = os.path.join(self.model_bin, model_name + '.png')
		self.pickle_dump(model_pickle_dir, self.net)
		self.plot_cost(plt_dir=model_graph_dir)
		self.write_model_desc(model_desc_dir)



if __name__ == '__main__':
	Train().train_loop()
























