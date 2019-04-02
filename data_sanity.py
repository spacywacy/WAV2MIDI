import matplotlib.pyplot as plt


def get_note(row, i):
	midi_vec = row[-88:]
	print('\nline({})'.format(i))
	for i in range(len(midi_vec)):
		if float(midi_vec[i]) == 1:
			print('Note:', i+1)





def plot_freq(row):
	row_upper = 300
	freq_offset = 0
	freq_vec = row[:row_upper]
	xs = list(range(freq_offset, freq_offset+len(freq_vec)))
	plt.plot(xs, freq_vec)
	#plt.xscale('log')
	plt.show()








if __name__ == '__main__':
	with open('data/dataset_train.csv', 'r') as f:
		i = 0
		n_lines = 50

		for line in f:
			if i>=n_lines:
				break

			i+=1

			#print(line)
			data_row = line[:-1].split(',')
			print('row length:', len(data_row))
			get_note(data_row, i)
			plot_freq(data_row)
			#if i==760:
				#print(data_row, '\n')
				#print(len(data_row))
			



