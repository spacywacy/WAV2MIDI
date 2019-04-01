import os
from scipy.io import wavfile
import numpy as np
import mido



class wav_processor():

	def __init__(self):
		#parameters
		self.interval = 0.2 #cut interval, in seconds
		self.write_to_file = True
		self.dataset = []
		self.n_rows = 0
		self.n_files = 0
		self.fourier_cutoffs = (25, 4400)
		self.time = 0.0

		#io
		self.wav_dir = 'audio_wav'
		self.midi_dir = 'midi'
		self.out_fname = 'dataset.csv'
		self.out_dir = 'data'
		if not os.path.exists(self.out_dir):
			os.makedirs(self.out_dir)
		self.out_full_dir = os.path.join(self.out_dir, self.out_fname)
		self.out_f = open(self.out_full_dir, 'w')

	#loop thru wav files in a dir
	def run(self):
		wav_dir_list = os.listdir(self.wav_dir)
		midi_dir_list = os.listdir(self.midi_dir)
		wav_dir_list.sort()
		midi_dir_list.sort()

		for wav_fname, midi_fname in zip(wav_dir_list, midi_dir_list):
			wav_full_dir = os.path.join(self.wav_dir, wav_fname)
			midi_full_dir = os.path.join(self.midi_dir, midi_fname)
			if self.check_same_dir(wav_full_dir, midi_full_dir):
				self.cut_wav(wav_full_dir, midi_full_dir)
			else:
				print('wav & midi files do not match')

			self.n_files += 1

		self.out_f.close()
		print('Total files:', self.n_files)
		print('Total rows:', self.n_rows)

	#cut a wav file into pieces
	def cut_wav(self, wav_dir, midi_dir):
		print('Processing({}): {}'.format(self.n_files, wav_dir))
		sample_rate, data = wavfile.read(wav_dir)
		data = np.array([x[0] for x in data])
		interval_len = int(sample_rate * self.interval)
		n_cuts = int(len(data)/interval_len)
		pos = 0
		midi_ = midi_wrapper(midi_dir, self.interval)

		for i, label in zip(range(n_cuts), midi_.gen_label()):
			cut = data[pos:pos+interval_len]
			pos += interval_len
			cut_fft = self.fourier_trans(cut)
			data_row = np.append(cut_fft, label)

			if self.write_to_file:
				line = ','.join([str(x) for x in list(data_row)]) + '\n'
				self.out_f.write(line)
			else:
				self.dataset.append(list(data_row))

			self.n_rows += 1


	def fourier_trans(self, wave_array):
		fft_array = abs(np.fft.fft(wave_array))
		#print('data length:',len(fft_array[self.fourier_cutoffs[0]: self.fourier_cutoffs[1]]))
		return fft_array[self.fourier_cutoffs[0]: self.fourier_cutoffs[1]]

	def check_same_dir(self, wav_dir, midi_dir):
		w_ = wav_dir.split('.')[0].split('/')[1]
		m_ = midi_dir.split('.')[0].split('/')[1]
		same = w_ == m_
		#print(w_, m_)
		#print('same:', same)
		return same


class midi_wrapper():

	def __init__(self, midi_fname, interval):
		self.mid = mido.MidiFile(midi_fname)
		self.current_nodes = []
		self.vec = [0 for x in range(88)]
		self.track = []
		self.n_msg = 0
		self.time = 0.0
		self.interval = interval
		self.get_start_time()
		self.loop_msg()

	def add_note(self, note):
		self.vec[note-21] = 1

	def rm_note(self, note):
		self.vec[note-21] = 0

	def get_start_time(self):
		time = 0.0
		for msg in self.mid:
			time += msg.time
			if hasattr(msg, 'note') and msg.type == 'note_on':
				self.start_time = time
				break


	def loop_msg(self):
		for msg in self.mid:
			#if self.n_msg > 50:
				#break

			self.time += msg.time
			if hasattr(msg, 'note'):
				if msg.type == 'note_on':
					self.add_note(msg.note)
				elif msg.type == 'note_off':
					self.rm_note(msg.note)

				line = (self.time, self.vec.copy())
				self.track.append(line)

			self.n_msg += 1

	def track_lookup(self, t):
		for i in range(len(self.track)):
			if t < self.track[i][0]:
				return self.track[i-1][1]

	def gen_label(self):
		n_labels = int(self.time/self.interval) + 1
		t = self.start_time
		for i in range(n_labels):
			label_vec = self.track_lookup(t)
			t += self.interval

			yield label_vec





if __name__ == '__main__':
	wav_processor().run()






























