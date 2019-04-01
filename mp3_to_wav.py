from pydub import AudioSegment
import os



def convert_wav():
	in_dir = 'audio'
	out_dir = 'audio_wav'
	if not os.path.isdir(os.path.abspath(out_dir)):
		os.mkdir(out_dir)

	for fname in os.listdir(in_dir):
		in_f = os.path.join(in_dir, fname)
		mp3_f = AudioSegment.from_mp3(in_f)

		out_f = fname.split('.')[0] + '.wav'
		out_f = os.path.join(out_dir, out_f)
		mp3_f.export(out_f, format='wav')
		print(out_f)





if __name__ == '__main__':
	convert_wav()



