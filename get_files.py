from bs4 import BeautifulSoup as bs
import requests
import re
import os


def make_soup(url):
	res = requests.post(url)
	url_obj = res.content.decode('utf-8')
	soup = bs(url_obj, 'lxml')
	return soup

def download(url, fname):
	res = requests.get(url)
	with open(fname, 'wb') as f:
		f.write(res.content)
	print(url)





if __name__ == '__main__':

	if not os.path.isdir(os.path.abspath('audio')):
		os.mkdir('audio')
	if not os.path.isdir(os.path.abspath('midi')):
		os.mkdir('midi')

	url_ = 'http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html'
	base_url = 'http://resources.mpi-inf.mpg.de/SMD/'
	soup_ = make_soup(url_)
	db_table = soup_.find_all(id='DatabaseTable')[0]
	
	for link in db_table.find_all('a'):
		res_url = base_url + link['href']
		fname = res_url.split('/')[-1]

		if link.text == 'mp3':
			fname = 'audio/' + fname
			download(res_url, fname)

		elif link.text == 'mid':
			fname = 'midi/' + fname
			download(res_url, fname)


























