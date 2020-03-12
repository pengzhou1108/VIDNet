import cv2
import numpy as np
import os
import glob
from os.path import isfile, join
import pdb
import argparse
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', metavar='INPUT', 
	                    help='filenames of input images', required=True)
	return parser.parse_args()
if __name__ == "__main__":
	args = get_args()
	output_folder = args.input
	folders = glob.glob(output_folder+'/*')
	#pdb.set_trace()
	fps = 20
	frame_array = []
	for pathIn in folders:
		files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
		#for sorting the file names properly
		if 'results' in pathIn:
			files.sort(key = lambda x: x[-6:-4])
		else:
			files.sort(key = lambda x: x[0:5])
		files.sort()
		frame_array = []
		for i in range(len(files)):
			filename=os.path.join(pathIn, files[i])
			#reading each files
			img = cv2.imread(filename)
			#height, width, layers = img.shape
			size = (427,240)
			img = cv2.resize(img, size)
			#inserting the frames into an image array
			frame_array.append(img)
		pathOut = pathIn.split('/')[-1] + '.mp4'
		out = cv2.VideoWriter(os.path.join(output_folder,pathOut),cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
		for i in range(len(frame_array)):
			# writing to a image array
			out.write(frame_array[i])
		out.release()