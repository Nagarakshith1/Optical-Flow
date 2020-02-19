from frameExtractor import frameExtractor
from makeMovie import makeMovie
from estimateAllTranslation import estimateAllTranslation
from getFeatures import getFeatures
from applyGeometricTransformation import applyGeometricTransformation
from utils import *
import bboxgen

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from datetime import datetime


def plot_trajectory(x2, y2, img2, img_name, folder_name):
	implot = plt.imshow(img2)
	plt.xticks([]), plt.yticks([])
	plt.plot(x2, y2, 'ro', markersize=1)
	plt.savefig(folder_name + '/frame_tracked_' + str(img_name) + '.png', bbox_inches='tight', pad_inches=0)

	plt.close()


def plot_box(xmin, ymin, xmax, ymax, img2, img_name, x2, y2, folder_name):
	fig, ax = plt.subplots(1)

	# Display the image
	ax.imshow(img2)
	plt.xticks([]), plt.yticks([])
	plt.plot(x2, y2, 'ro', markersize=3)
	for i in range(len(xmin)):
		if xmin[i] != -1:
			rect = patches.Rectangle((xmin[i], ymin[i]), xmax[i] - xmin[i], ymax[i] - ymin[i], linewidth=2,
									 edgecolor='b',
									 facecolor='none')
			ax.add_patch(rect)
	fig.savefig(folder_name + '/frame_tracked_' + str(img_name) + '.png', bbox_inches='tight', pad_inches=0)
	plt.close()


def objectTracking(rawVideo):
	folder_name = str(datetime.now()).replace('-', '_').replace(':', '_').replace('.', '_').replace(' ', '_')
	os.mkdir(folder_name + '_bounding_box')
	os.mkdir(folder_name + '_trajectory')
	n = 2  # Number of bounding boxes

	medium = 0
	nof = 5  # number of features

	if 'medium' in rawVideo:
		medium = 1
		nof = 10
		n = 1
	frameExtractor(folder_name=folder_name, video_path=rawVideo, medium=medium)

	noframes = len(os.listdir(folder_name))

	filename1 = folder_name + '/0.jpg'

	img1 = Image.open(filename1).convert('RGB')
	img1 = np.array(img1)
	gray1 = rgb2gray(img1).astype('float32')

	# to draw own bounding box, comment the next 4 lines and uncomment the 72nd line
	if 'easy' in rawVideo:
		bbox = bboxgen.get_bbox(img1, n)
	else:
		bbox = np.array([[[276, 457], [276, 518], [350, 518], [347, 456]]])

	# bbox = bboxgen.get_bbox(img1, n)
	x_index, y_index = getFeatures(gray1, bbox, nof)
	bbox = bbox.astype(float)

	for i in range(1, noframes):

		filename2 = folder_name + '/' + str(i) + '.jpg'
		img2 = Image.open(filename2).convert('RGB')
		img2 = np.array(img2)

		for b in range(len(bbox)):
			x_min = np.min(bbox[b, :, 0])
			x_max = np.max(bbox[b, :, 0])
			y_min = np.min(bbox[b, :, 1])
			y_max = np.max(bbox[b, :, 1])

			if len(x_index[x_index[:, b] != -1, b]) <= 3:
				if x_min < 0 or y_min < 0 or x_max >= img2.shape[1] or y_max >= img2.shape[0]:
					bbox[b] = 0 * bbox[b] - 1
				else:
					print('recomputing....')
					size = x_index[x_index[:, b] != -1, b].shape[0]
					x_old_index = x_index[x_index[:, b] != -1, b]
					y_old_index = y_index[y_index[:, b] != -1, b]
					gray1 = rgb2gray(img1).astype('float32')
					x_index, y_index = getFeatures(gray1, bbox.astype(int), nof)
					x_index[0:size, b] = x_old_index
					y_index[0:size, b] = y_old_index

		newXs, newYs = estimateAllTranslation(x_index, y_index, img1, img2)
		x_index[np.where(newXs == -1)] = -1
		y_index[np.where(newXs == -1)] = -1
		Xs, Ys, newbbox = applyGeometricTransformation(x_index, y_index, newXs, newYs, bbox)

		xmin = []
		ymin = []
		xmax = []
		ymax = []

		for b in range(len(newbbox)):
			max_x_y = newbbox[b].max(axis=0)
			min_x_y = newbbox[b].min(axis=0)

			Xs[Xs[:, b] < min_x_y[0], b] = -1

			Xs[Xs[:, b] > max_x_y[0], b] = -1
			Ys[Ys[:, b] < min_x_y[1], b] = -1
			Ys[Ys[:, b] > max_x_y[1], b] = -1

			Ys[Xs[:, b] == -1, b] = -1
			Xs[Ys[:, b] == -1, b] = -1

			xmin.append(min_x_y[0])
			xmax.append(max_x_y[0])
			ymin.append(min_x_y[1])
			ymax.append(max_x_y[1])

		x_index = np.copy(Xs)
		y_index = np.copy(Ys)
		img1 = np.copy(img2)
		bbox = np.copy(newbbox)

		plot_box(xmin, ymin, xmax, ymax, img2, i, Xs[Xs != -1], Ys[Ys != -1], folder_name + '_bounding_box')
		#plot_trajectory(Xs[Xs != -1], Ys[Ys != -1], img2, i, folder_name + '_trajectory')

	makeMovie(folder_name + '_bounding_box', folder_name + '_bb_output_gif', noframes)
	#makeMovie(folder_name + '_trajectory', folder_name + '_trajectory_output_gif', noframes)


if __name__ == '__main__':
	rawVideo = 'easy.mp4'
	output_gif = objectTracking(rawVideo)
