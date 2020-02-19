from utils import GaussianPDF_2D, rgb2gray
import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from estimateFeatureTranslation import estimateFeatureTranslation


def estimateAllTranslation(startXs, startYs, img1, img2):
	G = GaussianPDF_2D(0,1, 5, 5)
	dx, dy = np.gradient(G, axis=(1, 0))

	img1_gray = rgb2gray(img1)

	Ix = scipy.signal.convolve(img1_gray, dx, 'same')
	Iy = scipy.signal.convolve(img1_gray, dy, 'same')

	no_of_bounding_box = startXs.shape[1]

	newXs = np.zeros(startXs.shape)
	newYs = np.zeros(startYs.shape)

	newXs[:, :]=-1
	newYs[:, :]=-1

	for bounding_box_index in range(no_of_bounding_box):
		for i, (startX, startY) in enumerate(zip(startXs[:, bounding_box_index], startYs[:, bounding_box_index])):
			if startX != -1:
				newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)
				if ((newX>=img1.shape[1]) or (newY>=img1.shape[0]) or (newX<0) or (newY<0)):
					newX = -1
					newY = -1
			else:
				newX = -1
				newY = -1

			newXs[i, bounding_box_index] = newX
			newYs[i, bounding_box_index] = newY

	return newXs, newYs
