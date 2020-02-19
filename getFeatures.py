import numpy as np
import cv2
from PIL import Image
import bboxgen

def rgb2gray(I_rgb):
	r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
	I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return I_gray

def getFeatures(img,bbox,nof):

	x_index = -1 * np.ones((nof,bbox.shape[0])).astype(int)
	y_index = -1 * np.ones((nof, bbox.shape[0])).astype(int)
	for i in range(bbox.shape[0]):
		xmin = np.min(bbox[i,:,0])
		xmax = np.max(bbox[i,:,0])
		ymin = np.min(bbox[i, :, 1])
		ymax = np.max(bbox[i, :, 1])
		xx,yy = np.meshgrid(np.arange(xmin,xmax+1),np.arange(ymin,ymax+1))
		boximg = img[yy,xx]
		corners = cv2.goodFeaturesToTrack(boximg, nof, 0.01, 10)
		corners = np.int0(corners)
		k=0
		for j in corners:
			x_index[k,i],y_index[k,i] = j.ravel()
			k=k+1

		x_index[:,i] = x_index[:,i]+xmin
		y_index[:, i] = y_index[:, i] + ymin

	return x_index,y_index

if __name__ == '__main__':

	filename1 = 'easyFrames/easy0.jpg'
	img1 = Image.open(filename1).convert('RGB')
	img1 = np.array(img1)
	gray1 = rgb2gray(img1).astype('float32')

	filename2 = 'easyFrames/easy1.jpg'
	img2 = Image.open(filename2).convert('RGB')
	img2 = np.array(img2)
	gray2 = rgb2gray(img2).astype('float32')
	n = 2
	nof =10
	bbox = bboxgen.get_bbox(img1,n)
	x_index,y_index = getFeatures(gray1,bbox,nof)
	print(x_index)
	print(y_index)