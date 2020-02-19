import numpy as np
from skimage import transform as tf
import cv2

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):

	no_of_bounding_box = startXs.shape[1]
	new_bounding_box = np.copy(bbox).astype(float)

	Xs = np.zeros(startXs.shape)
	Ys = np.zeros(startYs.shape)
	Xs[:, :] = -1
	Ys[:, :] = -1

	for bounding_box_index in range(no_of_bounding_box):
		startX_without_neg_ones = startXs[:, bounding_box_index]
		startX_without_neg_ones = startX_without_neg_ones[startX_without_neg_ones!=-1]

		if len(startX_without_neg_ones) >= 2:

			startY_without_neg_ones = startYs[:, bounding_box_index]
			startY_without_neg_ones = startY_without_neg_ones[startY_without_neg_ones != -1]

			newX_without_neg_ones = newXs[:, bounding_box_index]
			newX_without_neg_ones = newX_without_neg_ones[newX_without_neg_ones != -1]

			newY_without_neg_ones = newYs[:, bounding_box_index]
			newY_without_neg_ones = newY_without_neg_ones[newY_without_neg_ones != -1]

			matrix_start = np.vstack((startX_without_neg_ones, startY_without_neg_ones)).T
			matrix_new = np.vstack((newX_without_neg_ones, newY_without_neg_ones)).T

			tform = tf.estimate_transform('similarity', matrix_start, matrix_new)

			start_translated = tform(matrix_start)

			new_bounding_box[bounding_box_index] = tform(bbox[bounding_box_index])

			s = (matrix_new - start_translated) ** 2

			subtract_new = np.sqrt(s.sum(axis=1))
			features_to_be_kept = subtract_new < 5

			finalXs = matrix_new[features_to_be_kept][:, 0]
			finalYs = matrix_new[features_to_be_kept][:, 1]

			Xs[:len(finalXs), bounding_box_index] = finalXs
			Ys[:len(finalYs), bounding_box_index] = finalYs

		else:
			Xs[:, bounding_box_index] = -1
			Ys[:, bounding_box_index] = -1

	return Xs, Ys, new_bounding_box