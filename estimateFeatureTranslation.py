import numpy as np
from utils import rgb2gray
from PIL import Image
from utils import interp2


def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
	final_u = 0
	final_v = 0

	img1_gray = rgb2gray(img1)
	img2_gray = rgb2gray(img2)

	It = img2_gray - img1_gray

	x, y = np.meshgrid(np.arange(-5, 6), np.arange(- 5, 6))
	ones = np.ones(x.shape)

	if startY - 5 >= 0 and startY + 5 < Ix.shape[0] and startX - 5 >= 0 and startX + 5 < Ix.shape[1]:
		xx = startX * ones + x
		yy = startY * ones + y

		Ix_patch_of_100_pixels = interp2(Ix, xx, yy)
		Iy_patch_of_100_pixels = interp2(Iy, xx, yy)

		Ixx = np.sum(Ix_patch_of_100_pixels * Ix_patch_of_100_pixels)
		Iyy = np.sum(Iy_patch_of_100_pixels * Iy_patch_of_100_pixels)
		Ixy = np.sum(Ix_patch_of_100_pixels * Iy_patch_of_100_pixels)
		img1_patch = interp2(img1_gray, xx, yy)
		A = np.array([[Ixx, Ixy], [Ixy, Iyy]])
	else:
		return -1, -1

	for t in range(8):
		if startY - 5 >= 0 and startY + 5 < Ix.shape[0] and startX - 5 >= 0 and startX + 5 < Ix.shape[1]:

			xx = startX * ones + x
			yy = startY * ones + y

			#print('shape of x', xx.shape, yy.shape)
			img2_patch=interp2(img2_gray, xx,yy)

			It_patch_of_100_pixels = img2_patch - img1_patch

			#print('sum', np.sum(It_patch_of_100_pixels))

			Ixt = np.sum(Ix_patch_of_100_pixels * It_patch_of_100_pixels)
			Iyt = np.sum(Iy_patch_of_100_pixels * It_patch_of_100_pixels)


			b = np.array([-Ixt, -Iyt]).reshape(2, 1)
			A_inv = np.linalg.inv(A)
			solution = np.matmul(A_inv, b)

			#print('solution', startX, solution)

			u = solution[0, 0]
			v = solution[1, 0]
			final_u +=u
			final_v+=v
			startX, startY = startX + u, startY + v
		else:
			return -1,-1

			#print('u v', u, v)

			#print('uv;',startX + u, startY + v)

	#print('final uv', final_u, final_v)
	return startX, startY

