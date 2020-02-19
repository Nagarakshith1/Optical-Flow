import imageio

images = []


def makeMovie(foldername, outputfilename, noframes):
	for i in range(1, noframes):
		filename = foldername + '/frame_tracked_' + str(i) + '.png'
		images.append(imageio.imread(filename))

	imageio.mimsave(outputfilename + '.gif', images)
