# Import modules

import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def read_scene():
	data_x = misc.imread('../../Data/Scene/times_square.jpg')

	return (data_x)


title = ["Original", 
		"KMeans Clustering (n=2)",
		"KMeans Clustering (n=5)",
		"KMeans Clustering (n=10)",
		"KMeans Clustering (n=25)",
		"KMeans Clustering (n=50)",
		"KMeans Clustering (n=75)",
		"KMeans Clustering (n=100)",
		"KMeans Clustering (n=200)"]

def plot_image(data,title):
	# pass in an array of plot data + array of labels
	print("I am in plotting")
	fig = plt.figure()

	cluster_size = [0, 2, 5, 10, 25, 50, 75, 100, 200]

	ax1 = fig.add_subplot(331)
	ax2 = fig.add_subplot(332)
	ax3 = fig.add_subplot(333)
	ax4 = fig.add_subplot(334)
	ax5 = fig.add_subplot(335)
	ax6 = fig.add_subplot(336)
	ax7 = fig.add_subplot(337)
	ax8 = fig.add_subplot(338)
	ax9 = fig.add_subplot(339)

	axis = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

	for i,d in enumerate(data):
		print("I am plotting for ", i)
		axis[i].imshow(d)
		axis[i].set_title(title[i])
		axis[i].set_xlabel("k_value"+str(cluster_size[i]))
	plt.tight_layout()
	plt.show()


def plot_the_error(error,title,ylabel):
	cluster_size = [ 2, 5, 10, 25, 50, 75, 100, 200]
	plt.plot(cluster_size , error,'-o')
	plt.title(title)
	plt.xlabel("Number of Clusters")
	plt.ylabel(ylabel)
	plt.show()

if __name__ == '__main__':
	
	################################################
	# K-Means
	cluster_size = [2, 5, 10, 25, 50, 75, 100, 200]
	# cluster_size = [2]

	data_x = read_scene()
	print('X = ', data_x.shape)

	flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
	# print('Flattened image = ', flattened_image.shape)
	# print("X val ", data_x[0], data_x[0].shape)
	print('Implement k-means here ...')


	plots_array = []
	sse_errors =[]
	plots_array.append(data_x)
	x = flattened_image
	totalpixels = x.shape[0]
	compressions_array = []
	for c in cluster_size:
		print("Running for cluster size", c)

		temp_image = np.zeros(flattened_image.shape)
		clf = KMeans(n_clusters= c, n_jobs= -1)
		clf.fit(flattened_image)
		cluster_centers = clf.cluster_centers_
		for i in range(flattened_image.shape[0]):
			temp_image[i] = cluster_centers[clf.labels_[i]]

		sse_errors.append(np.log(np.sum(np.square(temp_image-x))))
		temp_image = temp_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
		plots_array.append(temp_image/255)
		
		compression_rate = (np.ceil(np.log2(c))*totalpixels + c * 3 * 32) / (24*totalpixels)
		compressions_array.append([c,compression_rate])
	plot_image(plots_array,title)
	print("I am onto plotting")
	plot_the_error(sse_errors,"Error vs k-Clusters","Squared Error in log space")
	print("The Compression Rate is",compressions_array)

	# reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
	# print('Reconstructed image = ', reconstructed_image.shape)

