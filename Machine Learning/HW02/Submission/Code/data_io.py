# Import python modules
import numpy as np
from sklearn.metrics import accuracy_score

# Read in train and test data
def read_image_data():
	print('Reading image data ...')
	temp = np.load('../../Data/data_train.npz')
	train_x = temp['data_train']
	temp = np.load('../../Data/labels_train.npz')
	train_y = temp['labels_train']
	print('The shape of the training data is '
		  '{:d} samples by {:d} features'.format(*train_x.shape))
	return (train_x, train_y)
