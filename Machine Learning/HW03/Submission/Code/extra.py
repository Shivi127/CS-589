# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score

############################################################################
# Read in train and test synthetic data
def read_synthetic_data():
	print('Reading synthetic data ...')
	train_x = np.loadtxt('../../Data/Synthetic/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Synthetic/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Synthetic/data_test.txt', delimiter = ',', dtype=float)
	test_y = np.loadtxt('../../Data/Synthetic/label_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x, test_y)

############################################################################
# Read in train and test credit card data
def read_creditcard_data():
	print('Reading credit card data ...')
	train_x = np.loadtxt('../../Data/CreditCard/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/CreditCard/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/CreditCard/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Read in train and test tumor data
def read_tumor_data():
	print('Reading tumor data ...')
	train_x = np.loadtxt('../../Data/Tumor/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Tumor/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Tumor/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Compute MSE
def compute_MSE(y, y_hat):
	# mean squared error
	return np.mean(np.power(y - y_hat, 2))

############################################################################

train_x, train_y, test_x, test_y = read_synthetic_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

train_x, train_y, test_x  = read_creditcard_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values to compute MSE
test_y = np.random.rand(test_x.shape[0], train_y.shape[1])
predicted_y = np.random.rand(test_x.shape[0], train_y.shape[1])
print('DUMMY MSE=%0.4f' % compute_MSE(test_y, predicted_y))

# Output file location
file_name = '../Predictions/CreditCard/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, True)

train_x, train_y, test_x  = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values to compute accuracy
test_y = np.random.randint(0, 2, (test_x.shape[0], 1))
predicted_y = np.random.randint(0, 2, (test_x.shape[0], 1))
print('DUMMY Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))

# # Output file location
# file_name = '../Predictions/Tumor/best.csv'
# # Writing output in Kaggle format
# print('Writing output to ', file_name)
# kaggle.kaggleize(predicted_y, file_name, False)

################################################################################################
#


# ############################### Tumor Polynomial ####################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#
# C = [1,0.01,0.0001]
# gamma = [1,0.01,0.001]
# degree = [3,5,1]
# error_results =[]
#
# train_x, train_y, test_x = read_tumor_data()
# # X_train, X_test, y_train, y_test = train_test_split( train_x, train_y, random_state =0)
#
# for c in C:
# 	for g in gamma:
# 		for d in degree:
# 			print("Gamma: ", g ," C: ",c," Polynomial: ",d)
# 			clf = SVC(gamma=g, C=c, kernel='poly', degree=d)
# 			scores = cross_val_score(clf, train_x, train_y, cv=5)
# 			# print("Scores",scores)
# 			print("The Accuracy for this classifier is: " , scores.mean())
# 			print(" ")
# 			error_results.append(scores.mean())
#
# ############################### TumorRBF ####################
# from sklearn.svm import SVC
# C = [1,0.01,0.0001]
# gamma = [1,0.01,0.001]
# degree = [3,5,1]
# error_resultsrbf =[]
#
#
# for c in C:
# 	for g in gamma:
# 		print("Gamma: ", g ," C: ",c)
# 		clf = SVC(gamma=g ,C = c, kernel='rbf')
# 		scores = cross_val_score(clf, train_x, train_y, cv=5)
# 		print("The Accuracy for this classifier is: " , scores.mean())
# 		print(" ")
# 		error_results.append(scores.mean())
#
#
# import operator
# index, value = max(enumerate(error_results), key=operator.itemgetter(1))
# print("Max_Value", index, value)
#
#
#
######### Final
clf = SVC(gamma =0.0001, C=1)

clf.fit(train_x,train_y)
predictions = clf.predict(test_x)

# Output file location
file_name = '../Predictions/Tumor/best_extra_credit.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predictions, file_name, False)

