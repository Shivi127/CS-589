# Import python modules
import numpy as np
import kaggle

# Read in train and test data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	train_x = np.loadtxt('../../Data/PowerOutput/data_train.txt')
	train_y = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
	test_x = np.loadtxt('../../Data/PowerOutput/data_test.txt')

	return (train_x, train_y, test_x)

def read_data_localization_indoors():
	print('Reading indoor localization dataset ...')
	train_x = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
	train_y = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
	test_x = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()


def plotGraph(param_range,tvals):
	import matplotlib.pyplot as plt
	plt.plot(param_range,tvals)
	plt.xlabel('Parameter for max_depth of the classifier') 
	plt.ylabel('Time taken for 5-fold classification (in milliseconds) ') 
	plt.title('Time VS Max Depth for Binary Tree Classifier') 
	plt.show() 

#####################################  Decision Tree   #######################################
# Question 1 b) Decision Tree: use the same code just different Data Set
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict
import numpy as np
import time

param_range = [3,6,9,12,15]
error = []
runtime = []
for p in param_range:
    clf = DecisionTreeRegressor (max_depth=p)
    start = time.clock()
    predicted = cross_val_predict(clf, train_x, train_y, cv=5)
    run = time.clock() - start
    runtime.append(run*60)
    error.append(compute_error(predicted,train_y))

print (error)
print("Runtimes in milliseconds", runtime)
print ("Best Classifier Depth ", param_range[np.argmin(error)])

# Plotting Decision Tree Runtime
import matplotlib.pyplot as plt
plt.plot(param_range,runtime)
plt.xlabel('Parameter for max_depth of the classifier') 
plt.ylabel('Time taken for 5-fold classification (in milliseconds) ') 
plt.title('Time VS Max Depth for BTree Classifier for Power Plant Dataset') 
plt.show() 

####################################  K-Nearest Neighbors   #####################################

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
import numpy as np
import time

knn_params = [3, 5, 10, 20, 25]
error = []
runtime = []
for p in knn_params:
    clf = KNeighborsRegressor(n_neighbors =p)
    start = time.clock()
    predicted = cross_val_predict(clf, train_x, train_y, cv=5)
    run = time.clock() - start
    runtime.append(run*60)
    error.append(compute_error(predicted,train_y))

print ("Error", error)
print()
print("Runtimes in milliseconds", runtime)
print()
print ("Best Classifier: n_neighbors ", knn_params[np.argmin(error)])

# Plotting for KNN
plt.plot(param_range,runtime)
plt.xlabel('Parameter for N_Neighbors') 
plt.ylabel('Time taken for 5-fold classification (in milliseconds) ') 
plt.title('Time VS N_Neighbors for Power Plant Dataset') 
plt.show() 

################################     Linear Regression    #######################################


######### RIDGE ######################
from sklearn.linear_model import Ridge
alp = [10e-6, 10e-4, 10e-2, 1, 10]


from sklearn.model_selection import cross_val_predict
import numpy as np
import time

error = []
runtime = []
for a in alp:
    clf = Ridge(alpha= a)
    start = time.clock()
    predicted = cross_val_predict(clf, train_x, train_y, cv=5)
    run = time.clock() - start
    runtime.append(run*60)
    error.append(compute_error(predicted,train_y))

print ("Error", error)
print()
print("Runtimes in milliseconds", runtime)
print()
print ("Best Classifier: alpha ", alp[np.argmin(error)])


dtreeplant =  Ridge(alpha= alp[np.argmin(error)])
dtreeplant.fit(train_x, train_y)
predictionplant = dtreeplant.predict(test_x)


######### LASSO ######################
from sklearn.linear_model import Lasso
alp = [10e-6, 10e-4, 10e-2, 1, 10]


from sklearn.model_selection import cross_val_predict
import numpy as np
import time

error = []
runtime = []
for a in alp:
    clf = Lasso(alpha= a)
    start = time.clock()
    predicted = cross_val_predict(clf, train_x, train_y, cv=5)
    run = time.clock() - start
    runtime.append(run*60)
    error.append(compute_error(predicted,train_y))

print ("Error", error)
print()
print("Runtimes in milliseconds", runtime)
print()
print ("Best Classifier: alpha ", alp[np.argmin(error)])


dtreeplant =  Ridge(alpha= alp[np.argmin(error)])
dtreeplant.fit(train_x, train_y)
predictionplant = dtreeplant.predict(test_x)
################################     Question 5: Trying to find the best estimator    #######################################
# Part 1: I choose to go with a decision tree as it was giving the best performance in the previous parts
# Some of things I tried
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict
import numpy as np
import time


# features =[1,2,3,4]
param_range = [1,2,3,4,5,6,7,8,9,12,15,20]
error = []
runtime = []
for p in param_range:
#     for f in features:
    clf = DecisionTreeRegressor (max_depth=p, max_features =5, min_samples_split= 15)
    start = time.clock()
    predicted = cross_val_predict(clf, train_x, train_y, cv=5)
    run = time.clock() - start
    runtime.append(run*60)
    error.append(compute_error(predicted,train_y))

print (error)
# print("Runtimes in milliseconds", runtime)
print ("Best Classifier Depth ", param_range[np.argmin(error)])

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict
import numpy as np
import time


# features =[1,2,3,4]
param_range = [1,2,3,4,5,6,7,8,9,12,15,20]
error = []
runtime = []
for p in param_range:
#     for f in features:
    clf = DecisionTreeRegressor (max_depth=p,min_samples_leaf= 5, min_impurity_decrease=1.1)
    start = time.clock()
    predicted = cross_val_predict(clf, train_x, train_y, cv=8)
    run = time.clock() - start
    runtime.append(run*60)
    error.append(compute_error(predicted,train_y))

print (error)
# print("Runtimes in milliseconds", runtime)
print ("Best Classifier Depth ", param_range[np.argmin(error)])




# Part 2: 
# For the IndoorDataset I chose to go with KNN and tried different parameters with it the best solution was
# Some things I tried
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
import numpy as np
import time

knn_params = [3,4,5,6,8,9,10,17,25,30,35,40,200]
error = []
runtime = []
for p in knn_params:
    clf = KNeighborsRegressor(n_neighbors =p, weights= 'uniform', p=2, n_jobs=-1,leaf_size = 15)
    start = time.clock()
    predicted = cross_val_predict(clf, train_x, train_y, cv= 8)
    run = time.clock() - start
    runtime.append(run*60)
    error.append(compute_error(predicted,train_y))

print ("Error", error)
print()
print("Runtimes in milliseconds", runtime)
print()
print ("Best Classifier: n_neighbors ", knn_params[np.argmin(error)])


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
import numpy as np
import time

knn_params = [3,4,5,6,8,9,10,17,25,30,35,40,200]
error = []
runtime = []
for p in knn_params:
    clf = KNeighborsRegressor(n_neighbors =p, weights= 'distance', p=2, n_jobs=-1,leaf_size = 15)
    start = time.clock()
    predicted = cross_val_predict(clf, train_x, train_y, cv= 8)
    run = time.clock() - start
    runtime.append(run*60)
    error.append(compute_error(predicted,train_y))

print ("Error", error)
print()
print("Runtimes in milliseconds", runtime)
print()
print ("Best Classifier: n_neighbors ", knn_params[np.argmin(error)])


################################     KAGGLE Submission    #######################################

# Create dummy test output values for IndoorLocalization
predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

# Create dummy test output values for PowerOutput
predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name = '../Predictions/PowerOutput/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
