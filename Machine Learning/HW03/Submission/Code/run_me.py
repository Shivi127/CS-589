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
#
#
# # ############################### Tumor Polynomial ####################
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_val_score
#
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
# ######### Final
# clf = SVC(gamma= 1 ,C = 1, kernel='rbf')
#
# clf.fit(train_x,train_y)
# predictions = clf.predict(test_x)
#
# # Output file location
# file_name = '../Predictions/Tumor/best.csv'
# # Writing output in Kaggle format
# print('Writing output to ', file_name)
# kaggle.kaggleize(predictions, file_name, False)
#
#
# # #################################### CREDIT CARD #################################
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_val_predict
# from sklearn import cross_validation as CV
#
#
# def calldefaultrbf (a):
# 	print("Alpha",a)
# 	clf = KernelRidge(alpha=a, kernel ='rbf')
# 	scores = CV.cross_val_score(clf, train_x, train_y, cv=5, scoring='mean_squared_error')
# 	print("Scores",abs(scores.mean()))
# 	print("")
# 	print("")
# 	answers.append(abs(scores.mean()))
#
# train_x, train_y, test_x = read_creditcard_data()
#
# alpha = [1, 0.0001]
# gamma = [1, 0.001]
# degree =[3,1]
# answers = []
# for a in alpha:
# 	calldefaultrbf(a)
# 	for g in gamma:
# 		print("Alpha",a,"Gamma",g)
# 		print(" ")
# 		clf = KernelRidge(alpha=a, gamma=g, kernel ='rbf')
# 		scores = CV.cross_val_score(clf, train_x, train_y, cv=5, scoring='mean_squared_error')
# 		print("Scores",abs(scores.mean()))
# 		print("")
# 		print("")
# 		answers.append(abs(scores.mean()))
#
#
# def calldefaultpoly(a,d):
# 	print("Alpha",a,"Degree",d)
# 	print(" ")
# 	clf = KernelRidge(alpha=a, kernel ='poly',degree=d)
# 	scores = CV.cross_val_score(clf, train_x, train_y, cv=5, scoring='mean_squared_error')
# 	print("Scores",abs(scores.mean()))
# 	print("")
# 	print("")
# 	answers.append(abs(scores.mean()))
#
#
#
#
# for a in alpha:
# 	for d in degree:
# 		calldefaultpoly(a,d)
# 		for g in gamma:
# 			print("Alpha",a,"Gamma",g,"Degree",d)
# 			print(" ")
# 			clf = KernelRidge(alpha=a, gamma=g, kernel ='poly',degree=d)
# 			scores = CV.cross_val_score(clf, train_x, train_y, cv=5, scoring='mean_squared_error')
# 			print("Scores",abs(scores.mean()))
# 			print("")
# 			print("")
# 			answers.append(abs(scores.mean()))
#
#
#
#
# print("Ans",answers, " Length ",len(answers))
# import operator
# index1, value1 = min(enumerate(answers), key=operator.itemgetter(1))
# print("Min_Value", index1, value1)
#
# ######### Final ###################
# clf = KernelRidge(alpha=0.0001,gamma=1, kernel ='poly',degree=3)
#
#
# clf.fit(train_x,train_y)
# predictions = clf.predict(test_x)
#
# # Output file location
# file_name = '../Predictions/CreditCard/best.csv'
# # Writing output in Kaggle format
# print('Writing output to ', file_name)
# kaggle.kaggleize(predictions, file_name, True)



########################################## SYNTHETIC DATA ###########################


from KernelRidgeRegression import RidgeRegression

train_x, train_y, test_x, test_y = read_synthetic_data()
print("################## Running Kernel Ridge - Polynomial  ###############")

kridge = RidgeRegression()
order = [1,2,4,6]


MSE_Poly_Kernel_Erorrs = []

Kpredictions_poly = {}


for o in order:
	Kpredictions_poly[o]=[]
	kridge.kernel = lambda a,b : (1+a*b)**o
	kridge.fit(train_x,train_y)
	predictions = kridge.predict(test_x)
	# print("Predict",predictions,len(predictions))
	Kpredictions_poly[o].append(predictions)
	error = compute_MSE(test_y,predictions)
	MSE_Poly_Kernel_Erorrs.append([o,error])


print("MSE",MSE_Poly_Kernel_Erorrs)
# print(Kpredictions_poly.keys(),"Yippee")


# ############################################################################################

print("################## Running Basis Ridge - Polynomial  ###############")
from sklearn.linear_model import Ridge

kridge2 = Ridge(alpha=0.1)
order = [1,2,4,6]
BERRpredictions_poly = {}
BERR_Poly_Errors = []

for o in order:
	BERRpredictions_poly[o] = []
	BERR = lambda data : np.array([[rowval**i for i in range(o+1)] for rowval in data])
	kridge2.fit(BERR(train_x),train_y)
	predictions = kridge2.predict(BERR(test_x))
	BERRpredictions_poly[o].append( predictions)
	error = compute_MSE(test_y, predictions)
	BERR_Poly_Errors.append([o, error])

print(BERR_Poly_Errors,"Poop 2")
#
#
# ################################################################################################
#
print("################## Running Kernel Ridge - Trignometric  ###############")
from functools import reduce

order = [3,5,10]
delta = 0.5
KR_Trigg_Errors =[]
KR_Trigg_Results ={}

clf = RidgeRegression()

for o in order:

	# Kfunc = lambda a,b : 1 + for i in range(o+1):
	# 						sins = np.sin(i*delta*a)* np.sin(i*delta*b)
	# 						coss = np.cos(i*delta*a)* np.cos(i*delta*b)

	KR_Trigg_Results[o]=[]
	clf.kernel = lambda a, b: 1 + reduce(lambda x, y: x + y, map(
		lambda k: np.sin(k * delta * a) * np.sin(k * delta * b) + np.cos(k * delta * a) * np.cos(
			k * delta * b), [j for j in range(1, o + 1)]))
	clf.fit(train_x,train_y)
	predictions = clf.predict(test_x)
	KR_Trigg_Results[o].append(predictions)
	error = compute_MSE(test_y, predictions)
	KR_Trigg_Errors.append([o, error])

print(KR_Trigg_Errors,"Poop 3")

#
# ################################################################################################
#
print("################## Running Basis Ridge - Trignometric  ###############")


def BERRTrigExpansion(data, order):
	expansion = []
	delta = 0.5
	for row in data:
		hx = [1]
		for j in range(1,order+1):
			hx.append(np.sin(j*delta*row))
			hx.append(np.cos(j*delta*row))
		expansion.append(hx)
	return np.array(expansion)


from sklearn.linear_model import Ridge

kridge3 = Ridge(alpha=0.1)
order = [3,5,10]
BERRpredictions_trigg ={}
BERR_Trigg_Errors = []

for o in order:
	BERRpredictions_trigg[o]=[]
	BERR = lambda data : np.array([[rowval**i for i in range(o+1)] for rowval in data])
	kridge2.fit(BERRTrigExpansion(train_x,o),train_y)
	predictions = kridge2.predict(BERRTrigExpansion(test_x,o))
	BERRpredictions_trigg[o].append(predictions)
	error = compute_MSE(test_y, predictions)
	BERR_Trigg_Errors.append([o, error])

print(BERR_Trigg_Errors,"Poop 4")


################### 	PLOTTING     #######################
import matplotlib.pyplot as plt


def plot():

	print("--Writing plot for 1d to ../Figures/question1dPlot.jpeg--\n")
	# 4 ROWS AND 2 col
	fig = plt.figure()
	print("Poly 2")
	ax1 = fig.add_subplot(421)
	ax1.scatter(test_x,test_y,marker="*",c='b')
	ax1.scatter(test_x, Kpredictions_poly[2], marker="o", color='r')
	ax1.set_title("KRRS Polynomial 2")
	ax1.set_xlabel('Test X')
	ax1.set_ylabel('True/Predicted Y')


	print("Poly 6")
	ax2 = fig.add_subplot(422)
	ax2.scatter(test_x, test_y, marker="*", c='b')
	ax2.scatter(test_x, Kpredictions_poly[6], marker="o", color='r')
	ax2.set_title("KRRS Polynomial 6 ")
	ax2.set_xlabel('Test X')
	ax2.set_ylabel('True/Predicted Y')




	print("I am in Axes 3, 5")

	ax3 = fig.add_subplot(423)
	ax3.scatter(test_x, test_y, marker="*", c='b')
	ax3.scatter(test_x, KR_Trigg_Results[5], marker="o", color='r')
	ax3.set_title("KRRS Trignomeric order 5 ")
	ax3.set_xlabel('Test X')
	ax3.set_ylabel('True/Predicted Y')


	print("I am in Axes, Poly 10")

	ax4 = fig.add_subplot(424)
	ax4.scatter(test_x, test_y, marker="*", c='b')
	ax4.scatter(test_x, KR_Trigg_Results[10], marker="o", color='r')
	ax4.set_title("KRRS Trignomeric order 10")
	ax4.set_xlabel('Test X')
	ax4.set_ylabel('True/Predicted Y')

	print("I am in Axes, Poly 2, Basis")

	ax5 = fig.add_subplot(425)
	ax5.scatter(test_x, test_y, marker="*", c='b')
	ax5.scatter(test_x, BERRpredictions_poly[2], marker="o", color='r')
	ax5.set_title("BERR - Polynomial order 2")
	ax5.set_xlabel('Test X')
	ax5.set_ylabel('True/Predicted Y')

	print("I am in Axes, Poly 6, Basis")

	ax6 = fig.add_subplot(426)
	ax6.scatter(test_x, test_y, marker="*", c='b')
	ax6.scatter(test_x, BERRpredictions_poly[6], marker="o", color='r')
	ax6.set_title("BERR - Polynomial order 6")
	ax6.set_xlabel('Test X')
	ax6.set_ylabel('True/Predicted Y')

	# 5,10
	ax7 = fig.add_subplot(427)
	ax7.scatter(test_x, test_y, marker="*", c='b')
	ax7.scatter(test_x, BERRpredictions_trigg[5], marker="o", color='r')
	ax7.set_title("BERR - Trignometric Polynomial order 5")
	ax7.set_xlabel('Test X')
	ax7.set_ylabel('True/Predicted Y')

	ax8 = fig.add_subplot(428)
	ax8.scatter(test_x, test_y, marker="*", c='b')
	ax8.scatter(test_x, BERRpredictions_trigg[10], marker="o", color='r')
	ax8.set_title("BERR - Trignometric Polynomial order 10")
	ax8.set_xlabel('Test X')
	ax8.set_ylabel('True/Predicted Y')




	plt.tight_layout()
	fig.suptitle('Lambda : 0.1, delta = 0.5')
	plt.savefig("../Figures/MLPlot.jpeg")
	plt.show()

	print("Done")




# plot()