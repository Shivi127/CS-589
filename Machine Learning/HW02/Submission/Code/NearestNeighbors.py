# importing standard libraries
from sys import path
import os.path

# importing third party modules
# DO NOT import any other non-standard libraries: they may not be supported on Gradescope
import numpy as np
import sklearn.neighbors
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict
# importing local files
path.append(os.path.join(os.path.realpath(__file__), os.pardir))
import data_io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def best_hyperparams():
    # Modify this dictionary to contain the best hyperparameter value(s) you found
    params = {
                "n_neighbors": 5
             }
    return params

# This function will be executed by Gradescope
# You should probably also call it from main()
# Here is where you should create and train your decision tree
# train_x and train_y are the training data (numpy arrays)
# hyperparams is a dictionary of hyperparameters (in this case just 'n_neighbors')
#  see best_hyperparams() above for an example
# This function should return a trained decision tree
def create_model(train_x, train_y, hyperparams):

    # 'classifier' should be the decision tree model you train
    # See here: http://lmgtfy.com/?q=knn+classifier+sklearn
    classifier = KNeighborsClassifier(n_neighbors= hyperparams['n_neighbors'])
    classifier.fit(train_x,train_y)

    return classifier

# This function will be run when this file is executed.
# It will not be executed on Gradescope
# Use this function to run your experiements
def main():

    # returns the mean L0 norm between the labels and predictions given
    # in other words, the accuracy of your model on the test data
    # labels is the ground truth labels
    # predictions is your predictions on the data
    # outputs a float between 0 and 1 (technically inclusive, but it probably won't be 0 or 1)
    def accuracy(labels, predictions):
        return ((labels == predictions).astype(int)).mean()

    train_x, train_y = data_io.read_image_data()

    ##### YOUR CODE HERE ######

    # You'll probably want to change this line for cross-validation
    # This is just an example of a call to create_model, defined above


    classifier = create_model(train_x, train_y, best_hyperparams())

    # This is a dummy test data set.
    # They're all just random noise, not ships or horses or frogs or trucks,
    # so your classifier should and will perform poorly on these.
    # These lines of code are for demonstration purposes only.
    test_x = np.random.randn(*train_x.shape)
    test_y = np.random.randint(0, 4, size=train_y.shape)

    paramss = [3, 5, 7, 9, 11]
    accuracy_result = []
    for p in paramss:
        print("I am running for ", p)
        clf = KNeighborsClassifier(n_neighbors=p)
        clf.fit(train_x, train_y)
        predictions = cross_val_predict(clf, train_x, train_y, cv=5)
        accuracy_result.append(1- accuracy(predictions, test_y))

    print("Accuracy ", accuracy_result)
    index_min = np.argmin(accuracy_result)
    print("Best Parameter is ", paramss[index_min])



    # parametersG = {'n_neighbors': params}
    #
    # print("Performing GridSearch")
    # clf = KNeighborsClassifier()
    # grid = GridSearchCV(clf, parametersG, cv=5)
    # grid.fit(train_x, train_y)
    # print("Best Estimator", grid.best_estimator_, "Best Score", grid.best_score_)

    predictions = classifier.predict(test_x)

    # prints the accuracy of your predictions
    # should be about 0.25 for the random test data above
    print(accuracy(predictions, test_y))


    # While it's good practice to import at the beginning of a file,
    # since Gradescope won't run this function,
    # you can import anything you want here.
    # matplotlib's pyplot is a good tool for making plots.
    # You can install it here: http://lmgtfy.com/?q=matplotlib+download+python3+anaconda
    # You can read about it here: http://lmgtfy.com/?q=matplotlib+pyplot+tutorial
    from matplotlib import pyplot

# This runs 'main' upon loading the script
if __name__ == "__main__":
    main()
