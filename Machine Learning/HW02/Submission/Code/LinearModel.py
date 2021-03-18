# importing standard libraries
from sys import path
import os.path

# importing third party modules
# DO NOT import any other non-standard libraries: they may not be supported on Gradescope
import numpy as np
import sklearn.model_selection

# importing local files
path.append(os.path.join(os.path.realpath(__file__), os.pardir))
import data_io
from sklearn.model_selection import cross_val_predict

def best_hyperparams():
    # Modify this dictionary to contain the best hyperparameter value(s) you found
    params = {
                "_lambda": 100.0,
                "_alpha": 10.0
             }
    return params

class LinearModel:

    def __init__(self, _lambda=0, _alpha=1.0):
        self._lambda = _lambda # The regularization constant
        self._alpha = _alpha # the SGD update constant

    # fits this classifier (i.e. updates self._beta) using SGD
    # takes an array X such that X.shape = (n, P),
    #  where n is the number of samples, and P is the number of features
    # updates self._beta and returns self
    def SGD_fit(self, X, y, epochs=5):

        ##### IMPLEMENT THIS FUNCTION #####
        # In stochastic we update for every sample
        self._beta = np.zeros(X[0].shape)
        for e in range(epochs):
            n = X.shape[0] #number of training samples
            for i in range(n):
                gradient = self.loss_grad(X[i],y[i],n)
                new_grad = [gr * self._alpha for gr in gradient]
                self._beta = self._beta - new_grad
        return self

    # predict class labels (0 for non-ships, 1 for ships)
    # takes an array X such that X.shape = (n, P),
    #  where n is the number of samples, and P is the number of features
    # also takes an optional beta parameter that can be used instead of self._beta
    # returns an array of y_i values (0 or 1) with shape (n,)
    def predict(self, X, beta=None):
        if beta is None:
            beta = self._beta
        # You don't have to edit this function, but it does call predict_prob(),
        #  so it won't work properly until you've finished that function
        p = self.predict_prob(X, beta=beta)
        return np.greater(p, 0.5*np.ones(p.shape)).astype(int)

    # return how confident the model is that each input is a ship
    # takes an array X such that X.shape = (n, P),
    #  where n is the number of samples, and P is the number of features
    # also takes an optional beta parameter that can be used instead of self._beta
    # returns an array of p_i values (between 0 and 1) with shape (n,)
    def predict_prob(self, X, beta=None):
        if beta is None:
            beta = self._beta

        ##### IMPLEMENT THIS FUNCTION #####
        ##### BE CAREFUL OF MATRIX DIMENSIONS!!! #####

        result = np.empty(shape= X.shape[0],)
        dotproduct = np.dot(X, beta.T)
        for i,d in enumerate(dotproduct):
            frac = np.exp(-1 * d)
            p_value = 1 / (1 + frac)
            result[i] = p_value
        return result

    # computes the loss function of one data point
    # takes
    #  x, a feature vector as an array with shape (P,),
    #  y, the label for that example (0 or 1), and
    #  n, the number of samples in the entire data set
    # also takes an optional beta parameter that can be used instead of self._beta
    # returns the value for the loss, a float
    def loss(self, x, y, n, beta=None):
        if beta is None:
            beta = self._beta

        p = float(self.predict_prob(np.reshape(x, (1, x.shape[0])), beta=beta))

        return -1*(y*np.log(p) + (1-y)*np.log(1-p))

    # computes the gradient of the loss function at a given data point
    # takes x, a feature vector as array with shape (P,),
    #  y, the example's label (0 or 1), and
    #  n, the number of samples in the entire data set
    # also takes an optional beta parameter that can be used instead of self._beta
    # returns the value for the loss, a (P,)-shaped array
    def loss_grad(self, x, y, n, beta=None):
        if beta is None:
            beta = self._beta
        p = float(self.predict_prob(np.reshape(x, (1, x.shape[0])), beta=beta))

        gradient_loss = (-1* (y-p)*x) + (2 * float(self._lambda)/ float(n))*(beta)
        return gradient_loss
        # return np.zeros(x.shape)

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
    # two categories instead of 4: category 2 was ships
    train_y = np.equal(train_y, 2*np.ones(train_y.shape)).astype(int)

    ##### YOUR CODE HERE ######



    lambdas = [0,100]
    alphas = [10**-6, 10**-4, 10**-2, 1, 10 ]


    test_x = np.random.randn(*train_x.shape)
    test_y = np.random.randint(0, 2, size=train_y.shape)

    errorresults = {}
    for l in lambdas:
        for a in alphas:
            print("I am evaluating lambda, alpha", l, a)
            clf = LinearModel(_lambda = l,  _alpha = a)
            clf.SGD_fit(train_x, train_y)
            prediction = clf.predict(train_x)
            error = 1- accuracy(prediction, train_y)
            errorresults [(l,a)]= error

    sorted(errorresults.items(), key=lambda x: x[1])
    print(errorresults)

    # create the classifier
    classifier = LinearModel(_lambda=0, _alpha=1.0)
    #
    # # You'll probably want to change this line for cross-validation
    classifier.SGD_fit(train_x, train_y)
    p = classifier.predict(test_x)
    print("Error on best model for test set", 1-accuracy(test_y, p))

    # for k, v in errorresults.items():


    # This is a dummy test data set.
    # They're all just random noise, not ships or horses or frogs or trucks,
    # so your classifier should and will perform poorly on these.
    # These lines of code are for demonstration purposes only.



    # predictions = classifier.predict(test_x)

    # prints the accuracy of your predictions
    # should be about 0.5 for the random test data above
    # print(accuracy(predictions, test_y))

    # While it's good practice to import at the beginning of a file,
    # since Gradescope won't run this function,
    # you can import anything you want here.
    # matplotlib's pyplot is a good tool for making plots.
    # You can install it here: http://lmgtfy.com/?q=matplotlib+download+python3+anaconda
    # You can read about it here: http://lmgtfy.com/?q=matplotlib+pyplot+tutorial
    from matplotlib import pyplot

# This runs 'main' upon loading the script
if __name__ == '__main__':
    main()
