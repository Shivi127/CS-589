'''
We need two things at the minimum.
Fit and Predict
'''


import numpy as np
class RidgeRegression(object):
    def __init__(self, lmbda=0.1):
        self.lmbda = lmbda
        self.kernel = None

    def fit(self, X, y):

        # Number of training samples
        self.N = X.shape[0]
        self.X_training = np.copy(X)
        print(X.shape,"Shape")

        C = np.zeros((self.N,self.N))
        # kernelize
        for i in range(self.N):
            for j in range(self.N):
                C[i][j] = self.kernel(X[i],X[j])

        C += self.lmbda*np.eye(X.shape[0])
        self.w = np.linalg.inv(C).dot(y)

    def predict(self, X):
        # Number of test samples
        n = X.shape[0]
        predictions = np.zeros((n,))

        for i in range(n):
            for j in range(self.N):
                predictions[i] += (self.w[j]* self.kernel(X[i],self.X_training[j]))
        return predictions

    def get_params(self, deep=True):
        return {"lmbda": self.lmbda}

    def set_params(self, lmbda=0.1):
        self.lmbda = lmbda
        return self