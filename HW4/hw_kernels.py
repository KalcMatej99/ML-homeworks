import numpy as np
from cvxopt import solvers, matrix



class Polynomial:
    def __init__(self, M = 1):
        self.M = M

    def __call__(self, A, B):
        return (1 + np.dot(A, B)) ** self.M

class RBF:
    def __init__(self, sigma = 1):
        self.sigma = sigma

    def __call__(self, A, B):
        return np.exp(-(np.norm(A - B)**2)/(2 * self.sigma ** 2))

class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_ = 0):
        self.lmba = lambda_
        self.kernel = kernel

    def fit(self, X, y):
        self.X = X
        self.number_of_instances = len(X)
        K = [[self.kernel(x, x_) for x_ in X] for x in X]
        A = np.linalg.inv(K + self.lmba * np.identity(self.number_of_instances))
        self.R = np.dot(A, y)

    def predict(self, X):
        return [np.dot(np.transpose([self.kernel(x, x_) for x_ in self.X]), self.R) for x in X]

class SVR:
    def __init__(self, kernel, lambda_=0.0001, epsilon=0.1):
        self.kernel = kernel
        self.lmba = lambda_
        self.eps = epsilon
        self.C = 1/self.lmba

    def fit(self, X, y):
        self.number_of_instances = len(X)
        l = self.number_of_instances
        A = matrix([1, -1] * l)

        P = matrix([[matrix([[np.dot(X[row], X[col]), -np.dot(X[row], X[col])], [-np.dot(X[row], X[col]), np.dot(X[row], X[col])]])  for col in range(l)]for row in range(l)])

        q = matrix([matrix([self.eps - y[i], self.eps + y[i]]) for i in range(l)])
        sol = solvers.qp(P = P, q = q, A = A, b = 0)

        a = sol['x']

        vals = [a[i * 2] - a[i * 2 + 1] for i in range(len(a)/2)]
        self.w = np.sum([X[i] * vals[i] for i in range(self.number_of_instances)])

    def predict(self, X):
        pass

