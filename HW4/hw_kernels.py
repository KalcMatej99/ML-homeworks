import numpy as np
from cvxopt import solvers, matrix
import pandas as pd


class Polynomial:
    def __init__(self, M = 1):
        self.M = M

    def __call__(self, A, B):
        return (1 + np.dot(A, B.T)) ** self.M

class RBF:
    def __init__(self, sigma = 1):
        self.sigma = sigma

    def __call__(self, A, B):
        if len(A.shape) > 1 and len(B.shape) > 1:
            return np.array([self.__call__(a, B) for a in A])
        elif len(A.shape) > 1:
            return np.array([self.__call__(a, B) for a in A])
        elif len(B.shape) > 1:
            return np.array([self.__call__(A, b) for b in B])
        return np.exp(-(np.linalg.norm(A - B)**2)/(2 * self.sigma ** 2))

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
        return self

    def predict(self, X):
        return [np.dot(np.transpose([self.kernel(x, x_) for x_ in self.X]), self.R) for x in X]

class SVR:
    def __init__(self, kernel, lambda_=0.0001, epsilon=0.1):
        self.kernel = kernel
        self.lmba = lambda_
        self.eps = epsilon
        self.C = 1/self.lmba

    def get_alpha(self):
        return np.reshape(self.alpha, (int(len(self.alpha)/2), 2))

    def get_b(self):
        return self.b

    def fit(self, X, y):
        self.X = X
        self.number_of_instances = len(X)
        l = self.number_of_instances

        A = matrix([1.0, -1.0] * l, (1, l*2))
        P = matrix([[matrix([[self.kernel(X[row], X[col]), -self.kernel(X[row], X[col])], [-self.kernel(X[row], X[col]), self.kernel(X[row], X[col])]])  for col in range(l)]for row in range(l)])

        Qinside = []
        for i in range(2*l):
            Qinside.append(list(np.concatenate((np.concatenate((np.zeros((1, i)), [[1]]), axis = 1), np.zeros((1, 2*l - i - 1))), axis = 1)[0]))
            Qinside.append(list(np.concatenate((np.concatenate((np.zeros((1, i)), [[-1]]), axis = 1), np.zeros((1, 2*l - i - 1))), axis = 1)[0]))
        Qinside = list(np.transpose(Qinside))
        for i in range(len(Qinside)):
            Qinside[i] = list(Qinside[i])
            
        G = matrix(Qinside)

        hinside = []
        for i in range(2*l):
            hinside.append(self.C)
            hinside.append(0)
        h = matrix(hinside)
        b = matrix([0.0])

        q = matrix([matrix([self.eps - y[i], self.eps + y[i]]) for i in range(l)])


        sol = solvers.qp(P = P, q = q, A = A, b = b, G = G, h = h)

        self.alpha = sol['x']

        vals = [self.alpha[i * 2] - self.alpha[i * 2 + 1] for i in range(int(len(self.alpha)/2))]
        self.w = np.sum([X[i] * vals[i] for i in range(self.number_of_instances)])

        bmax = np.max([self.eps + y[i] - np.sum([(self.alpha[j * 2] - self.alpha[j * 2 + 1]) * self.kernel(self.X[j], self.X[i]) for j in range(l)]) for i in range(int(len(self.alpha)/2)) if self.alpha[i * 2] < self.C or self.alpha[i * 2 + 1] > 0])
        bmin = np.max([self.eps + y[i] - np.sum([(self.alpha[j * 2] - self.alpha[j * 2 + 1]) * self.kernel(self.X[j], self.X[i]) for j in range(l)]) for i in range(int(len(self.alpha)/2)) if self.alpha[i * 2] > 0 or self.alpha[i * 2 + 1] < self.C])
        self.b = np.average([bmax, bmin])
        return self

    def predict(self, X):
        alphas = self.get_alpha()
        return [np.sum([(alphas[i, 0] - alphas[i, 1]) * self.kernel(x, self.X[i]) for i in range(self.number_of_instances)]) + self.b for x in X]

