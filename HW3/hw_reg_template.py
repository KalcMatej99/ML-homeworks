from audioop import rms
import unittest
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd

class RidgeReg:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def fit(self, X, y):
        self.X = X
        self.y = y
        X_e = np.concatenate((self.X, [[1]] * len(self.X)), axis=1)

        number_of_columns = len(X_e[0])

        lmbda_ide = self.lmbda * np.identity(number_of_columns)
        lmbda_ide[number_of_columns - 1, number_of_columns - 1] = 0
        self.B = np.dot(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_e), X_e) + lmbda_ide), np.transpose(X_e)), y)

    def predict(self, X):
        X_e = np.concatenate((X, [[1]] * len(X)), axis=1)
        return np.matmul(X_e, self.B)

class LassoReg:
    def __init__(self, lmbda):
        self.lmbda = lmbda
        self.var = 1

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.X_e = np.concatenate((self.X, [[1]] * len(self.X)), axis=1)

        initia_guess = [1] * len(self.X_e[0])
        output = minimize(self.loglikelihood, initia_guess)
        self.B = output["x"]

    def predict(self, X):
        X_e = np.concatenate((X, [[1]] * len(X)), axis=1)
        return np.matmul(X_e, self.B)

    def loglikelihood(self, betas):
        number_of_betas = len(betas)
        return np.sum((np.dot(self.X_e, betas) - self.y) ** 2) + self.lmbda * np.sum(np.abs(betas[:number_of_betas - 1]))

class RegularizationTest(unittest.TestCase):

    def test_ridge_simple(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = RidgeReg(1)
        model.fit(X, y)
        y = model.predict([[10],
                           [20]])
        self.assertAlmostEqual(y[0], 30, delta=0.1)
        self.assertAlmostEqual(y[1], 50, delta=0.1)

    def test_lasso_simple(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = LassoReg(1)
        model.fit(X, y)
        y = model.predict([[10],
                           [20]])
        self.assertAlmostEqual(y[0], 30, delta=0.1)
        self.assertAlmostEqual(y[1], 50, delta=0.1)

    # ... add your tests



def load(fname):
    df = pd.read_csv(fname)
    features = df.columns

    X = df.loc[:, df.columns != 'critical_temp'].to_numpy()
    y = df[["critical_temp"]].to_numpy()

    return features, X[:200,], y[:200,], X[200:,], y[200:,]


def superconductor(X_train, y_train, X_test, y_test):
    def ridge_superconductor(params):
        model = RidgeReg(params[0])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(np.average(np.square(y_pred - y_test)))
        return rmse

    lambdas = []
    rmses = []
    for method in ["Powell", 'Nelder-Mead', "CG", "BFGS", "TNC", "COBYLA", "SLSQP"]:
        print(method)
        output = minimize(ridge_superconductor, [20000], method = method)

        model = RidgeReg(output["x"][0])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.average(np.square(y_pred - y_test)))

        print("Lambda:", output["x"][0])
        print("RMSE:", rmse)

        lambdas.append(output["x"][0])
        rmses.append(rmse)

    index = rmses.index(np.min(rmses))
    print(lambdas[index])
    print(rmses[index])

if __name__ == "__main__":
    features, X_train, y_train, X_test, y_test = load("superconductor.csv")
    superconductor(X_train, y_train, X_test, y_test)
    unittest.main()
