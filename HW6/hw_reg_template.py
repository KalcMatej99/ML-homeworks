from audioop import rms
import unittest
import numpy as np
from scipy.optimize import minimize
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 - 2*X[:,0]
        model = RidgeReg(1)
        model.fit(X, y)
        y = model.predict([[10],
                           [20]])
        self.assertAlmostEqual(y[0], -10, delta=0.1)
        self.assertAlmostEqual(y[1], -30, delta=0.1)

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
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 - 2*X[:,0]
        model = LassoReg(1)
        model.fit(X, y)
        y = model.predict([[10],
                           [20]])
        self.assertAlmostEqual(y[0], -10, delta=0.1)
        self.assertAlmostEqual(y[1], -30, delta=0.1)

    def test_no_lambda(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = RidgeReg(0)
        model.fit(X, y)
        y1 = model.predict([[10],
                           [20]])
        
        model = LassoReg(0)
        model.fit(X, y)
        y2 = model.predict([[10],
                           [20]])
        self.assertAlmostEqual(y1[0], y2[0], delta=0.1)
        self.assertAlmostEqual(y1[1], y2[1], delta=0.1)

    def test_lambda(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = RidgeReg(1)
        model.fit(X, y)
        y1 = model.predict([[10],
                           [20]])
        model = RidgeReg(100)
        model.fit(X, y)
        y2 = model.predict([[10],
                           [20]])
        self.assertNotEqual(y1[0], y2[0])
        self.assertNotEqual(y1[1], y2[1])
        
        model = LassoReg(1)
        model.fit(X, y)
        y1 = model.predict([[10],
                           [20]])
        model = LassoReg(100)
        model.fit(X, y)
        y2 = model.predict([[10],
                           [20]])
        self.assertNotEqual(y1[0], y2[0])
        self.assertNotEqual(y1[1], y2[1])

    def test_diff_input(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = RidgeReg(1)
        model.fit(X, y)
        y1 = model.predict([[10],
                           [20]])
        X = np.array([[2],
                      [20],
                      [200]])
        y = 10 + 2*X[:,0]
        model = RidgeReg(1)
        model.fit(X, y)
        y2 = model.predict([[10],
                           [20]])
        self.assertNotEqual(y1[0], y2[0])
        self.assertNotEqual(y1[1], y2[1])
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = LassoReg(1)
        model.fit(X, y)
        y1 = model.predict([[10],
                           [20]])
        X = np.array([[2],
                      [20],
                      [200]])
        y = 10 + 2*X[:,0]
        model = LassoReg(1)
        model.fit(X, y)
        y2 = model.predict([[10],
                           [20]])
        self.assertNotEqual(y1[0], y2[0])
        self.assertNotEqual(y1[1], y2[1])
        



def load(fname):
    df = pd.read_csv(fname)
    features = df.columns

    X = df.loc[:, df.columns != 'critical_temp'].to_numpy()
    y = df[["critical_temp"]].to_numpy()

    def normalize2(X_train, X_test):
        for column in range(len(X_train[0])):
            u = np.mean(X_train[:, column])
            std = np.std(X_train[:, column])
            X_train[:, column] = (X_train[:, column] - u) / std
            X_test[:, column] = (X_test[:, column] - u) / std
        return X_train, X_test

    k = 200
    X_train = X[:k,]
    X_test = X[k:,]

    X_train, X_test = normalize2(X_train, X_test)

    return features, X_train, y[:k,], X_test, y[k:,]



def CV(X_train, y_train, X_test, y_test):
    def ridge_superconductor(params):
        
        number_of_folds = 10#int(params[1])
        number_of_elem = len(X_train)
        number_of_elem_in_fold = int(number_of_elem/number_of_folds)
        model = RidgeReg(params[0])

        folds = [X_train[i * number_of_elem_in_fold: i * number_of_elem_in_fold + number_of_elem_in_fold, :] for i in range(number_of_folds)]
        folds_y = [y_train[i * number_of_elem_in_fold: i * number_of_elem_in_fold + number_of_elem_in_fold] for i in range(number_of_folds)]
        all = set(range(number_of_elem))
        tfolds = [X_train[list(all - set(range(i * number_of_elem_in_fold, i * number_of_elem_in_fold + number_of_elem_in_fold))), :] for i in range(number_of_folds)]
        tfolds_y = [y_train[list(all - set(range(i * number_of_elem_in_fold, i * number_of_elem_in_fold + number_of_elem_in_fold)))] for i in range(number_of_folds)]

        errors = []
        for i in range(number_of_folds):
            model.fit(tfolds[i], tfolds_y[i])
            y_pred = model.predict(folds[i])
            error = np.transpose(y_pred - folds_y[i])[0]
            errors.extend(error)
            
        rmse = np.sqrt(np.average(np.square(errors)))
        return rmse
    
    print("Start CV")
    output = minimize(ridge_superconductor, [100], method = "Powell")

    model = RidgeReg(output["x"][0])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmse = np.sqrt(np.average(np.square(y_pred - y_train)))

    #print("CV:", output["x"][1])
    print("Lambda:", output["x"][0])
    print("RMSE on train:", rmse)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.average(np.square(y_pred - y_test)))
    print("RMSE on test:", rmse)

def superconductor(X_train, y_train, X_test, y_test):
    CV(X_train, y_train, X_test, y_test)
    def ridge_superconductor(params):
        model = RidgeReg(params[0])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(np.average(np.square(y_pred - y_test)))
        return rmse


    output = minimize(ridge_superconductor, [1], method = "Powell")

    model = RidgeReg(output["x"][0])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmse = np.sqrt(np.average(np.square(y_pred - y_train)))

    print("Ideal Lambda on test set:", output["x"][0])
    print("RMSE on train set with idela Lambda:", rmse)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.average(np.square(y_pred - y_test)))
    print("RMSE on test set with idela Lambda:", rmse)

if __name__ == "__main__":
    features, X_train, y_train, X_test, y_test = load("superconductor.csv")
    superconductor(X_train, y_train, X_test, y_test)
    unittest.main()
