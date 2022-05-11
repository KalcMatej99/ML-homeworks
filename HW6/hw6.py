from enum import unique
import numpy as np
import pandas as pd

class ANNClassification:
    def __init__(self, units = [], lambda_ = 0):
        self.X = []
        self.y = []
        self.units = units
        self.lambda_ = lambda_
        self.number_of_layers = len(units) + 2

    def sigmoid(self, X):
        return 1/(1 + np.exp(X))

    def fit(self, X, y):
        self.X = X
        self.number_of_instances = self.X.shape[0]
        ones = np.transpose([np.ones(self.number_of_instances)])
        self.X_e = np.concatenate((X, ones), axis = 1)
        self.y = y
        self.number_of_outputs = len(np.unique(self.y))
        self.number_of_inputs = self.X_e.shape[1]
        #self.units.insert(0, self.number_of_inputs)

        print(self.units)

        if len(self.units) > 0:
            self.W = [np.random.uniform(size=self.units[0] * self.number_of_inputs).reshape(self.number_of_inputs, self.units[0])]

            for i, unit in enumerate(self.units):
                if i > 0:
                    self.W.append(np.random.uniform(size=unit * self.units[i - 1]).reshape(self.units[i - 1], unit))

            self.W.append(np.random.uniform(size=self.number_of_outputs * self.units[-1]).reshape(self.units[-1], self.number_of_outputs))
        else:
            self.W = [np.random.uniform(size=self.number_of_outputs * self.number_of_inputs).reshape(self.number_of_inputs, self.number_of_outputs)]
        

        # Feed forword

        A = [self.X_e]
        Z = [self.X_e * 0]

        for l in range(1, self.number_of_layers):
            Z.append(np.matmul(A[l - 1], self.W[l - 1]))
            A.append(self.sigmoid(Z[l]))

        # GD last layer
        last_layer = self.number_of_layers - 1
        A[last_layer] = np.array([row/np.sum(row) for row in A[last_layer]])
        matrix_Y = A[last_layer] * 0
        matrix_Y[:, self.y] = 1

        D_last_layer = np.multiply(A[last_layer] - matrix_Y, np.multiply(A[last_layer], 1 - A[last_layer]))
        prev_grad_J_A = A[last_layer] - matrix_Y
        grad_W_last_layer = np.matmul(np.transpose(A[last_layer - 1]), D_last_layer)/self.number_of_instances

        self.W[last_layer - 1] -= grad_W_last_layer

        # GD not last layer
        for l in np.flipud(list(range(self.number_of_layers - 1))):
            D = np.multiply(A[l], 1 - A[l])
            dA_Z = np.multiply(A[l + 1], 1 - A[l + 1])
            print(l)
            print(A[l - 1].shape)
            print(A[l].shape)
            print(A)
            print(self.W)
            print(self.W[l].shape)
            print(dA_Z.shape)
            print(prev_grad_J_A.shape)
            print(D.shape)
            D = np.multiply(D, self.W[l] * dA_Z * prev_grad_J_A)
            prev_grad_J_A = D
            grad_W_layer_l = np.matmul(np.transpose(A[l - 1]), D)/self.number_of_instances
            self.W[l - 1] -= grad_W_layer_l


        return self

    def predict(self, X):
        return []
    
    
    
    
class ANNRegression:
    def __init__(self):
        pass