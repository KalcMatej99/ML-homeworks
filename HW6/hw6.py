import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class ANN:
    def __init__(
        self, units=[], lambda_=0.0001, learning_rate=0.05, number_of_iterations=1000
    ):
        self.X = []
        self.y = []
        self.units = units
        self.lambda_ = lambda_
        self.number_of_layers = len(units) + 2
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.momentum = 0.01
        self.batch_size = 64

    def sigmoid(self, X):
        return 1 / (1 + np.exp(X))

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def get_number_of_outputs(self):
        return 1

    def last_layer_feed_forword_post_process(self, A):
        return A

    def create_Y(self, batch_y):
        return batch_y

    def make_batches(self, X, y):
        i = 0
        batches_X = []
        batches_y = []
        while i < X.shape[0]:
            batches_X.append(X[i:i + self.batch_size, :])
            batches_y.append(y[i:i + self.batch_size])
            i += self.batch_size
        return batches_X, batches_y

    def fit(self, X, y):
        self.X = X
        self.number_of_instances = self.X.shape[0]
        self.X_e = np.hstack((np.ones((self.number_of_instances, 1)), self.X))
        self.y = y
        self.number_of_outputs = self.get_number_of_outputs()
        self.number_of_inputs = self.X_e.shape[1]
        # self.units.insert(0, self.number_of_inputs)

        if len(self.units) > 0:
            self.W = [
                np.random.uniform(size=self.units[0] * self.number_of_inputs).reshape(
                    self.number_of_inputs, self.units[0]
                )
            ]

            for i, unit in enumerate(self.units):
                if i > 0:
                    self.W.append(
                        np.random.uniform(size=unit * (self.units[i - 1] + 1)).reshape(
                            self.units[i - 1] + 1, unit
                        )
                    )

            self.W.append(
                np.random.uniform(
                    size=self.number_of_outputs * (self.units[-1] + 1)
                ).reshape(self.units[-1] + 1, self.number_of_outputs)
            )
        else:
            self.W = [
                np.random.uniform(
                    size=self.number_of_outputs * self.number_of_inputs
                ).reshape(self.number_of_inputs, self.number_of_outputs)
            ]
        self.prevW = self.W

        # Feed forword

        batches_X, batches_y = self.make_batches(self.X, self.y)
        
        for iteration in range(self.number_of_iterations):
            for batch_X, batch_y in zip(batches_X, batches_y):
                A = [batch_X]
                batch_X_e = np.hstack((np.ones((A[-1].shape[0], 1)), A[-1]))
                Z = [batch_X_e * 0]
                A_e = [batch_X_e]

                for l in range(1, self.number_of_layers - 1):
                    Z.append(np.matmul(A_e[-1], self.W[l - 1]))
                    A.append(self.sigmoid(Z[-1]))
                    A_e.append(np.hstack((np.ones((A[-1].shape[0], 1)), A[-1])))

                Z.append(np.matmul(A_e[-1], self.W[-1]))
                A.append(Z[-1])
                A_e.append(np.hstack((np.ones((A[-1].shape[0], 1)), A[-1])))
                A[-1] = self.last_layer_feed_forword_post_process(A[-1])

                # GD last layer
                matrix_Y = self.create_Y(A[-1], batch_y)

                prev_grad_J_A = A[-1] - matrix_Y
                grad_W_last_layer = (
                    np.matmul(A_e[-2].T, prev_grad_J_A) / self.number_of_instances
                )
                self.W[-1] *= (
                    1 - (self.learning_rate * self.lambda_) / self.number_of_instances
                )
                if iteration > 0:
                    currentW = self.W[-1]
                    self.W[-1] -= self.learning_rate * grad_W_last_layer - self.momentum * (self.W[-1] - self.prevW[-1])
                    self.prevW[-1] = currentW
                else:
                    self.W[-1] -= self.learning_rate * grad_W_last_layer

                # GD not last layer
                for l in np.flipud(list(range(1, self.number_of_layers - 1))):
                    prev_grad_J_A = (
                        -1
                        * A[l]
                        * (1 - A[l])
                        * np.matmul(prev_grad_J_A, self.W[l][1:, :].T)
                    )
                    grad_W_layer_l = (
                        np.matmul(A_e[l - 1].T, prev_grad_J_A) / self.number_of_instances
                    )
                    self.W[l - 1] *= (
                        1 - (self.learning_rate * self.lambda_) / self.number_of_instances
                    )
                    if iteration > 0:
                        currentW = self.W[l-1]
                        self.W[l - 1] -= self.learning_rate * grad_W_layer_l - self.momentum * (self.W[l-1] - self.prevW[l-1])
                        self.prevW[l-1] = currentW
                    else:
                        self.W[l - 1] -= self.learning_rate * grad_W_layer_l
        return self

    def predict(self, X):

        number_of_instances = X.shape[0]
        X_e = np.hstack((np.ones((number_of_instances, 1)), X))

        A = [X]
        A_e = [X_e]
        Z = [X_e * 0]

        for l in range(1, self.number_of_layers - 1):
            Z.append(np.matmul(A_e[-1], self.W[l - 1]))
            A.append(self.sigmoid(Z[-1]))
            A_e.append(np.hstack((np.ones((A[-1].shape[0], 1)), A[-1])))

        Z.append(np.matmul(A_e[-1], self.W[-1]))
        A.append(Z[-1])

        return A

    def weights(self):
        return self.W


class ANNClassification(ANN):
    def __init__(self, units=[], lambda_=0):
        self.X = []
        self.y = []
        self.units = units
        self.lambda_ = lambda_

        ANN.__init__(
            self, units, lambda_
        )

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def get_number_of_outputs(self):
        return len(np.unique(self.y))

    def last_layer_feed_forword_post_process(self, A):
        return self.softmax(A.T).T

    def create_Y(self, A, batch_y):
        matrix_Y = A * 0
        y_unique = list(np.unique(self.y))
        integer_y = []
        for y_i in batch_y:
            integer_y.append(y_unique.index(y_i))
        integer_y = np.array(integer_y)
        matrix_Y[np.arange(integer_y.size), integer_y] = 1
        return matrix_Y

    def fit(self, X, y):
        self.X = X
        self.y = y
        ANN.fit(self, X, y)
        return self

    def weights(self):
        return ANN.weights(
            self,
        )

    def predict(self, X):
        A = ANN.predict(self, X)
        return self.softmax(A[-1].T).T


class ANNRegression(ANN):
    def __init__(self, units=[], lambda_=0):
        self.X = []
        self.y = []
        self.units = units
        self.lambda_ = lambda_
        self.number_of_layers = len(units) + 2

        ANN.__init__(
            self, units, lambda_
        )

    def weights(self):
        return ANN.weights(self)

    def get_number_of_outputs(self):
        return 1

    def last_layer_feed_forword_post_process(self, A):
        return A

    def create_Y(self, A, batch_y):
        matrix_Y = np.array([batch_y]).T
        return matrix_Y

    def fit(self, X, y):
        self.X = X
        self.y = y
        ANN.fit(self, X, y)
        return self

    def predict(self, X):
        A = ANN.predict(self, X)
        return A[-1].T[0]


#relative_path = "../input/mlds1hw6dataset"
relative_path = "."
def house2():
    df = pd.read_csv(f"{relative_path}/housing2r.csv")
    X = df[["RM", "AGE", "DIS", "RAD", "TAX"]].to_numpy()
    y = np.transpose(df[["y"]].to_numpy())[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )
    fitter = ANNRegression(units=[13, 6], lambda_=1)
    m = fitter.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    print(rmse)


def house3():
    df = pd.read_csv(f"{relative_path}/housing3.csv")
    X = df[
        [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
        ]
    ].to_numpy()
    y = np.transpose(df[["Class"]].to_numpy())[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )
    fitter = ANNClassification(units=[10], lambda_=2)
    m = fitter.fit(X_train, y_train)
    y_pred_probs = m.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    y_unique = list(np.unique(y_train))
    integer_y = []
    for y_i in y_test:
        integer_y.append(y_unique.index(y_i))
    integer_y = np.array(integer_y)
    ca = accuracy_score(y_pred, integer_y)
    print(ca)


def create_final_predictions():
    number_of_cv_reps = 1
    df = pd.read_csv(f"{relative_path}/train.csv")
    X_train = df.loc[:, df.columns != "target"].to_numpy()
    y_train = np.transpose(df[["target"]].to_numpy())[0]

    bestLamdas = []
    for cv_repitition in range(number_of_cv_reps):
        bestCA = 0
        bestLambda = -1
        kf = KFold(5, shuffle=True)
        for cv_lambda in [0.01, 0.1, 1, 10]:
            err = []
            for train_index, test_index in kf.split(X_train):
                X_train_val, X_test_val = X_train[train_index], X_train[test_index]
                y_train_val, y_test_val = y_train[train_index], y_train[test_index]

                krr_p = ANNClassification(units=[30, 10], lambda_=cv_lambda)
                krr_p.fit(X_train_val, y_train_val)
                y_pred_probs = krr_p.predict(X_test_val)
                y_pred = y_pred_probs.argmax(axis = 1)

                y_unique = list(np.unique(y_train_val))
                integer_y = []
                for y_i in y_test_val:
                    integer_y.append(y_unique.index(y_i))
                integer_y = np.array(integer_y)
                err.extend(y_pred == integer_y)
            ca = np.sum(err)

            if ca > bestCA:
                bestCA = ca
                bestLambda = cv_lambda
        bestLamdas.append(bestLambda)
    bestLambda = np.average(bestLamdas)
    print(bestLamdas)

    bestLambda = 2

    kf = KFold(5, shuffle=True)
    err = []
    for train_index, test_index in kf.split(X_train):
        print(1)
        X_train_val, X_test_val = X_train[train_index], X_train[test_index]
        y_train_val, y_test_val = y_train[train_index], y_train[test_index]

        krr_p = ANNClassification(units=[2, 2], lambda_=bestLambda)
        krr_p.fit(X_train_val, y_train_val)
        y_pred_probs = krr_p.predict(X_test_val)
        y_pred = y_pred_probs.argmax(axis=1)

        y_unique = list(np.unique(y_train_val))
        integer_y = []
        for y_i in y_test_val:
            integer_y.append(y_unique.index(y_i))
        integer_y = np.array(integer_y)
        err.extend(y_pred == integer_y)
    ca = np.average(err)
    print(ca)

    df = pd.read_csv(f"{relative_path}/test.csv")
    X_test = df.loc[:, df.columns != "target"].to_numpy()
    model = ANNClassification(units=[2, 2], lambda_=bestLambda)
    model.fit(X_train, y_train)
    y_pred_probs = model.predict(X_test)

    with open("final.txt", "w") as f:
        for id, pred_probs in zip(X_test[["id"]].values[0], y_pred_probs):
            row = f"{id},"
            for pred_prob in pred_probs:
                row += f"{pred_prob},"
            f.write(row)


if __name__ == "__main__":
    # house2()
    # house3()
    create_final_predictions()
