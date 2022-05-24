import enum
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from hw_reg_template import RidgeReg
from solution import MultinomialLogReg
import datetime

np.random.seed(1)


class ANN:
    def __init__(
        self,
        regressionProblem=True,
        units=[],
        lambda_=0.0001,
        learning_rate=0.1,
        number_of_iterations=10000,
        enable_early_stopping=False,
        check_gradient = True
    ):
        self.regressionProblem = regressionProblem
        self.X = []
        self.y = []
        self.units = units
        self.lambda_ = lambda_
        self.number_of_layers = len(units) + 2
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.momentum = 0.0
        self.batch_size = 1000
        self.gd_method = "GD"
        self.enable_early_stopping = enable_early_stopping
        self.validation_set_split = 0.1
        if self.regressionProblem:
            self.tolerance_validation = 10
        else:
            self.tolerance_validation = 10
        self.min_number_of_iterations_before_early_stopping = 40
        self.activation_function = self.sigmoid
        self.check_gradient = check_gradient

    def sigmoid(self, X, deriv=False):
        if deriv:
            return self.sigmoid(X) * (1 - self.sigmoid(X))
        else:
            return 1 / (1 + np.exp(-X))

    def perceptron(self, X, deriv=False):
        if deriv:
            return X * 0.0
        else:
            X[X == 0] = 1
            return X/np.abs(X)

    def ReLU(self, X, deriv=False):
        if deriv:
            return abs(np.maximum(X, 0)/X)
        else:
            return np.maximum(X, 0)

    def tanh(self, X, deriv=False):
        if deriv:
            return 1 - self.tanh(X) ** 2
        else:
            return np.tanh(X)

    def softplus(self, X, deriv=False):
        if deriv:
            return 1/(1 + np.exp(-X))
        else:
            return self.sigmoid(X)

    def softmax(self, Z, deriv=False):
        if deriv:
            eye = np.eye(len(Z)) * sum(np.exp(Z))
            for i, z_i in enumerate(np.exp(Z)):
                for j, z_j in enumerate(np.exp(Z)):
                    eye[i, j] -= z_i * z_j
                eye[i, :] /= z_i ** 2
            return eye
        else:
            A = np.exp(Z) / sum(np.exp(Z))
            A = np.nan_to_num(A, copy=True, nan=0.0)
            return A

    def get_number_of_outputs(self):
        return 1

    def last_layer_feed_forword_post_process(self, A):
        return A

    def create_Y(self, batch_y):
        return batch_y

    def evaluate_validation_results(self, results):
        return 0.5

    def make_batches(self, X, y):
        i = 0
        batches_X = []
        batches_y = []
        while i < X.shape[0]:
            batches_X.append(X[i: i + self.batch_size, :])
            batches_y.append(y[i: i + self.batch_size])
            i += self.batch_size
        return batches_X, batches_y

    def feed_forword(self, batch_X):
        A = [batch_X]
        batch_X_e = np.hstack((np.ones((A[-1].shape[0], 1)), A[-1]))
        Z = [batch_X_e * 0]
        A_e = [batch_X_e]

        for l in range(1, self.number_of_layers):
            Z.append(np.matmul(A_e[-1], self.W[l - 1]))

            if l == self.number_of_layers - 1:
                A.append(Z[-1])
            else:
                A.append(self.activation_function(Z[-1]))
            A_e.append(np.hstack((np.ones((A[-1].shape[0], 1)), A[-1])))

        A[-1] = self.last_layer_feed_forword_post_process(A[-1])

        return A, A_e, Z

    def mse(self, y_, y):
        return np.sum((np.squeeze(y_) - np.squeeze(y)) ** 2) / (2 * len(y))

    def cross_entropy(self, probs, y):
        integer_y = np.array([self.y_unique.index(y_i) for y_i in y])
        choosen_probs = probs[np.arange(y.size), integer_y]
        return -np.sum(np.log(choosen_probs)) / y.size

    def numerical_gradients(self, X, y, h=0.000001):

        output_layer = self.predict(X)

        if self.regressionProblem:
            current_cost = self.mse(output_layer, y)
        else:
            current_cost = self.cross_entropy(output_layer, y)
        numerical_W_list = []
        for layer in range(len(self.W) - 1, -1, -1):
            numerical_W = self.W[layer] * 0
            for r in range(numerical_W.shape[0]):
                for c in range(numerical_W.shape[1]):
                    self.W[layer][r, c] += h
                    output_with_h = self.predict(X)

                    if self.regressionProblem:
                        numerical_W[r, c] = (self.mse(output_with_h, y) - current_cost)/h
                    else:
                        numerical_W[r, c] = (self.cross_entropy(
                            output_with_h, y) - current_cost)/h
                    self.W[layer][r, c] -= h
            numerical_W_list.append(numerical_W)
            
        return numerical_W_list

    def fit(self, X, y):
        self.X = X
        self.y = y

        if self.enable_early_stopping:
            self.X, self.validation_X, self.y, self.validation_y = train_test_split(
                self.X, self.y, test_size=self.validation_set_split
            )
        if not self.regressionProblem:
            self.y_unique = list(np.unique(self.y))
        self.number_of_instances = self.X.shape[0]
        self.X_e = np.hstack((np.ones((self.number_of_instances, 1)), self.X))
        self.number_of_outputs = self.get_number_of_outputs()
        self.number_of_inputs = self.X_e.shape[1]

        if len(self.units) > 0:
            self.W = [np.random.uniform(
                size=(self.number_of_inputs, self.units[0]))]

            for i, unit in enumerate(self.units):
                if i > 0:
                    self.W.append(np.random.uniform(
                        size=(self.units[i - 1] + 1, unit)))

            self.W.append(
                np.random.uniform(
                    size=(self.units[-1] + 1, self.number_of_outputs))
            )
        else:
            self.W = [
                np.random.uniform(
                    size=(self.number_of_inputs, self.number_of_outputs))
            ]
        self.prevW = self.W

        batches_X, batches_y = self.make_batches(self.X, self.y)
        best_metric = 999999999
        iterations_with_no_improvment = 0

        for iteration in range(self.number_of_iterations):
            batch_X = self.X
            batch_y = self.y
            #for batch_X, batch_y in zip(batches_X, batches_y):
            self.number_of_instances = batch_X.shape[0]

            if iteration == 0 and self.check_gradient:
                numerical_gradients = self.numerical_gradients(batch_X, batch_y)

            list_gradients = []

            # Feed forword
            A, A_e, Z = self.feed_forword(batch_X)

            correctionsW = []

            # GD last layer
            prev_grad_J_A = A[-1] - self.create_Y(A[-1], batch_y)
            grad_W_last_layer = (
                np.matmul(A_e[-2].T, prev_grad_J_A) /
                self.number_of_instances
            )
            correctionsW.insert(0, self.learning_rate * grad_W_last_layer)
            list_gradients.append(grad_W_last_layer)

            # GD not last layer
            for l in np.flipud(list(range(1, self.number_of_layers - 1))):
                prev_grad_J_A = np.hstack((np.ones((Z[l].shape[0], 1)), self.activation_function(Z[l], deriv = True))) * np.matmul(prev_grad_J_A, self.W[l].T)
                grad_W_layer_l = (
                    np.matmul(A_e[l - 1].T, prev_grad_J_A)
                    / self.number_of_instances
                )
                grad_W_layer_l = grad_W_layer_l[:, 1:]
                prev_grad_J_A = prev_grad_J_A[:, 1:]
                correctionsW.insert(0, self.learning_rate * grad_W_layer_l)
                list_gradients.append(grad_W_layer_l)
            
            for layer in range(len(self.W) - 1, -1, -1):
                self.W[layer][1:, :] *= (
                    1 - (self.learning_rate * self.lambda_) /
                    self.number_of_instances
                )
                self.W[layer] -= correctionsW[layer]

            if iteration == 0 and self.check_gradient:
                for numerical_gradient_layer_i, gradient_layer_i in zip(numerical_gradients, list_gradients):
                    np.testing.assert_allclose(
                        numerical_gradient_layer_i, gradient_layer_i, atol=0.001)

            if self.enable_early_stopping:
                y_validation_pred_probs = self.predict(self.validation_X)

                if self.regressionProblem:
                    metric = self.mse(y_validation_pred_probs, self.validation_y)
                else:
                    metric = self.cross_entropy(y_validation_pred_probs, self.validation_y)
                iterations_with_no_improvment += 1

                if metric < best_metric:
                    best_metric = metric
                    iterations_with_no_improvment = 0
                    best_weights = self.W

                if (
                    iterations_with_no_improvment >= self.tolerance_validation
                    and iteration > self.min_number_of_iterations_before_early_stopping
                ):
                    self.W = best_weights
                    print("Early stop at iteration", iteration)
                    return self

        return self

    def predict(self, X):
        A, A_e, Z = self.feed_forword(X)
        return A[-1]

    def weights(self):
        return self.W


class ANNClassification(ANN):
    def __init__(self, units=[], lambda_=0, early_stopping=False, check_gradient = True):
        ANN.__init__(
            self,
            regressionProblem=False,
            units=units,
            lambda_=lambda_,
            enable_early_stopping=early_stopping,
            check_gradient = check_gradient
        )

    def get_number_of_outputs(self):
        return len(self.y_unique)

    def last_layer_feed_forword_post_process(self, A):
        return self.softmax(A.T).T
      
    def create_Y(self, A, batch_y):
        matrix_Y = A * 0
        integer_y = []
        for y_i in batch_y:
            integer_y.append(self.y_unique.index(y_i))
        integer_y = np.array(integer_y)
        matrix_Y[np.arange(integer_y.size), integer_y] = 1
        return matrix_Y

    def fit(self, X, y):
        ANN.fit(self, X, y)
        return self

    def weights(self):
        return ANN.weights(
            self,
        )

    def predict(self, X):
        A = ANN.predict(self, X)
        return A


class ANNRegression(ANN):
    def __init__(self, units=[], lambda_=0, early_stopping=False, check_gradient = True):
        ANN.__init__(
            self,
            regressionProblem=True,
            units=units,
            lambda_=lambda_,
            enable_early_stopping=early_stopping,
            check_gradient = check_gradient
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
        return A.T[0]


# relative_path = "../input/mlds1hw6dataset"
relative_path = "."


def normalize2(X_train, X_test):
    for column in range(X_train.shape[1]):
        u = np.mean(X_train[:, column])
        std = np.std(X_train[:, column])
        X_train[:, column] = (X_train[:, column] - u) / std
        X_test[:, column] = (X_test[:, column] - u) / std
    return X_train, X_test


def house2():
    print("housing2r")
    df = pd.read_csv(f"{relative_path}/housing2r.csv")
    X = df[["RM", "AGE", "DIS", "RAD", "TAX"]].to_numpy()
    y = np.transpose(df[["y"]].to_numpy())[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    X_train, X_test = normalize2(X_train, X_test)

    bestLamdas = []
    number_of_cv_reps = 10
    for cv_repitition in range(number_of_cv_reps):
        bestRMSE = 99999999999999999999
        bestLambda = -1
        kf = KFold(5, shuffle=True)
        for cv_lambda in [0, 0.1, 1, 5, 10, 100, 1000]:
            err_squares = []
            for train_index, test_index in kf.split(X_train):
                X_train_val, X_test_val = X_train[train_index], X_train[test_index]
                y_train_val, y_test_val = y_train[train_index], y_train[test_index]

                krr_p = RidgeReg(cv_lambda)
                krr_p.fit(X_train_val, y_train_val)
                err_squares.extend(
                    np.square(krr_p.predict(X_test_val) - y_test_val))
            rmse = np.sqrt(np.average(err_squares))

            if rmse < bestRMSE:
                bestRMSE = rmse
                bestLambda = cv_lambda
        bestLamdas.append(bestLambda)
    bestLambda = np.average(bestLamdas)

    model = RidgeReg(bestLambda)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    print("Ridge", rmse)

    bestRMSE = 99999999999999999999
    bestLambda = -1
    kf = KFold(5, shuffle=True)
    for cv_lambda in [0.001, 0.1, 1, 10, 100]:
        err_squares = []
        for train_index, test_index in kf.split(X_train):
            X_train_val, X_test_val = X_train[train_index], X_train[test_index]
            y_train_val, y_test_val = y_train[train_index], y_train[test_index]

            krr_p = ANNRegression(
                units=[13, 6], lambda_=cv_lambda, early_stopping=False)
            krr_p.fit(X_train_val, y_train_val)
            err_squares.extend(
                np.square(krr_p.predict(X_test_val) - y_test_val))
        rmse = np.sqrt(np.average(err_squares))

        if rmse < bestRMSE:
            bestRMSE = rmse
            bestLambda = cv_lambda
    bestLamdas.append(bestLambda)

    fitter = ANNRegression(
        units=[13, 6], lambda_=bestLambda, early_stopping=False)
    m = fitter.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    print("NN", rmse)


def house3():
    print("housing3")
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

    X_train, X_test = normalize2(X_train, X_test)

    model = MultinomialLogReg()
    model.build(X_train, y_train)
    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    y_unique = list(np.unique(y_train))
    integer_y = np.array([y_unique.index(y_i) for y_i in y_test])
    ca = np.average(y_pred == integer_y)

    print("Multinomial Logistic Regression", ca)

    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.1
    )

    bestLoss = 999999999999
    bestLambda = -1
    for cv_lambda in [0.01, 0.1, 1, 10]:
        model = ANNClassification(
            units=[10], lambda_=cv_lambda, early_stopping=False)
        model.fit(X_train_val, y_train_val)
        y_pred_probs = model.predict(X_val)
        loss = model.cross_entropy(y_pred_probs, y_val)
        print("loss", loss)
        if loss < bestLoss:
            bestLoss = loss
            bestLambda = cv_lambda
    print(bestLambda)

    model = ANNClassification(
        units=[10], lambda_=bestLambda, early_stopping=False)
    model.fit(X_train, y_train)
    y_pred_probs = model.predict(X_test)
    print("loss", model.cross_entropy(y_pred_probs, y_test))
    y_pred = y_pred_probs.argmax(axis=1)

    y_unique = list(np.unique(y_train))
    integer_y = np.array([y_unique.index(y_i) for y_i in y_test])
    ca = np.average(y_pred == integer_y)
    print("NN", ca)


def create_final_predictions():
    df = pd.read_csv(f"{relative_path}/train.csv")
    X_train = df.loc[:, df.columns != "target"].to_numpy()[:, 1:]
    y_train = np.transpose(df[["target"]].to_numpy())[0]

    df = pd.read_csv(f"{relative_path}/test.csv")
    X_test = df.loc[:, df.columns != "target"].to_numpy()[:, 1:]
    test_ids = df.loc[:, df.columns != "target"].to_numpy()[:, 0]

    X_train, X_test = normalize2(X_train, X_test)

    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2
    )

    bestLoss = 9999999999999
    bestUnits = []
    for layer1 in [0, 10, 50]:
        for layer2 in [0, 10, 50]:
            for layer3 in [0, 10, 50]:
                
                units = []
                
                if layer1 != 0:
                    units.append(layer1)
                if layer2 != 0:
                    units.append(layer2)
                if layer3 != 0:
                    units.append(layer3)

                model = ANNClassification(
                    units=units, lambda_=0.001, early_stopping=True, check_gradient=False
                )
                model.fit(X_train_val, y_train_val)
                y_pred_probs = model.predict(X_val)
                loss = model.cross_entropy(y_pred_probs, y_val)
                print("units", units, "loss", loss)

                if loss < bestLoss:
                    bestLoss = loss
                    bestUnits = units
    print(bestUnits)

    model = ANNClassification(
        units=bestUnits, lambda_=0.001, early_stopping=True, check_gradient=False)
    start = datetime.datetime.now().replace(microsecond=0)
    model.fit(X_train, y_train)
    finish = datetime.datetime.now().replace(microsecond=0)
    elapsed_time = finish - start
    print("Elapsed time", elapsed_time)
    y_pred_probs = model.predict(X_test)

    print(y_pred_probs)

    with open("final.txt", "w") as f:
        f.write(
            "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\r\n"
        )
        for id, pred_probs in zip(test_ids, y_pred_probs):
            row = f"{id},"
            for i, pred_prob in enumerate(pred_probs):
                if i == pred_probs.shape[0] - 1:
                    row += f"{pred_prob}\r\n"
                else:
                    row += f"{pred_prob},"
            f.write(row)


if __name__ == "__main__":

    house2()
    house3()
    create_final_predictions()
