import numpy as np
from cvxopt import solvers, matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.model_selection import KFold

class Polynomial:
    def __init__(self, M=1):
        self.M = M

    def __call__(self, A, B):
        ret = (1 + np.dot(A, B.T)) ** self.M
        return ret


class RBF:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, A, B):
        if len(A.shape) == 1:
            return self.__call__(np.array([A]), B)[0]
        if len(B.shape) == 1:
            return self.__call__(A, np.array([B]))[0]

        left = np.matmul(A, B.T)
        middle = np.transpose(
            np.repeat(
                [np.linalg.norm(np.transpose(A), axis=0) ** 2], B.shape[0], axis=0
            )
        )
        right = np.repeat(
            [np.linalg.norm(np.transpose(B), axis=0) ** 2], A.shape[0], axis=0
        )

        return np.exp((left - 0.5 * middle - 0.5 * right) / self.sigma**2)


class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_=0):
        self.lmba = lambda_
        self.kernel = kernel

    def fit(self, X, y):
        self.X = X
        self.number_of_instances = len(X)
        K = self.kernel(X, X)
        A = np.linalg.inv(K + self.lmba * np.identity(self.number_of_instances))
        self.R = np.dot(A, y)
        return self

    def predict(self, X):
        return np.matmul(self.kernel(X, self.X), self.R)


class SVR:
    def __init__(self, kernel, lambda_=0.0001, epsilon=0.1):
        self.kernel = kernel
        self.lmba = lambda_
        self.eps = epsilon
        self.C = 1 / self.lmba

    def get_alpha(self):
        return np.reshape(self.alpha, (int(len(self.alpha) / 2), 2))

    def get_b(self):
        return self.b

    def is_pos_def(self, x):
        return np.all(np.linalg.eigvals(x) > 0)

    def fit(self, X, y):
        self.X = X
        self.number_of_instances = len(X)
        l = self.number_of_instances

        A = np.array([[1.0, -1.0] * l])
        A_ = matrix([1.0, -1.0] * l, (1, l * 2))
        ones_pos_neg = np.zeros((2 * l, l))
        for i, j in zip(range(0, 2 * l, 2), range(l)):
            ones_pos_neg[i : i + 2, j] = np.array([1, -1])
        P = np.linalg.multi_dot(
            (ones_pos_neg, self.kernel(self.X, self.X), ones_pos_neg.T)
        )
        P_ = matrix(P)

        G = np.zeros((4 * l, 2 * l))
        G[: 2 * l, :] = np.eye(2 * l)
        G[2 * l :, :] = -1 * np.eye(2 * l)

        G_ = matrix(G)

        h = np.zeros(4 * l)
        h[: 2 * l] = np.ones(2 * l) * self.C
        h_ = matrix(h)
        b = matrix([0.0])
        q = matrix([matrix([self.eps - y[i], self.eps + y[i]]) for i in range(l)])

        solvers.options["show_progress"] = False
        sol = solvers.qp(P=P_, q=q, A=A_, b=b, G=G_, h=h_)

        self.alpha = sol["x"]

        alphas = self.get_alpha()
        indexesMax = (
            (alphas[:, 0] < self.C).astype(np.float64)
            * (alphas[:, 1] > 0).astype(np.float64)
        ).astype(bool)

        bmax = np.max(
            -self.eps
            + y[indexesMax]
            - np.dot(
                alphas[:, 0] - alphas[:, 1], self.kernel(self.X, self.X[indexesMax])
            )
        )
        indexesMin = (
            (alphas[:, 0] > 0).astype(np.float64)
            * (alphas[:, 1] < self.C).astype(np.float64)
        ).astype(bool)
        bmin = np.min(
            self.eps
            + y[indexesMin]
            - np.dot(
                alphas[:, 0] - alphas[:, 1], self.kernel(self.X, self.X[indexesMin])
            )
        )

        self.b = np.average([bmax, bmin])
        return self

    def predict(self, X):
        alphas = self.get_alpha()
        return (
            np.transpose(
                np.dot(
                    np.transpose(alphas[:, 0] - alphas[:, 1]),
                    np.transpose(self.kernel(X, self.X)),
                )
            )
            + self.b
        )
        

def sine():
    df = pd.read_csv("sine.csv")

    X = df[["x"]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.transpose(df[["y"]].values)[0]

    X_pred = np.transpose([np.linspace(np.min(X), np.max(X), 1000)])

    figure, axis = plt.subplots(2, 2, figsize=(10,10))

    axis[0,0].scatter(X, y, label="Original")
    krr_p = KernelizedRidgeRegression(kernel=Polynomial(M = 10), lambda_=0.1)
    krr_p.fit(X, y)
    y_kkr = krr_p.predict(X_pred)
    axis[0,0].plot(X_pred, y_kkr, "black", label="KRR - Polynomial")
    axis[0,0].set_title("KRR - Polynomial \n M = 10, lambda = 0.1")


    axis[0,1].scatter(X, y, label="Original")
    krr_p = KernelizedRidgeRegression(kernel=RBF(sigma = 0.2), lambda_=0.5)
    krr_p.fit(X, y)
    y_kkr = krr_p.predict(X_pred)
    axis[0,1].plot(X_pred, y_kkr, "black", label = "KRR - RBF")
    axis[0,1].set_title("KRR - RBF \n sigma = 0.2, lambda = 0.5")

    axis[1,0].scatter(X, y, label="Original")
    krr_p = SVR(kernel=Polynomial(M = 10), lambda_=0.001, epsilon=0.5)
    krr_p.fit(X, y)
    y_kkr = krr_p.predict(X_pred)
    axis[1,0].plot(X_pred, y_kkr, "black", label = "SVR - Polynomial")
    axis[1,0].plot(X_pred, y_kkr + 0.5, "k--")
    axis[1,0].plot(X_pred, y_kkr - 0.5, "k--")
    axis[1,0].set_title("SVR - Polynomial \n M = 10, lambda = 0.001, epsilon = 0.5")

    axis[1,1].scatter(X, y, label="Original")
    krr_p = SVR(kernel=RBF(sigma = 0.5), lambda_=0.4, epsilon=1)
    krr_p.fit(X, y)
    y_kkr = krr_p.predict(X_pred)
    axis[1,1].plot(X_pred, y_kkr, "black", label = "SVR - RBF")
    axis[1,1].plot(X_pred, y_kkr + 1, "k--")
    axis[1,1].plot(X_pred, y_kkr - 1, "k--")
    axis[1,1].set_title("SVR - RBF \n sigma = 0.5, lambda = 0.4, epsilon = 1")
    plt.show()

def house():
    df = pd.read_csv("../input/mlds1hw41/housing2r.csv")
    X = df[["RM","AGE","DIS","RAD","TAX"]].to_numpy()
    y = np.transpose(df[["y"]].to_numpy())[0]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    df_results = pd.DataFrame(columns=["model", "feature_type", "feature_value", "lambda_type", "lambda_value", "rmse"])
    number_of_cv_reps = 200

    for M in range(1, 11):
        bestLamdas = []
        for cv_repitition in range(number_of_cv_reps):
            bestRMSE = 99999999999999999999
            bestLambda = -1
            kf = KFold(5, shuffle=True)
            for cv_lambda in [0, 0.1, 1, 5, 10, 100, 1000]:
                err_squares = []
                for train_index, test_index in kf.split(X_train):
                    X_train_val, X_test_val = X_train[train_index], X_train[test_index]
                    y_train_val, y_test_val = y_train[train_index], y_train[test_index]
                    krr_p = KernelizedRidgeRegression(kernel=Polynomial(M=M), lambda_=cv_lambda)
                    krr_p.fit(X_train_val, y_train_val)
                    err_squares.extend(np.square(krr_p.predict(X_test_val) - y_test_val))
                rmse = np.sqrt(np.average(err_squares))
                if rmse < bestRMSE:
                    bestRMSE = rmse
                    bestLambda = cv_lambda
            bestLamdas.append(bestLambda)
        bestLambda = np.average(bestLamdas)

        krr_p = KernelizedRidgeRegression(kernel=Polynomial(M=M), lambda_=1)
        krr_p.fit(X_train, y_train)
        rmse = np.sqrt(np.average(np.square(krr_p.predict(X_test) - y_test)))
        df_results.loc[len(df_results.index)] = ["krr", "M", M, "normal_lambda", 1, rmse]
        df_results.to_csv("./results.csv")


        print("Best CV lambda", bestLambda)
        krr_p = KernelizedRidgeRegression(kernel=Polynomial(M=M), lambda_=bestLambda)
        krr_p.fit(X_train, y_train)
        rmse = np.sqrt(np.average(np.square(krr_p.predict(X_test) - y_test)))
        df_results.loc[len(df_results.index)] = ["krr", "M", M, "cv_lambda", bestLambda, rmse]
        df_results.to_csv("./results.csv")
    
    for M in range(1, 11):
        print(M)
        try:
            krr_p = SVR(kernel=Polynomial(M=M), lambda_=1, epsilon=0.01)
            krr_p.fit(X_train, y_train)
            rmse = np.sqrt(np.average(np.square(krr_p.predict(X_test) - y_test)))
            df_results.loc[len(df_results.index)] = ["svr", "M", M, "normal_lambda", 1, rmse]
            df_results.to_csv("./results.csv")
        except:
            print("SVR failed at", M)

    
        bestLamdas = []
        for cv_repitition in range(number_of_cv_reps):
            bestRMSE = 99999999999999999999
            bestLambda = -1
            kf = KFold(5, shuffle=True)
            for cv_lambda in [0, 0.1, 1, 5, 10, 100, 1000]:
                err_squares = []
                try:
                    for train_index, test_index in kf.split(X_train):
                        X_train_val, X_test_val = X_train[train_index], X_train[test_index]
                        y_train_val, y_test_val = y_train[train_index], y_train[test_index]

                        krr_p = SVR(kernel=Polynomial(M=M), lambda_=cv_lambda, epsilon=0.01)
                        krr_p.fit(X_train_val, y_train_val)
                        err_squares.extend(np.square(krr_p.predict(X_test_val) - y_test_val))
                    rmse = np.sqrt(np.average(err_squares))

                    if rmse < bestRMSE:
                        bestRMSE = rmse
                        bestLambda = cv_lambda
                except:
                    pass
            bestLamdas.append(bestLambda)
        bestLambda = np.average(bestLamdas)
        print("Best CV lambda", bestLambda)
        try:
            krr_p = SVR(kernel=Polynomial(M=M), lambda_=bestLambda, epsilon=0.01)
            krr_p.fit(X_train, y_train)
            rmse = np.sqrt(np.average(np.square(krr_p.predict(X_test) - y_test)))
            df_results.loc[len(df_results.index)] = ["svr", "M", M, "cv_lambda", bestLambda, rmse]
            df_results.to_csv("./results.csv")
        except:
            print("SVR failed at", M)

    for sigma in [0.1, 0.5, 1, 5, 10, 20, 50, 75, 100]:
    
        krr_p = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=1)
        krr_p.fit(X_train, y_train)
        rmse = np.sqrt(np.average(np.square(krr_p.predict(X_test) - y_test)))
        df_results.loc[len(df_results.index)] = ["krr", "sigma", sigma, "normal_lambda", 1, rmse]
        df_results.to_csv("./results.csv")


        bestLamdas = []
        for cv_repitition in range(number_of_cv_reps):
            bestRMSE = 99999999999999999999
            bestLambda = -1
            kf = KFold(5, shuffle=True)
            for cv_lambda in [0, 0.1, 1, 5, 10, 100, 1000]:
                err_squares = []
                for train_index, test_index in kf.split(X_train):
                    X_train_val, X_test_val = X_train[train_index], X_train[test_index]
                    y_train_val, y_test_val = y_train[train_index], y_train[test_index]
                    krr_p = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=cv_lambda)
                    krr_p.fit(X_train_val, y_train_val)
                    err_squares.extend(np.square(krr_p.predict(X_test_val) - y_test_val))
                rmse = np.sqrt(np.average(err_squares))
                if rmse < bestRMSE:
                    bestRMSE = rmse
                    bestLambda = cv_lambda
            bestLamdas.append(bestLambda)
        bestLambda = np.average(bestLamdas)
        
        krr_p = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=bestLambda)
        krr_p.fit(X_train, y_train)
        rmse = np.sqrt(np.average(np.square(krr_p.predict(X_test) - y_test)))
        df_results.loc[len(df_results.index)] = ["krr", "sigma", sigma, "cv_lambda", bestLambda, rmse]
        df_results.to_csv("./results.csv")

    for sigma in [0.1, 0.5, 1, 5, 10, 20, 50, 75, 100]:
        try:
            krr_p = SVR(kernel=RBF(sigma=sigma), lambda_=1, epsilon=0.01)
            krr_p.fit(X_train, y_train)
            rmse = np.sqrt(np.average(np.square(krr_p.predict(X_test) - y_test)))
            df_results.loc[len(df_results.index)] = ["svr", "sigma", sigma, "normal_lambda", 1, rmse]
            df_results.to_csv("./results.csv")
        except:
            print("SVR failed at", sigma)

        bestLamdas = []
        for cv_repitition in range(number_of_cv_reps):
            bestRMSE = 99999999999999999999
            bestLambda = -1
            kf = KFold(5, shuffle=True)
            for cv_lambda in [0, 0.1, 1, 5, 10, 100, 1000]:
                err_squares = []
                try:
                    for train_index, test_index in kf.split(X_train):
                        X_train_val, X_test_val = X_train[train_index], X_train[test_index]
                        y_train_val, y_test_val = y_train[train_index], y_train[test_index]
                        krr_p = SVR(kernel=RBF(sigma=sigma), lambda_=cv_lambda, epsilon=0.01)
                        krr_p.fit(X_train_val, y_train_val)
                        err_squares.extend(np.square(krr_p.predict(X_test_val) - y_test_val))
                    rmse = np.sqrt(np.average(err_squares))
                    if rmse < bestRMSE:
                        bestRMSE = rmse
                        bestLambda = cv_lambda
                except:
                    pass
            bestLamdas.append(bestLambda)
        bestLambda = np.average(bestLamdas)
        print("Best CV lambda", bestLambda)
        try:
            krr_p = SVR(kernel=RBF(sigma=sigma), lambda_=bestLambda, epsilon=0.01)
            krr_p.fit(X_train, y_train)
            rmse = np.sqrt(np.average(np.square(krr_p.predict(X_test) - y_test)))
            df_results.loc[len(df_results.index)] = ["svr", "sigma", sigma, "cv_lambda", bestLambda, rmse]
            df_results.to_csv("./results.csv")
        except:
            print("SVR failed at", sigma)
            
    df = pd.read_csv("results.csv", index_col="index")
    figure, axis = plt.subplots(2, 2, figsize=(13,13))
    for i, model in enumerate(["krr", "svr"]):
        df_model = df[df["model"] == model]
        for j, feature in enumerate(["M", "sigma"]):
            df_feature = df_model[df_model["feature_type"] == feature]
            for lambdaType in ["lambda = 1", "tuned lambda"]:
                df_rmse = df_feature[df_feature["lambda_type"] == lambdaType]
                #print("Best rmse", np.min(df_rmse["rmse"]))
                axis[i, j].plot(df_rmse["feature_value"].values, df_rmse["rmse"].values, marker = 's', label=lambdaType)

                if lambdaType == "lambda = 1":
                    for label, x, y in zip(df_rmse["rmse"].values, df_rmse["feature_value"].values, df_rmse["rmse"].values):
                        axis[i, j].annotate(
                            np.round(label,1),
                            xy=(x, y), xytext=(0, 10),
                            textcoords='offset points', ha = "center", va='top')
                        
                else:
                    for label, x, y in zip(df_rmse["rmse"].values, df_rmse["feature_value"].values, df_rmse["rmse"].values):
                        axis[i, j].annotate(
                            np.round(label,1),
                            xy=(x, y), xytext=(0, -20),
                            textcoords='offset points', ha = "center", va='bottom')
            
            if model == "krr":
                model2 = "KRR"
            else:
                model2 = "SVR"

            if feature == "M":
                kernel = "Polynomial"
            else:
                kernel = "RBF"
            axis[i, j].set_title(f"{model2} - {kernel}")
            axis[i, j].set_ylabel("RMSE")
            axis[i, j].set_xlabel(feature)

            axis[i, j].legend()
    plt.show()


if __name__ == "__main__":
    #sine()
    #house()