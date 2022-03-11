from dis import dis
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class MultinomialLogReg:
    def __init__(self):
        self.X = None
        self.y = None
        self.m = 0

    def build(self, X, y):
        self.X = X
        self.X_e = np.concatenate((self.X, [[1]] * len(self.X)), axis=1)
        self.y = y

        self.possible_outcomes = np.unique(y)
        m = len(self.possible_outcomes)
        self.m = m

        self.number_of_columns = len(X[0])

        initial_guess = np.array(
            [np.array([1] * (self.number_of_columns + 1))] * (m - 1)
        )
        B = fmin_l_bfgs_b(self.loglikelihood, initial_guess, approx_grad=True)
        

        B = B[0]
        B = np.concatenate((B, [0] * (self.number_of_columns + 1)), axis=0)

        B = B.reshape((self.m, self.number_of_columns + 1))

        self.B = B

        print(self.B)

        return self

    def predict(self, X):
        prob = []
        for x_i in X:
            x_i_e = np.concatenate((x_i, [1]))
            N = np.sum([np.exp(-np.dot(b, x_i_e)) for b in self.B])
            prob_x_i = [np.exp(-np.dot(b, x_i_e)) / N for b in self.B]
            prob.append(prob_x_i)
        prob = np.array(prob)

        return prob

    def loglikelihood(self, params):

        B = params
        B = np.concatenate((B, [0] * (self.number_of_columns + 1)), axis=0)
        B = B.reshape((self.m, self.number_of_columns + 1))

        """
        loss = 0
        for x_i, y_i in zip(self.X, self.y):
            x_i_e = np.concatenate((x_i, [1]))
            N = np.sum([np.exp(-np.dot(b, x_i_e)) for b in B])
            loss += np.sum([np.log(np.exp(-np.dot(B[j], x_i_e))/N) for j in range(len(B)) if self.possible_outcomes[j] == y_i])
        print("loss", -loss)
        """

        values = -np.dot(self.X_e, np.transpose(B))
        exp_values = np.exp(values)
        Ns = np.sum(exp_values, axis=1)
        #Ns = [[1 / N] * len(B) for N in Ns]
        #log_exp_values = np.log(np.multiply(exp_values, Ns))
        mask = np.transpose(
            [self.y == self.possible_outcomes[j] for j in range(len(B))]
        )
        Ns[Ns == 0] += 0.000001
        log_exp_values = exp_values[mask]/(Ns)
        loss = np.sum(np.log(log_exp_values))
        return -loss


class OrdinalLogReg:
    def __init__(self):
        self.X = None
        self.y = None
        self.m = 0
        self.t = []

    def build(self, X, y):
        self.X = X
        self.y = y

        m = len(np.unique(y))
        self.k = m - 2
        self.m = m

        self.number_of_columns = len(X[0])

        initial_guess = [1] * (self.number_of_columns + 1)
        bounds = [(np.NINF, np.Inf)] * (self.number_of_columns + 1)

        initial_guess += [2] * self.k
        bounds += [(0, np.Inf)] * self.k

        output = fmin_l_bfgs_b(
            self.loglikelihood,
            initial_guess,
            approx_grad=True,
            bounds=bounds
        )
        output = output[0]
        self.B = output[: self.number_of_columns + 1]
        print(self.B)

        discplacments = output[self.number_of_columns + 1 :]
        c = 0
        t_i = []
        for d in discplacments:
            t_i.append(c + d)
            c += d
        self.t = [np.NINF, 0]
        self.t.extend(t_i)
        self.t.extend([np.Inf])
        print(self.t)

        return self

    def cdf_standard_logistic_d(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        prob = []
        for x_i in X:
            x_i_e = np.concatenate((x_i, [1]))
            prob_x_i = []
            u_i = np.dot(self.B, x_i_e)
            for j in range(1, len(self.t)):
                prob_x_i.append(
                    self.cdf_standard_logistic_d(self.t[j] - u_i)
                    - self.cdf_standard_logistic_d(self.t[j - 1] - u_i)
                )
            prob.append(prob_x_i)
        prob = np.array(prob)
        return prob

    def loglikelihood(self, params):

        B = params[: self.number_of_columns + 1]

        discplacments = params[self.number_of_columns + 1 :]
        c = 0
        t_i = []
        for d in discplacments:
            t_i.append(c + d)
            c += d
        t = [np.NINF, 0]
        t.extend(t_i)
        t.extend([np.Inf])

        loss = 0

        possible_y = list(np.unique(self.y))

        for x_i, y_i in zip(self.X, self.y):
            prob_x_i = []
            x_i_e = np.concatenate((x_i, [1]))
            u_i = np.dot(B, x_i_e)
            for j in range(1, len(t)):
                if y_i == possible_y[j - 1]:
                    prob_x_i.append(
                        np.log(
                            self.cdf_standard_logistic_d(t[j] - u_i)
                            - self.cdf_standard_logistic_d(t[j - 1] - u_i)
                        )
                    )
                else:
                    prob_x_i.append(0)
            loss += np.sum(prob_x_i)
        #print("loss", -loss)
        return -loss


def multinomial_bad_ordinal_good(n, rand):
    return [0] * n


MBOG_TRAIN = 100
