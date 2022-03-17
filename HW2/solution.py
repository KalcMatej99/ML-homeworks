import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from sklearn.model_selection import train_test_split
import random


class MultinomialLogReg:
    def __init__(self):
        self.X = None
        self.y = None
        self.m = 0
        self.best_log_score = 999999999999999999999

    def build(self, X, y):
        self.X = X
        self.X_e = np.concatenate((self.X, [[1]] * len(self.X)), axis=1)
        self.y = y

        m = len(np.unique(y))
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

        return self

    def predict(self, X):
        prob = []
        for x_i in X:
            x_i_e = np.concatenate((x_i, [1]))
            N = np.sum([np.exp(np.dot(b, x_i_e)) for b in self.B])
            prob_x_i = [np.exp(np.dot(b, x_i_e)) / N for b in self.B]
            prob.append(prob_x_i)
        prob = np.array(prob)

        return prob

    def loglikelihood(self, params):

        B = params
        B = np.concatenate((B, [0] * (self.number_of_columns + 1)), axis=0)
        B = B.reshape((self.m, self.number_of_columns + 1))

        values = np.dot(self.X_e, np.transpose(B))
        exp_values = np.exp(values)
        Ns = np.sum(exp_values, axis=1)
        mask = np.transpose([self.y == j for j in range(len(B))])
        
        log_exp_values = exp_values[mask] / (Ns)
        loss = np.sum(np.log(log_exp_values))

        if -loss < self.best_log_score:
            self.best_log_score = -loss
        return -loss


class OrdinalLogReg:
    def __init__(self):
        self.X = None
        self.y = None
        self.m = 0
        self.t = []
        self.best_log_score = 99999999999999999

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
            self.loglikelihood, initial_guess, approx_grad=True, bounds=bounds
        )
        output = output[0]
        self.B = output[: self.number_of_columns + 1]

        discplacments = output[self.number_of_columns + 1 :]
        c = 0
        t_i = []
        for d in discplacments:
            t_i.append(c + d)
            c += d
        self.t = [np.NINF, 0]
        self.t.extend(t_i)
        self.t.extend([np.Inf])

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

        for x_i, y_i in zip(self.X, self.y):
            x_i_e = np.concatenate((x_i, [1]))
            u_i = np.dot(B, x_i_e)
            j = y_i + 1
            loss += np.log(
                self.cdf_standard_logistic_d(t[j] - u_i)
                - self.cdf_standard_logistic_d(t[j - 1] - u_i)
            )

        if -loss < self.best_log_score:
            self.best_log_score = -loss
        return -loss


def multinomial_bad_ordinal_good(n, rand):
    X = np.array([[rand.uniform(0, 1) + i] for i in range(n)])
    y = np.array(
        [
            int(x[0] <= n / 3)
            + 2 * int(x[0] <= (2 * n) / 3 and x[0] > n / 3)
            + 3 * int(x[0] > (2 * n) / 3)
            for x in X
        ]
    )
    y -= 1
    return X, y


MBOG_TRAIN = 50

if __name__ == "__main__":
    rand = random.Random(0)
    
    print("Part 2")
    df = pd.read_csv("dataset.csv", delimiter=";")
    df["ShotType"] = df["ShotType"].map({'above head':0,
                             'dunk':1,
                             'hook shot':2,
                             'layup':3,
                             'other':5,
                             'tip-in':4},
                             na_action=None)
    X = df.loc[:, df.columns != 'ShotType']
    y = np.array([v[0] for v in df[["ShotType"]].to_numpy()])
    
    X = pd.get_dummies(X, columns = ['Competition', 'PlayerType', 'Movement'], drop_first=True)


    def normalize(X_train, column):
        u = np.mean(X_train[column])
        std = np.std(X_train[column])
        X_train[column] = (X_train[column] - u) / std
        return X_train

    X = normalize(X, "Angle")
    X = normalize(X, "Distance")

    columns = X.columns

    X = X.to_numpy()
    number_of_samples = len(X)

    df_B = []

    oob_CAs = []
    for i in range(10):
        seq = list(range(number_of_samples))
        samples_to_use_in_split = []
        out_of_bag_samples = list(range(len(X)))
        while len(samples_to_use_in_split) < number_of_samples:
            newSample = rand.choice(seq)
            samples_to_use_in_split.append(newSample)
            if newSample in out_of_bag_samples:
                out_of_bag_samples.remove(newSample)
        bootstraped_X = X[samples_to_use_in_split, :]
        bootstraped_y = y[samples_to_use_in_split]
        oob_X = X[out_of_bag_samples, :]
        oob_y = y[out_of_bag_samples]

        lr = MultinomialLogReg()
        lr.build(bootstraped_X, bootstraped_y)

        oob_pred = lr.predict(oob_X)
        oof_CA = np.average(oob_pred.argmax(axis=1) == oob_y)
        print("CA out of bag:", oof_CA)
        oob_CAs.append(oof_CA)

        df_B_i = pd.DataFrame(np.concatenate((lr.B, [[p] for p in range(6)]), axis = 1), columns = np.concatenate((columns, ["B_0", "possible_outcomes"])))
        df_B_i = df_B_i.set_index("possible_outcomes")
        df_B.append(df_B_i)
    
    boo_df_B = df_B
    print("Average oob CA", np.mean(oob_CAs), "+/-", np.std(oob_CAs))
    tab = []
    for p in range(6):
        vs = []
        for df in boo_df_B:
            vs.append(df.loc[[p]].values[0])
        vs = np.transpose(np.array(vs))
        mean_vs = np.array([np.mean(v.astype(np.float64)) for v in vs])
        se_vs = np.array([np.std(v.astype(np.float64)) for v in vs])

        row = []
        for m, s in zip(mean_vs, se_vs):
            row.append(f"{round(m, 2)}")
        row.append(p)
        tab.append(row)

    df_B = pd.DataFrame(tab, columns = np.concatenate((columns, ["B_0", "ShotType"]))).round(2)
    df_B["ShotType"] = df_B["ShotType"].map({0:'above head',
                             1:'dunk',
                             2:'hook shot',
                             3:'layup',
                             5:'other',
                             4:'tip-in'},
                             na_action=None)
    df_B.to_csv("part_2_betas.csv")
    tab = []
    for p in range(6):
        vs = []
        for df in boo_df_B:
            vs.append(df.loc[[p]].values[0])
        vs = np.transpose(np.array(vs))
        mean_vs = np.array([np.mean(v.astype(np.float64)) for v in vs])
        se_vs = np.array([np.std(v.astype(np.float64)) for v in vs])

        row = []
        for m, s in zip(mean_vs, se_vs):
            row.append(f"{round(m, 2)} +/- {round(2 * s, 2)}")
        row.append(p)
        tab.append(row)

    df_B = pd.DataFrame(tab, columns = np.concatenate((columns, ["B_0", "ShotType"]))).round(2)
    df_B["ShotType"] = df_B["ShotType"].map({0:'above head',
                             1:'dunk',
                             2:'hook shot',
                             3:'layup',
                             5:'other',
                             4:'tip-in'},
                             na_action=None)
    df_B.to_csv("part_2_betas_with_uncertainty.csv")
    print("Saved B to file")

    print("Part 3")
    X, y = multinomial_bad_ordinal_good(MBOG_TRAIN + 1000, rand)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size=1000, random_state=42)

    lr = MultinomialLogReg()
    lr.build(X_train, y_train)
    prob = lr.predict(X_test)
    y_pred = [max(enumerate(p),key=lambda x: x[1])[0] for p in prob]

    possible_outcomes = np.unique(y_train)
    for i, possible_outcome in zip(range(len(possible_outcomes)), possible_outcomes):
        y_pred = [possible_outcome if j==i else j for j in y_pred]

    print("Log score Multinomial", lr.best_log_score)

    lr = OrdinalLogReg()
    lr.build(X_train, y_train)
    prob = lr.predict(X_test)
    y_pred = [max(enumerate(p),key=lambda x: x[1])[0] for p in prob]

    for i, possible_outcome in zip(range(len(possible_outcomes)), possible_outcomes):
        y_pred = [possible_outcome if j==i else j for j in y_pred]

    print("Log score Ordinal", lr.best_log_score)

