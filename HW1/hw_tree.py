import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def all_columns(X, rand):
    return range(X.shape[1])

def get_candidate_columns(X, rand):
    number_of_features_per_split = int(np.floor(np.sqrt(len(X[0]))))
    seq = list(range(len(X[0])))
    features_to_use_in_split = [rand.choice(seq) for _ in range(number_of_features_per_split)]
    return features_to_use_in_split

class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples

    def build(self, X, y):
        if self.stoppingCriterionMet(None, None, X, y):
            node = TreeNode(-1, -1, X, y)
            node.endNode = True
            return node
        indexes_of_left_X = []
        indexes_of_right_X = []
        while len(X[indexes_of_left_X, :]) == 0 or len(X[indexes_of_right_X, :]) == 0:
            features_to_use = self.get_candidate_columns(X, self.rand)
            indexes_of_left_X, y_l, indexes_of_right_X, y_r, criterionColumn, criterionVal = self.split(X[:, features_to_use], y)
        node = TreeNode(features_to_use[criterionColumn], criterionVal, X[:, features_to_use], y)
        if self.stoppingCriterionMet(criterionColumn, criterionVal, X, y):
            node.endNode = True
            return node
        node.L = self.build(X[indexes_of_left_X, :], y_l)
        node.R = self.build(X[indexes_of_right_X, :], y_r)
        return node

    def split(self, X, y):
        bestColumn = -1
        criterionOfBestColumn = -1
        bestCost = 999999
        X_l = []
        X_r = []
        y_l = []
        y_r = []
        for column in range(len(X[0])):
            values = X[:, column]
            for val in values:
                leftNode = y[values <= val]
                rightNode = y[values > val]

                if len(leftNode) == 0 or len(rightNode) == 0:
                    continue
                
                p = float(len(leftNode)) / (len(leftNode) + len(rightNode))

                giniLeft = self.gini(leftNode)
                giniRight = self.gini(rightNode)

                cost = p * giniLeft + (1 - p) * giniRight

                if cost < bestCost:
                    bestCost = cost
                    bestColumn = column
                    criterionOfBestColumn = val
                    X_l = values <= val
                    X_r = values > val
                    y_l = leftNode
                    y_r = rightNode

        return X_l, y_l, X_r, y_r, bestColumn, criterionOfBestColumn

    def gini(self, y):
        p = [np.average(y == a) for a in np.unique(y)]
        return 1 - np.sum(np.square(p))

    def stoppingCriterionMet(self, criterionColumn, criterionVal, X, y):
        if len(list(set(y))) == 1:
            return True

        if len(X) < self.min_samples:
            return True

        return False

class TreeNode:

    def __init__(self, criterionColumn, criterionVal, X, y):
        self.L = None
        self.R = None
        self.criterionColumn = criterionColumn
        self.criterionVal = criterionVal
        self.X = X
        self.y = y
        self.endNode = False

    def predict(self, X):
        return [self.predictOne(row) for row in X]

    def predictOne(self, row):
        if self.endNode:
            return np.unique(self.y)[0]
        elif row[self.criterionColumn] <= self.criterionVal:
            return self.L.predictOne(row)
        else:
            return self.R.predictOne(row)

class RandomForest:

    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand

    def build(self, X, y, X_test = [], y_test = []):
        return RFModel(X, y, X_test = X_test, y_test = y_test, n = self.n, rand = self.rand)

class RFModel:

    def __init__(self, X, y, X_test = [], y_test = [], n = 10, rand = random.Random(1)):
        self.X = X
        self.y = y
        self.n = n
        self.X_test = X_test
        self.y_test = y_test
        self.trees = []
        self.rand = rand
        self.importances = []
        self.out_of_bag_samples_in_tree = []
        self.generateTrees(numberOfTrees = n, predictAtEachStep=True)
    
    def generateTrees(self, numberOfTrees, predictAtEachStep = False):
        number_of_samples_per_split = len(self.X)
        misclassification_rates_test = []
        for i in range(numberOfTrees):
            seq = list(range(len(self.X)))
            samples_to_use_in_split = []
            out_of_bag_samples = list(range(len(self.X)))
            while len(samples_to_use_in_split) < number_of_samples_per_split:
                newSample = self.rand.choice(seq)
                samples_to_use_in_split.append(newSample)
                if newSample in out_of_bag_samples:
                    out_of_bag_samples.remove(newSample)
            X_split = self.X[samples_to_use_in_split, :]
            y_split = self.y[samples_to_use_in_split]
            newTree = Tree(rand=self.rand, get_candidate_columns=get_candidate_columns, min_samples=2).build(X_split, y_split)
            self.trees.append(newTree)
            self.out_of_bag_samples_in_tree.append(out_of_bag_samples)
            

            if predictAtEachStep and i%1 == 0:
                y_preds = self.predict(self.X)
                CA_train = np.sum(y_preds == self.y)/len(self.y)
                y_preds = self.predict(self.X_test)
                CA_train = np.sum(y_preds == self.y_test)/len(self.y_test)
                misclassification_rates_test.append(1 - CA_train)
        np.savetxt("misclassification_rates_test_RF_over_n.csv", 
           [list(range(1, numberOfTrees + 1)), misclassification_rates_test],
           delimiter =", ", 
           fmt ='% s')

    def most_frequent(self, votes):

        classes = list(set(self.y))
        output = []
        votes = np.array(votes)
        
        for sample_index in range(len(votes[0])):

            topFreqCl = -1
            counter = 0
            for cl in classes:
                curr_frequency = np.count_nonzero(votes[:, sample_index] == cl)
                if(curr_frequency > counter):
                    counter = curr_frequency
                    topFreqCl = cl
            output.append(topFreqCl)
    
        return output

    def predict(self, X):
        votes = [tree.predict(X) for tree in self.trees]
        return self.most_frequent(votes)
        

    def importance(self):
        self.importances = []
        for i in range(self.n):
            out_of_bag_samples = self.out_of_bag_samples_in_tree[i]
            out_of_bag_X = self.X[out_of_bag_samples, :]
            out_of_bag_y = self.y[out_of_bag_samples]
            newTree = self.trees[i]
            y_predictions = newTree.predict(out_of_bag_X)
            real_misclassification_error = 1 - np.average(y_predictions == out_of_bag_y)

            shuffled_misclassification_errors = []
            if len(out_of_bag_X) > 0 and len(out_of_bag_X[0]) > 0:
                for column in range(len(out_of_bag_X[0])):
                    permuted_out_of_bag_X = out_of_bag_X.copy()
                    self.rand.shuffle(permuted_out_of_bag_X[:, column])
                    y_predictions = newTree.predict(permuted_out_of_bag_X)
                    shuffled_misclassification_errors.append(1- np.average(y_predictions == out_of_bag_y))
                
                importances = np.array(shuffled_misclassification_errors) - np.array([real_misclassification_error]*len(shuffled_misclassification_errors))
                self.importances.append(importances)
        feature_importances = [np.average(misclaffication_rates_of_feature) for misclaffication_rates_of_feature in np.transpose(self.importances)]
        return feature_importances

def normalize2(X, X_test):
    for i in range(len(X[0])):
        X[:, i] = (X[:, i] - np.average(X[:, i]))/np.std(X[:, i])
        X_test[:, i] = (X_test[:, i] - np.average(X_test[:, i]))/np.std(X_test[:, i])
    return X, X_test

def tki(normalize=False):
    df = pd.read_csv("tki-resistance.csv")
    df.columns=list(range(len(df.columns)-1)) + ['y']
    train = df.head(n=130)
    test = df.tail(n=len(df.index) - 130)
    X = train.loc[:, train.columns != 'y'].to_numpy()
    X_test = test.loc[:, test.columns != 'y'].to_numpy()
    if normalize:
        X, X_test = normalize2(X, X_test)
    y = train['y'].to_numpy()
    y_test = test['y'].to_numpy()
    legend = None
    return (X, y), (X_test, y_test), legend

def hw_tree_full(train, test):
    rand = random.Random(1)
    X = np.array(train[0])
    X_test = np.array(test[0])
    y = train[1]
    y_test = test[1]

    tree = Tree(rand=rand, min_samples=2).build(X, y)

    CA_train = np.average(tree.predict(X) == y)
    number_of_samples = len(X)
    accuracies_for_train = []
    for i in range(100):
        seq = list(range(len(X)))
        samples_to_use_in_split = []
        while len(samples_to_use_in_split) < number_of_samples:
            newSample = rand.choice(seq)
            samples_to_use_in_split.append(newSample)
        y_predictions = tree.predict(X[samples_to_use_in_split, :])
        accuracies_for_train.append(np.average(y_predictions == y[samples_to_use_in_split]))
    SE_train = np.std(accuracies_for_train)

    CA_test = np.average(tree.predict(X_test) == y_test)
    number_of_samples = len(X_test)
    accuracies_for_test = []
    for i in range(100):
        seq = list(range(len(X_test)))
        samples_to_use_in_split = []
        while len(samples_to_use_in_split) < number_of_samples:
            newSample = rand.choice(seq)
            samples_to_use_in_split.append(newSample)
        y_predictions = tree.predict(X_test[samples_to_use_in_split, :])
        accuracies_for_test.append(np.average(y_predictions == y_test[samples_to_use_in_split]))
    SE_test = np.std(accuracies_for_test)

    return (1.0 - CA_train, SE_train), (1.0 - CA_test, SE_test)

def rf_importance(rf, n, X, y, rand):
    importance_of_feature = rf.importance()

    number_of_samples = len(X)
    times_root_feature = [0] * len(X[0])
    for i in range(100):
        seq = list(range(number_of_samples))
        samples_to_use_in_split = []
        while len(samples_to_use_in_split) < number_of_samples:
            newSample = rand.choice(seq)
            samples_to_use_in_split.append(newSample)
        tree = Tree(rand=rand, min_samples=2).build(X[samples_to_use_in_split, :], y[samples_to_use_in_split])
        root_feature_in_tree = tree.criterionColumn
        times_root_feature[root_feature_in_tree] += 1

    x = np.array(range(len(X[0])))
    times_root_feature = np.array(times_root_feature)
    importance_of_feature = np.array(importance_of_feature)
    possible_values = list(set(times_root_feature))
    possible_values.sort()
    for possible_value in possible_values:
        plt.scatter(x[times_root_feature == possible_value], importance_of_feature[times_root_feature == possible_value], label = possible_value)
    plt.title("Misclassification increase permuted features")
    plt.xlabel("Index of used features")
    plt.ylabel("Misclassification increase")
    plt.legend()
    plt.savefig(f"feature_importance_with_{n}_trees.png", format="png")
    plt.show()

def hw_randomforests(train, test, n = 5):
    rand = random.Random(1)
    X = train[0]
    X_test = test[0]
    y = train[1]
    y_test = test[1]

    rf = RandomForest(rand=rand, n=n).build(X, y, X_test, y_test)

    rf_importance(rf, n, X, y, rand)
    
    CA_train = np.average(rf.predict(X) == y)
    number_of_samples = len(X)
    accuracies_for_train = []
    for i in range(100):
        seq = list(range(len(X)))
        samples_to_use_in_split = []
        while len(samples_to_use_in_split) < number_of_samples:
            newSample = rand.choice(seq)
            samples_to_use_in_split.append(newSample)
        y_predictions = rf.predict(X[samples_to_use_in_split, :])
        accuracies_for_train.append(np.average(y_predictions == y[samples_to_use_in_split]))
    SE_train = np.std(accuracies_for_train)

    CA_test = np.average(rf.predict(X_test) == y_test)
    number_of_samples = len(X_test)
    accuracies_for_test = []
    for i in range(100):
        seq = list(range(len(X_test)))
        samples_to_use_in_split = []
        while len(samples_to_use_in_split) < number_of_samples:
            newSample = rand.choice(seq)
            samples_to_use_in_split.append(newSample)
        y_predictions = rf.predict(X_test[samples_to_use_in_split, :])
        accuracies_for_test.append(np.average(y_predictions == y_test[samples_to_use_in_split]))
    SE_test = np.std(accuracies_for_test)

    return (1 - CA_train, SE_train), (1 - CA_test, SE_test)

def plot_randomforests_over_n():
    df_RF_over_n = pd.read_csv("misclassification_rates_test_RF_over_n.csv")
    test_accuracy = df_RF_over_n.values[0]
    
    fig, ax = plt.subplots()
   
    ax.set_xlabel('Number of trees') 
    ax.set_ylabel('Misclassification rate') 
    
    ax.plot(test_accuracy)
    ax.set_title("Misclassification rate over number of trees in RF")
    plt.savefig("RF_over_n.png", format="png")
    
if __name__ == "__main__":
    learn, test, legend = tki(normalize=False)

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test, n = 100))
    plot_randomforests_over_n()
