from decision_tree import DecisionTree
import csv
import numpy as np  
import ast



class RandomForest(object):
    num_trees = 0
    decision_trees = []
    bootstraps_datasets = []
    bootstraps_labels = []

    def __init__(self, num_trees):
        # Initialization done here
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]


    def _bootstrapping(self, XX, n):
        samples = [] 
        labels = []  
        indices = np.random.choice(n,n,replace=True)
        for index in indices:
            samples.append(XX[index][:-1])
            labels.append(XX[index][-1])
        return (samples, labels)


    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)


    def fitting(self):
        
        for i in range(self.num_trees):
            X,y = (self.bootstraps_datasets[i], self.bootstraps_labels[i])
            self.decision_trees[i].learn(X, y)
              

    def voting(self, X):
        y = []

        for record in X:
            # Following steps have been performed here:
            #   1. Find the set of trees that consider the record as an 
            #      out-of-bag sample.
            #   2. Predict the label using each of the above found trees.
            #   3. Use majority vote to find the final label for this recod.
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)
            counts = np.bincount(votes)
            
            if len(counts) == 0:
                # TODO: Special case 
                #  Handle the case where the record is not an out-of-bag sample
                #  for any of the trees. 
                y = np.append(y, 0)
            else:
                y = np.append(y, np.argmax(counts))

        return y

