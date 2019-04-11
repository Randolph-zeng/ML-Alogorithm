from util import entropy, information_gain, partition_classes
import numpy as np 
import ast



class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.root = {}
        self.max_depth = 10
        self.min_size = 20

    def _get_split_val(self, X):
        ''' return a dictionary with key being split attributes and
        values being an array of split values '''
        split_dict = {}
        # np.random.choice(len(X[0]), int(np.sqrt(len(X[0]))))
        for split_attr in np.random.choice(len(X[0]), int(np.sqrt(len(X[0]))), replace=False):
            split_values = set()
            for row in range(len(X)):
                split_values.add(X[row][split_attr])
            split_dict[split_attr] = list(split_values)
        return split_dict
        
    def _split_helper(self, X, y):
        ''' Find best cutpoint among all attributes and return a node '''
        split_dict = self._get_split_val(X)
        best_gain = -1
        best_split_attr = -1
        best_split_val = None
        left_group = None
        right_group = None
        for split_attr, split_values in split_dict.items():
            for split_val in split_values:
                (X_left, X_right, y_left, y_right) = partition_classes(
                    X, y, split_attr, split_val)
                info_gain = information_gain(y, [y_left, y_right])
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_split_attr = split_attr
                    best_split_val = split_val
                    left_group = X_left, y_left
                    right_group = X_right, y_right

        return {"split_attr":best_split_attr, "split_val":best_split_val,
            "left_group": left_group, "right_group": right_group}

    def _build_leaf(self, y):
        return np.argmax(np.bincount(y))


    def _split(self, node, depth):
        (X_left,y_left) = node["left_group"]
        (X_right,y_right) = node["right_group"]
        del node["left_group"] 
        del node["right_group"]
        # if node is pure 
        if not len(X_left) or not len(X_right):
            node["left"] = node["right"] = self._build_leaf((y_left+y_right))
            return
        if depth >= self.max_depth:
            node["left"] = self._build_leaf(y_left)
            node["right"] = self._build_leaf(y_right)
            return
        # check if y are from same class
        if len(X_left) <= self.min_size :
            node["left"] = self._build_leaf(y_left)
        else:
            if np.sum(y_left) == len(y_left) or np.sum(y_left) == 0 :
                node["left"] = self._build_leaf(y_left)
            else:    
                node["left"] = self._split_helper(X_left, y_left)
                self._split(node["left"], depth+1)
        if len(X_right) <= self.min_size:
            node["right"] = self._build_leaf(y_right)
        else:
            if np.sum(y_right) == len(y_right) or np.sum(y_right) == 0:
                node["right"] = self._build_leaf(y_right)
            else:    
                node["right"] = self._split_helper(X_right, y_right)
                self._split(node["right"], depth+1)


    def learn(self, X, y):
        ''' '''
        self.root = self._split_helper(X, y)
        self._split(self.root, 1)
        
        

    def classify(self, record):
        node = self.root
        # it is not robust to look at a single record entry to judge if it is categorical or not, 
        # it requires domain knowledge, following categorical columns is computed beforehand 
        categorical =  [2, 3, 4, 5, 6, 8, 9, 10, 15, 18, 19, 20, 22, 23, 24, 25, 28, 31, 32, 33, 34, 37, 39, 40]
        while True:
            if node["split_attr"] in categorical:
                if record[node["split_attr"]] == node["split_val"]:
                    if type(node["left"]) is dict:
                        node = node["left"]
                        continue
                    else:
                        return node["left"]
                else:
                    if type(node["right"]) is dict:
                        node = node["right"]
                        continue
                    else:
                        return node["right"]
            else:
                if record[node["split_attr"]] <= node["split_val"]:
                    if type(node["left"]) is dict:
                        node = node["left"]
                        continue
                    else:
                        return node["left"]
                else:
                    if type(node["right"]) is dict:
                        node = node["right"]
                        continue
                    else:
                        return node["right"]

