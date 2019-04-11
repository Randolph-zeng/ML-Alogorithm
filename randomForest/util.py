from scipy import stats
import numpy as np


# This method computes entropy for information gain
def entropy(class_y):
    # Input:            
    #   class_y         : list of class labels (0's and 1's)
    #
    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92    
    entropy = 0
    classes = {}
    for cla in class_y:
        if cla not in classes:
            classes[cla] = 1
        else:
            classes[cla] +=1
    for k,v in classes.items():
        prob = v/len(class_y)
        entropy -= prob * np.log2(prob)
    return entropy


def partition_classes(X, y, split_attribute, split_val):
    
    X_left = []
    X_right = []
    y_left = []
    y_right = []
    # categorical column judgement requires domain knowledge, these columns are computed with
    # code:  categorical = all([type(row[split_attribute]) is int for row in X]) in whole dataset
    categorical = [2, 3, 4, 5, 6, 8, 9, 10, 15, 18, 19, 20, 22, 23, 24, 25, 28, 31, 32, 33, 34, 37, 39, 40]
    for i in range(len(X)):
        if split_attribute in categorical:
            if X[i][split_attribute] == split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
        else:
            if X[i][split_attribute] <= split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])   
    return (X_left, X_right, y_left, y_right)

    
def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value
    
    # TODO: Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.
    """
    Example:
    
    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    
    info_gain = 0.45915
    """
    info_gain = entropy(previous_y)
    for i in range(len(current_y)):
        prob = len(current_y[i])/len(previous_y)
        info_gain -= prob * entropy(current_y[i])
    return info_gain
    
    