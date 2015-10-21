import pickle
import numpy as np
import types
from sklearn.cross_validation import train_test_split
from math import log
from collections import defaultdict

# ref : http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

def p_load(file):
    with open("processed_data/" + file, 'rb') as infile:
        return pickle.load(infile)
        
# word_names = p_load('names.dat')
# single row of word names corresponding to below word columns

data_matrix = p_load('mi_features.dat')
# each row is an interview excerpt
# each column is a word
# each cell represents the number of times a word occurs in an interview

print (data_matrix.shape)

data = data_matrix[:,0:data_matrix.shape[1]-1]
target = data_matrix[:,-1]
#target.reshape()

print (data.shape)
print (target.shape)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.33, random_state=42)

# dt functions

# ref : http://kldavenport.com/pure-python-decision-trees/
def entropy(data):
    total_samples = len(data)
    class_count = defaultdict(int) # ref : http://ludovf.net/blog/python-collections-defaultdict/
    
    for x in data:
        label = x[-1]
        class_count[label] += 1 # get total count of each class
    
    probs = [c/total_samples for c in class_count] # divide counts by len
    
    entropy = 0.0
    for p in probs:
        entropy += -(p * log(p,2)) # formula for entropy: -P(i)*log2(Pi)
    
    return entropy

# ref : https://gist.github.com/cmdelatorre/fd9ee43167f5cc1da130
def information_gain(data, feature_split_index):
    total_samples = len(data)    
    data_split_on_feature = defaultdict(list)
    
    for x in data:
        # Append the data item to the list corresponding to the feature to split
        # So each list will contain data items with the same value for that feature
        data_split_on_feature[x[feature_split_index]].append(x)
        
    cond_entropy = 0.0
    for partition in data_split_on_feature.values():
        targets = [x[-1] for x in partition] # get classes for each item in this partition
        prob = len(partition) / total_samples
        # formula for cond. entropy: P(x = feature)*E(classes | x = feature)
        cond_entropy += prob * entropy(targets)
        
    info_gain = entropy(data) - cond_entropy
    return info_gain

# ref : http://www.jdxyw.com/?p=2095
def get_best_feature(data, feature_indices): # choose best feature to split on data
    best_info_gain = 0.0
    best_feature = -1
    
    for i in feature_indices: # iterate over features
        info_gain = information_gain(data, i)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_index = i
    
    return best_feature_index # return index of best feature to split on data

# ref: https://gist.github.com/cmdelatorre/fd9ee43167f5cc1da130
# feature_indices base val: list(range(len(data[0]))) I.e.: {0, ..., n} where n = num. features
# Returns a tree in the form of nested 3-tuples
#  First element is the feature on which the decision is being made
#  Second element is a dictionary containing the subtrees, indexed on the possible values of the feature
#  If the subtree is a leaf, its value will instead be an integer representing the class to predict
def build_decision_tree(data, feature_indices):
    classes = [x[-1] for x in data]
    if len(feature_indices) == 0: # no features left...
        # find most common class in the data and return a leaf with value of that class
        # ref : https://docs.python.org/2/library/collections.html#collections.Counter
        counter = Counter(classes)
        k, = counter.most_common(n=1) # get the most common class. returns array of tuples (val, count).
        commonest_class = k[0]
        return commonest_class;
    else: # some features left to look at
        if len(set(classes)) == 1: # but all the data has the same class, so it doesn't matter.
            data_class = data[0][-1] # get the class
            return data_class;
        else: # data has more than 1 class
            best_feature = get_best_feature(data, feature_indices)
            feature_indices.pop(best_feature) # remove the chosen feature from the array of features to look at
            best_feature_vals = { x[best_feature] for x in data } # create unique set of vals of best feature
            subtrees = {};
            for val in best_feature_vals: # for each possible value of the best feature
                matching_data_items = [ x for x in data if x[best_feature] == val ] # get all data items with that value
                child_node = build_decision_tree(matching_data_items, feature_indices)
                subtrees[val] = child_node;
            return (best_feature, subtrees);

# tree: A tree built by build_decision_tree
# data: A 2D list where each row is a data point, and each column represents a feature.
def predict(tree, data):
    return map(lambda d: predict_one(tree,d), data);
    #default_class = 0
    #predicted_classes = []
    #num_data_points = data.shape[0];

    #if num_data_points == 0:
    #    return [];

    #if ~isinstance(tree, types.DictType):
    #    return tree;

    #data_point = data[0];
    #feature = tree[0];
    #subtrees = tree[1];
    #prediction = predict(subtree[data_point[feature]], [data_point]);

    #return [prediction] + predict(tree, data[1:][:]);

def predict_one(tree, data):

    # If the subtree isn't a dictionary, then we're at a root
    if ~isinstance(tree, types.DictType):
        return tree;

    feature = tree[0];
    subtrees = tree[1];
    return predict(subtree[data[feature]], data);
