import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from math import log2
from collections import defaultdict

# ref : http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

def p_load(file):
    with open("processed_data/" + file, 'rb') as infile:
        return pickle.load(infile)
        
# word_names = p_load('names.dat')
# single row of word names corresponding to below word columns

data_matrix = p_load('mi.dat')
# each row is an interview excerpt
# each column is a word
# each cell represents the number of times a word occurs in an interview

print (word_count_matrix.shape)

data = word_count_matrix[:,0:word_count_matrix.shape[1]-1]
target = word_count_matrix[:,-1]
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
        entropy += -(p * log2(p)) # formula for entropy: -P(i)*log2(Pi)
    
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
def get_best_feature(data): # choose best feature to split on data
    total_features = len(data[0]) - 1
    best_info_gain = 0.0
    best_feature = -1
    
    for i in range(total_features): # iterate over features
        info_gain = information_gain(data, i)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    
    return best_feature

def build_decision_tree(data):
    