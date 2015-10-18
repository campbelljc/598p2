import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from math import log2

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

def entropy(class_list):
    # get number of each class
    total_samples = len(class_list)
    class_count = dict()
    for x in class_list:  # x is class name
        class_count[x] += 1
    probs = [x/total_samples for c in class_count] # divide counts by len
    entropy = 0.0
    for p in probs:
        entropy += -(p * log2(p))
    return entropy

#def information_gain(data, classes, feature):
#    