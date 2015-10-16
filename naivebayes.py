import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB

# ref : http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

def p_load(file):
    with open("processed_data/" + file, 'rb') as infile:
        return pickle.load(infile)
        
word_names = p_load('names.dat')
# single row of word names corresponding to below word columns

word_count_matrix = p_load('words.dat')
# each row is an interview excerpt
# each column is a word
# each cell represents the number of times a word occurs in an interview

data = word_count_matrix[:][0:2000]
target = word_count_matrix[:][-1]

gnb = GaussianNB()
y_pred = gnb.fit(data, target).predict(data)
print("Number of mislabeled points out of a total %d points : %d", (iris.data.shape[0],(iris.target != y_pred).sum()))