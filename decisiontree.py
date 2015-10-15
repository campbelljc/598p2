import pickle
import numpy as np

# ref : http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

def p_load(file):
    with open("processed_data/" + file, 'rb') as infile:
        return pickle.load(infile)
        
word_names = p_load('names.dat')
word_count_matrix = p_load('words.dat')
tfidf_words = p_load('tfidf_words.dat')

