import pickle
import numpy as np

def p_load(file):
    with open(file, 'rb') as infile:
        return pickle.load(infile)
        
word_names = p_load('names.dat')
# single row of word names corresponding to below word columns

word_count_matrix = p_load('words.dat')
# each row is an interview excerpt
# each column is a word
# each cell represents the number of times a word occurs in an interview