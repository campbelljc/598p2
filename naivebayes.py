from __future__ import division
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

# ref : http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

def p_load(file):
    with open("processed_data/" + file, 'rb') as infile:
        return pickle.load(infile)
        
dataset = p_load('mi.dat')
data = dataset[:,0:dataset.shape[1]-1]
target = dataset[:,-1]

data_train_full = data
target_train_full = data

print (dataset.shape)
print (data.shape)
print (target.shape)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.20, random_state=42)

gnb = GaussianNB()
y_pred = gnb.fit(data_train, target_train).predict(data_train)
#for i in y_pred:
#    print i
#print(y_pred)
print("Number of mislabeled points out of a total %d points : %d (training)" % (data_train.shape[0],(target_train != y_pred).sum()))
train_p = ((target_train != y_pred).sum())/(data_train.shape[0])*100
print("Training error: %d" % train_p)

y_pred = gnb.fit(data_train, target_train).predict(data_test)
#for i in y_pred:
#    print i
#print(y_pred)
print("Number of mislabeled points out of a total %d points : %d (test)" % (data_test.shape[0],(target_test != y_pred).sum()))
test_p = ((target_test != y_pred).sum())/(data_test.shape[0])*100
print("Test error: %d" % test_p)
