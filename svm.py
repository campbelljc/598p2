from __future__ import division
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split

def p_load(file):
    with open("processed_data/" + file, 'rb') as infile:
        return pickle.load(infile)
        
dataset = p_load('mi.dat')
data = dataset[:,0:dataset.shape[1]-1]
target = dataset[:,-1]

print (dataset.shape)
print (data.shape)
print (target.shape)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.20, random_state=42)

# ref : http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)

predicted_train = clf.fit(data_train, target_train).predict(data_train)

print("Number of mislabeled points out of a total %d points : %d (training)" % (data_train.shape[0],(target_train != predicted_train).sum()))
train_p = ((target_train != predicted_train).sum())/(data_train.shape[0])*100
print("Training error: %d" % train_p)

predicted_test = clf.fit(data_train, target_train).predict(data_test)

print("Number of mislabeled points out of a total %d points : %d (training)" % (data_test.shape[0],(target_test != predicted_test).sum()))
test_p = ((target_test != predicted_test).sum())/(data_test.shape[0])*100
print("Training error: %d" % test_p)