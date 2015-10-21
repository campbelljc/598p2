from __future__ import division
import csv
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

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.25, random_state=42)
data_train_full = data
target_train_full = target
data_test_full = p_load('mi_test.dat')

# ref : http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.005, n_iter=10, random_state=42, n_jobs=-1)

predicted_train = clf.fit(data_train_full, target_train_full).predict(data_train_full)
#print("Number of mislabeled points out of a total %d points : %d (training)" % (data_train_full.shape[0],(target_train_full != predicted_train).sum()))
train_p = ((target_train_full != predicted_train).sum())/(data_train_full.shape[0])*100
print("Training error on full set: %d" % train_p)

predicted_test = clf.fit(data_train_full, target_train_full).predict(data_test_full)

print("Saving to csv")

save_data = []
for i in range(len(data_test_full)):
    save_data.append((i, predicted_test[i]))

with open('data/svm_mi_test.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Id', 'Prediction'])
    for row in save_data:
        csv_out.writerow(row)

predicted_train = clf.fit(data_train, target_train).predict(data_train)
#print("Number of mislabeled points out of a total %d points : %d (training)" % (data_train.shape[0],(target_train != predicted_train).sum()))
train_p = ((target_train != predicted_train).sum())/(data_train.shape[0])*100
print("Training error on 75p: %d" % train_p)

predicted_test = clf.fit(data_train, target_train).predict(data_test)
#print("Number of mislabeled points out of a total %d points : %d (training)" % (data_test.shape[0],(target_test != predicted_test).sum()))
test_p = ((target_test != predicted_test).sum())/(data_test.shape[0])*100
print("Validation error on 25p: %d" % test_p)