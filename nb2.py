import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from common import loadTrainingData
from common import loadTestData
from pfile import p_load
from sklearn.feature_extraction.text import CountVectorizer
import csv

# ref : http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

features = p_load('mi_features.dat');

interviews = loadTrainingData();
testInterviews = loadTestData();

data = [i[0] for i in interviews];
target = [i[1] for i in interviews];

cv = CountVectorizer(vocabulary=features);
wordCounts = cv.fit_transform(data).sign();
cvt = CountVectorizer(vocabulary=features);
wordCountsTest = cv.fit_transform(testInterviews).sign();

#data_train, data_test, target_train, target_test = train_test_split(wordCounts, target, test_size=0.33, random_state=42)
data_train = wordCounts;
target_train = target;
data_test = wordCountsTest;

gnb = GaussianNB()
y_pred = gnb.fit(data_train.toarray(), target_train).predict(data_test.toarray())
#print("Number of mislabeled points out of a total %d points : %d" % (data_test.shape[0],(target_test != y_pred).sum()))

output = [[i, int(y_pred[i])] for i in range(len(y_pred))];
np.savetxt('submission.csv', np.asarray(output), header='Id,Prediction', fmt='%d,%d', comments='')
