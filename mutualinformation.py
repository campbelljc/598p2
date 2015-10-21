import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import math
import numpy as np
from pfile import p_save
from common import loadTrainingData;
from nltk.tokenize import TreebankWordTokenizer

NUM_CLASSES = 4;

interviews = loadTrainingData();
data = [interview[0] for interview in interviews];

# Count the words in each document
#cv = CountVectorizer(ngram_range=(1,2));
cv = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize);
#cv = CountVectorizer();
wordCounts = cv.fit_transform(data);
wordPresence = wordCounts.sign();

# Split matrix between document classes
classIndices = [0] * NUM_CLASSES;
for i in range(NUM_CLASSES):
    classIndices[i] = [j for j in range(0, len(interviews)) if interviews[j][1] == str(i)]

# Number of different features in the corpus
numWords = len(cv.get_feature_names());
# Number of different documents in the corpus
numDocs = float(len(interviews));
# Number of documents of each class in the corpus
# Each row of the matrix represents a class
numDocsInClass = np.zeros([NUM_CLASSES,1]);
for i in range(NUM_CLASSES):
    numDocsInClass[i] = len(classIndices[i]);
# Number of documents of each class for which each word/feature appears
classSumsList = [0] * NUM_CLASSES;
for i in range(NUM_CLASSES):
    classSumsList[i] = wordPresence[classIndices[i]].sum(0);
classSums = np.vstack(classSumsList);
# Number of documents in which each each word/feature appears
wordPresenceCount = classSums.sum(0);

# Compute MI for all class-feature pairs 
# See http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
# Equation 13.17
N__ = np.matrix(numDocs * np.ones([NUM_CLASSES,numWords]));
N1_ = np.matrix(numDocsInClass * np.ones([1,numWords]));
N0_ = N__ - N1_;
N_1 = np.matrix(np.ones([NUM_CLASSES,1]) * wordPresenceCount);
N_0 = N__ - N_1;
N11 = classSums;
N10 = N1_ - N11;
N01 = N_1 - N11;
N00 = N0_ - N01;
#mi = np.zeros([NUM_CLASSES, numWords]);

w = np.multiply(np.divide(N11,N__), np.log2(np.divide(np.multiply(N__,N11),np.multiply(N1_,N_1))));
x = np.multiply(np.divide(N01,N__), np.log2(np.divide(np.multiply(N__,N01),np.multiply(N0_,N_1))));
y = np.multiply(np.divide(N10,N__), np.log2(np.divide(np.multiply(N__,N10),np.multiply(N1_,N_0))));
z = np.multiply(np.divide(N00,N__), np.log2(np.divide(np.multiply(N__,N00),np.multiply(N0_,N_0))));

w[w != w] = 0;
x[x != x] = 0;
y[y != y] = 0;
z[z != z] = 0;

mi = w+x+y+z;

# Sort them in descending order of mutual information
author    = np.argsort(-np.array(mi[0])[0])
movie     = np.argsort(-np.array(mi[1])[0])
music     = np.argsort(-np.array(mi[2])[0])
interview = np.argsort(-np.array(mi[3])[0])

# Print out a list of features
NUM_FEATURES = 200;
#print('Authors: ');
#for i in range(NUM_FEATURES):
#    print(' ' + cv.get_feature_names()[author[i]]);
#print('Movie: ');
#for i in range(NUM_FEATURES):
#    print(' ' + cv.get_feature_names()[movie[i]]);
#print('Music: ');
#for i in range(NUM_FEATURES):
#    print(' ' + cv.get_feature_names()[music[i]]);
#print('Interview: ');
#for i in range(NUM_FEATURES):
#    print(' ' + cv.get_feature_names()[interview[i]]);

# Sum up the mutual information for all classes
miSum = mi.sum(0);
sortedIndices = np.argsort(-np.array(miSum[0])[0]);
features = [];
for i in range(NUM_FEATURES):
    feature = cv.get_feature_names()[sortedIndices[i]];
    print(' ' + feature);
    features.append(feature);
p_save(features, "mi_features.dat");
