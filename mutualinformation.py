import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords # import the stop word list
from stemming.porter2 import stem
from nltk.stem.snowball import SnowballStemmer
import itertools
import math
import numpy as np
import string
from pfile import p_save
from sklearn.cross_validation import train_test_split

#stemmer = SnowballStemmer("english")
def parse_interview(raw_text):
    lower_case = raw_text.lower() # Convert to lower case
    lower_case = lower_case.decode('utf-8', 'ignore')
    lower_case = lower_case.replace("__eos__", " ")
    lower_case = lower_case.replace("-", " ")
    lower_case = lower_case.replace("/", " ")
 #   lower_case = "".join(l for l in lower_case if l not in string.punctuation)
    words = lower_case.split() # Split into words
#    stops = set(stopwords.words("english"))
#    words = [w for w in words if not w in stops] #remove stopwords
#    words = [ stemmer.stem(w) for w in words ]
    return( " ".join( words ))
    
# ref: http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

def save_binary(words, filename, parsed_texts, predictions):
    print("Saving to binary file.")
    vec = CountVectorizer(analyzer = "word", vocabulary = words)                        
    train_data_features = vec.fit_transform(parsed_texts)
    features_arr = train_data_features.toarray()
    features_arr = np.sign(features_arr)
    
    i = 0
    for row in features_arr:
        i += 1
        if i > 200:
            break
            
    print(features_arr)
    features_arr = np.insert(features_arr, features_arr.shape[1], values=predictions, axis=1)
    p_save(features_arr, filename)

NUM_CLASSES = 4;

file = open('data/ml_dataset_train.csv', 'rb');
fileReader = csv.reader(file);

# First line is a header, so skip it.
fileReader.next();

# Loop through all interviews
interviews = [];
for row in fileReader:
	index = row[0];
	data = row[1];
	output = row[2];

        # Remove __EOS__
        interview = data.replace('__EOS__', '');

	# Trim the strings (Saves a bit of space?)
        interview.strip();

	# Add to the list
	interviews.append((interview, output));

data = [interview[0] for interview in interviews];
data_train, data_validation = train_test_split(data, test_size=0.25, random_state=42)

# Count the words in each document
cv = CountVectorizer();
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
NUM_FEATURES = 1000;
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
  #  print(' ' + feature);
    features.append(feature);

p_save(features, "mi_features.dat");

print("Loading dataset.")

data = []
ifile  = open('data/ml_dataset_train.csv', "r")
reader = csv.reader(ifile)
i = 0
for row in reader:
    if i == 0:
        i = 1
        continue
    data.append(row)

ifile.close()

print("Parsing text.")

parsed_texts = []
predictions = []
for item in data:
    if (len(item) == 3):
        parsed_texts.append(parse_interview(item[1]))
        predictions.append(item[2])
        
save_binary(features, 'mi.dat', parsed_texts, predictions)