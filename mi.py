import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords # import the stop word list
from stemming.porter2 import stem
from nltk.stem.snowball import SnowballStemmer
import itertools
import math
import pickle
import numpy as np
import string
from pfile import p_save

#stemmer = SnowballStemmer("english")
def parse_interview(raw_text):
    lower_case = raw_text.lower() # Convert to lower case
    lower_case = lower_case.decode('utf-8', 'ignore')
    lower_case = lower_case.replace("__eos__", " ")
    lower_case = lower_case.replace("-", " ")
    lower_case = lower_case.replace("/", " ")
#    lower_case = "".join(l for l in lower_case if l not in string.punctuation)
    words = lower_case.split() # Split into words
#    stops = set(stopwords.words("english"))
#    words = [w for w in words if not w in stops] #remove stopwords
#    words = [ stemmer.stem(w) for w in words ]
    return( " ".join( words ))

def save_binary(words, filename, parsed_texts): #, predictions):
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
#    features_arr = np.insert(features_arr, features_arr.shape[1], values=predictions, axis=1)
    p_save(features_arr, filename)


def p_load(file):
    with open("processed_data/" + file, 'rb') as infile:
        return pickle.load(infile)

print("Loading dataset.")

features = p_load('mi_features.dat')

data = []
ifile  = open('data/ml_dataset_test_in.csv', "r")
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
for item in data:
    parsed_texts.append(parse_interview(item[1]))
        
save_binary(features, 'mi_test.dat', parsed_texts) #, predictions)