import csv
import pickle
from nltk.corpus import stopwords # import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from stemming.porter2 import stem
from nltk.stem.snowball import SnowballStemmer
import string
import numpy as np

# src : https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

stemmer = SnowballStemmer("english")
def parse_interview(raw_text):
    lower_case = raw_text.lower() # Convert to lower case
    lower_case = lower_case.decode('utf-8', 'ignore')
    lower_case = lower_case.replace("__eos__", " ")
    lower_case = lower_case.replace("-", " ")
    lower_case = lower_case.replace("/", " ")
    lower_case = "".join(l for l in lower_case if l not in string.punctuation)
    words = lower_case.split() # Split into words
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops] #remove stopwords
    stemmed_words = [ stemmer.stem(w) for w in meaningful_words ]
    return( " ".join( words ))
    
# ref: http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

def p_save(obj, filename):
    with open("processed_data/" + filename, 'wb') as outfile:
        pickle.dump(obj, outfile, 2)
    #np.savetxt(filename, full_matrix, delimiter=",") # save as csv (large files)
    
def save_binary(words, filename, parsed_texts, predictions):
    vec = CountVectorizer(analyzer = "word", vocabulary = words)                        
    train_data_features = vec.fit_transform(parsed_texts)
    print("Saving to binary file.")
    features_arr = train_data_features.toarray()
    for row in features_arr:
        for col in row:
            if (col > 1):
                col = 1
    features_arr = np.insert(features_arr, features_arr.shape[1], values=predictions, axis=1)
    p_save(features_arr, filename)

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
        
print("Getting bag of words.")

vec = CountVectorizer(analyzer = "word", max_features = 3000) 
                       
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vec.fit_transform(parsed_texts)

print("Saving to file.")

features_arr = train_data_features.toarray()
# add predictions as last column of features array
features_arr = np.insert(features_arr, features_arr.shape[1], values=predictions, axis=1)
p_save(features_arr, 'words.dat')

# save feature names (header row of features array)
vocab = vec.get_feature_names()
vocab.append("Prediction")
p_save(vocab, 'names.dat')

print("Tf-idf.")

vectorizer = TfidfVectorizer(min_df=1)
tfidf = vectorizer.fit_transform(parsed_texts)

print("Getting high-valued words.")

# ref : http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
high_feature_names = set()
feature_names = vectorizer.get_feature_names()
i = 0
threshold = 0.6
for item in parsed_texts:
    item_vals = vectorizer.transform([item])
    i += 1
    for col in item_vals.nonzero()[1]:
        if (item_vals[0, col] > threshold):
            high_feature_names.add(feature_names[col])
            
print(len(high_feature_names))

save_binary(high_feature_names, 'tfidf.dat', parsed_texts, predictions)