import csv
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# src : https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

def parse_interview(raw_text):
    lower_case = raw_text.lower() # Convert to lower case
    words = lower_case.split() # Split into words
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops] #remove stopwords
    return( " ".join( meaningful_words ))

ifile  = open('ml_dataset_train.csv', "r")
reader = csv.reader(ifile)

data = []

print("Loading dataset.")
for row in reader:
    # Save header row.
    data.append(row)

ifile.close()

parsed_texts = []

print("Parsing text.")
for item in data:
    if (len(item) == 3):
        parsed_texts.append(parse_interview(item[1]))
    
print("Getting bag of words.")
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(parsed_texts).toarray()

#print(train_data_features.shape)
#vocab = vectorizer.get_feature_names()
#print(vocab)
# tf-idf

print("Tf-idf.")

transformer = TfidfTransformer()

transformer.fit(train_data_features)
print(transformer.transform(train_data_features).toarray())