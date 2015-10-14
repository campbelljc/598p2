import csv
from nltk.corpus import stopwords # import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from stemming.porter2 import stem
from nltk.stem.snowball import SnowballStemmer
import string
import regex

# src : https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

stemmer = SnowballStemmer("english")
def parse_interview(raw_text):
    lower_case = raw_text.lower() # Convert to lower case
    lower_case = lower_case.replace("__eos__", " ")
    lower_case = lower_case.replace("-", " ")
    lower_case = lower_case.replace("/", " ")
    lower_case = "".join(l for l in lower_case if l not in string.punctuation)
    words = lower_case.split() # Split into words
    large_words = [w for w in words if len(w) > 2]
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in large_words if not w in stops] #remove stopwords
    stemmed_words = [ stemmer.stem(w) for w in meaningful_words ]
    return( " ".join( stemmed_words ))

print("Loading dataset.")

data = []
ifile  = open('data/ml_dataset_train.csv', "r")
reader = csv.reader(ifile)
for row in reader:
    data.append(row)
ifile.close()

print("Parsing text.")

parsed_texts = []
for item in data:
    if (len(item) == 3):
        parsed_texts.append(parse_interview(item[1]))
    
print("Getting bag of words.")

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  
vec = CountVectorizer(analyzer = "word") #,   \
                       #      max_features = 5000) 
                       
for i in parsed_texts:
    print("\n" + i)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vec.fit_transform(parsed_texts)

vocab = vec.get_feature_names()
print(vocab)
print(train_data_features)
print(train_data_features.shape)

print("Tf-idf.")

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(train_data_features)
print(tfidf)
print(tfidf.shape)