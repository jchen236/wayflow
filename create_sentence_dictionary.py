import sys
import re
import string
import csv
import scipy.sparse as sp
import numpy as np
import gensim
import numpy as np
import collections
from sklearn.feature_extraction.text import (TfidfVectorizer,
                                             _document_frequency)
import pickle

with open('tfidf_dict.pickle', 'rb') as handle:
    tfidf_dict = pickle.load(handle)

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

with open('new.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
#
# with open('new_answers.csv', 'r') as r:
#     reader2 = csv.reader(r)
#     answer_list = list(reader2)

pattern = re.compile('[\W_]+')

def convert(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data
tfidf_dict = convert(tfidf_dict)

for i in range(len(your_list)):
    your_list[i][6] = re.sub('[^0-9a-zA-Z\']+', ' ', cleanhtml(your_list[i][6]).replace('\n', ''))

model = gensim.models.Word2Vec.load('word2vec_model')

print(tfidf_dict)
def avg_feature_vector(words, model, num_features, index2word_set):
        #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    #list containing names of words in the vocabulary
    #index2word_set = set(model.index2word) this is moved as input param for performance reasons
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            print(nwords)
            featureVec = np.add(featureVec, model.wv[word]) * tfidf_dict[word]

    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec
print(avg_feature_vector('leased', model, 300, tfidf_dict))
