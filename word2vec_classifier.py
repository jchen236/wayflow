import sys
import re
import string
import csv
import scipy.sparse as sp
import numpy as np
import gensim
from sklearn.feature_extraction.text import (TfidfVectorizer,
                                             _document_frequency)
import pickle
class MyTfidfVectorizer(TfidfVectorizer):

   def fit(self, X, y=None):
        """Learn the idf vector (global term weights)
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

           # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

           # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            ####### + 1 is commented out ##########################
            idf = np.log(float(n_samples) / df) #+ 1.0
           #######################################################
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

with open('new.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

pattern = re.compile('[\W_]+')
stops = ['my', 'need', 'want', 'any', 'there', 'into', 'have', 'all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'o', 'hadn', 'herself', 'll', 'had', 'should', 'to', 'only', 'won', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'd', 'did', 'didn', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'hasn', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 're', 'does', 'above', 'between', 'mustnt', 't', 'be', 'we', 'who', 'were', 'here', 'shouldn', 'hers', 'by', 'on', 'about', 'couldn', 'of', 'against', 's', 'isn', 'or', 'own', 'into', 'yourself', 'down', 'mightn', 'wasn', 'your', 'from', 'her', 'their', 'aren', 'there', 'been', 'whom', 'too', 'wouldn', 'themselves', 'weren', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'ma', 'these', 'up', 'will', 'below', 'ain', 'can', 'theirs', 'my', 'and', 've', 'then', 'is', 'am', 'it', 'doesnt', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'yo', 'shan', 'needn', 'haven', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'm', 'yours', 'so', 'y', 'the', 'having', 'once']

corpus = []
sentences = []
for i in range(len(your_list)):
    words = re.sub('[^0-9a-zA-Z\']+', ' ', cleanhtml(your_list[i][6]).replace('\n', ''))
    words = words.lower().split();
    words = [word for word in words if word not in stops]
    # print(words)
    # words = [word for word in words if word not in stops]
    sentences.append(words)
    for j in range(len(words)):
        corpus.append(words[j])

vectorizer = MyTfidfVectorizer()
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
tfidf_dict =  dict(zip(vectorizer.get_feature_names(), idf))
with open('tfidf_dict.pickle', 'wb') as handle:
    pickle.dump(tfidf_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
model = gensim.models.Word2Vec(sentences, size = 300, window = 5, workers = 4)
model.save('word2vec_model')
