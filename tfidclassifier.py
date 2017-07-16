import re
import string
import numpy as np
from numpy import genfromtxt
import csv
from os import listdir
from os.path import isfile, join
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim import corpora, models, similarities
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

with open('new.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

pattern = re.compile('[\W_]+')


# print(your_list)[1]
print("AAAAAAAA\n")
question_list = []
for i in range(100):
    your_list[i][6] = re.sub('[^0-9a-zA-Z\']+', ' ', cleanhtml(your_list[i][6]).replace('\n', ''))
    question_list.append(your_list[i][6])
vect = TfidfVectorizer(min_df=0, stop_words='english', ngram_range=(1,7))

# print(question_list)
tfidf = vect.fit_transform(question_list)
print(tfidf).shape
from sklearn.metrics.pairwise import cosine_similarity
print cosine_similarity(tfidf[0:1], tfidf)
# print(your_list[1])
