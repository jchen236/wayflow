import re
import string
import numpy as np
from numpy import genfromtxt
import csv
from os import listdir
from os.path import isfile, join
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim import models
from sklearn.metrics.pairwise import cosine_similarity

model = models.Doc2Vec.load('questiondupemodel3')
term = model.docvecs['SENT_0']

# search_phrase = ['to', 'reverse', 'linked', 'list', 'linkedlist']
# term = model.infer_vector(search_phrase, alpha=0.025, min_alpha=0.025, steps=20)

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
for i in range(len(your_list)):
    your_list[i][6] = re.sub('[^0-9a-zA-Z\']+', ' ', cleanhtml(your_list[i][6]).replace('\n', ''))


max_similarity = 0
max = 0

for i in range(8000):
    print('hi')
    index = 'SENT_' + str(i)
    if cosine_similarity(model.docvecs[index], term) > max_similarity and  i != 0:
        max_similarity = cosine_similarity(model.docvecs[index], term)
        max = i

print(your_list[0][6])
print(max)
print(max_similarity)
print(your_list[max][6])
topn = model.docvecs.most_similar([term], topn = 5)
indices = [(i[0]) for i in topn]
print(indices)
for index, item in enumerate(indices):
    indices[index] = your_list[(int)(item.replace('SENT_', ''))]
    print(indices[index])
