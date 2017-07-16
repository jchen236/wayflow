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
from nltk.corpus import stopwords

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
# print(your_list[1])
stops = ['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'o', 'hadn', 'herself', 'll', 'had', 'should', 'to', 'only', 'won', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'd', 'did', 'didn', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'hasn', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 're', 'does', 'above', 'between', 'mustn', 't', 'be', 'we', 'who', 'were', 'here', 'shouldn', 'hers', 'by', 'on', 'about', 'couldn', 'of', 'against', 's', 'isn', 'or', 'own', 'into', 'yourself', 'down', 'mightn', 'wasn', 'your', 'from', 'her', 'their', 'aren', 'there', 'been', 'whom', 'too', 'wouldn', 'themselves', 'weren', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'ma', 'these', 'up', 'will', 'below', 'ain', 'can', 'theirs', 'my', 'and', 've', 'then', 'is', 'am', 'it', 'doesn', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'yo', 'shan', 'needn', 'haven', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'm', 'yours', 'so', 'y', 'the', 'having', 'once']
print(stops)
sentences = []
for i in range(len(your_list)):
    words = your_list[i][6].lower().split();
    words = [word for word in words if word not in stops]
    sentence = models.doc2vec.LabeledSentence(words=words, tags = ["SENT_" + str(i)])
    sentences.append(sentence)

model = models.Doc2Vec(size = 300, window = 10, alpha=.025, min_alpha=.025, min_count=3, workers =11)
model.build_vocab(sentences)

for epoch in range(8):
    model.train(sentences, total_examples= model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate`
    model.min_alpha = model.alpha  # fix the learning rate, no decay

print('DONE WITH TRAINING')
search_phrase = ['what', 'code', 'analysis', 'tools', 'do', 'you', 'use', 'on', 'your', 'Java', 'projects']
term = model.infer_vector(search_phrase, alpha=0.025, min_alpha=0.025, steps=20)
#print(cosine_similarity(s1, model.docvecs['SENT_9']))
 # Print out = ~0.00795774
print(cosine_similarity(model.docvecs['SENT_45'], model.docvecs['SENT_375']))
max_similarity = 0
max = 0
for i in range(len(your_list)):
    print('hi')
    index = 'SENT_' + str(i)
    if cosine_similarity(model.docvecs[index], term) > max_similarity and  i != 45:
        max_similarity = cosine_similarity(model.docvecs[index], term)
        max = i

print(cosine_similarity(model.docvecs['SENT_45'], model.docvecs['SENT_374']))
print(cosine_similarity(model.docvecs['SENT_45'], model.docvecs['SENT_375']))
print(cosine_similarity(model.docvecs['SENT_45'], model.docvecs['SENT_376'])) #78 here means 79 on csv
print(your_list[max][6])
print(max)
model.save('questiondupemodel3')
