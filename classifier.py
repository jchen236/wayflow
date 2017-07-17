import sys
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

model = models.Doc2Vec.load('questiondupemodel4')
#term = model.docvecs['SENT_0']
question =sys.argv[1]
# search_phrase = ['to', 'reverse', 'linked', 'list', 'linkedlist']
# term = model.infer_vector(search_phrase, alpha=0.025, min_alpha=0.025, steps=20)

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


for i in range(len(your_list)):
    your_list[i][6] = re.sub('[^0-9a-zA-Z\']+', ' ', cleanhtml(your_list[i][6]).replace('\n', ''))


max_similarity = 0
max = 0

#print(model.most_similar(positive=['javascript', 'console'],  topn=50) )
tokens = str(question).split()

new_vector = model.infer_vector(tokens)
indices = model.docvecs.most_similar([new_vector], topn = 5)
res = [(i[0]) for i in indices]
# print(res)
returned_questions = []
returned_indices = []
returned_answers = []
for index, item in enumerate(res):
    res[index] = (int)(item.replace('SENT_', ''))
    returned_questions.append(your_list[res[index]][6])
    #returned_questions.append(res[index])
    returned_indices.append(res[index])
    question_id = your_list[res[index]][0]
print(returned_questions)
print(returned_indices)

# for i in range(len(res)):
#     print(your_list[res[i]][6])
