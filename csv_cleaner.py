import re
import numpy as np
from numpy import genfromtxt
import csv


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

with open('new.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

print(your_list)[1]
print("AAAAAAAA\n")
for i in range(len(your_list)):
    your_list[i][6] = cleanhtml(your_list[i][6]).replace('\n', '')
print(your_list[1])
