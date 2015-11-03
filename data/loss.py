#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : loss.py
#from gensim.models import Word2Vec
#import numpy as np
#from random import sample

#model = Word2Vec.load_word2vec_format("./vectors.txt", binary=False)

#words = model.index2word
#print words[:10]
#print list(words)[:10]
#vectors1 = []
#vectors2 = []
words = {}

with open("./ptb.trn") as fin:
    for line in fin:
        line = line.strip().split()
        for word in line:
            words.setdefault(word, 0)
            words[word] += 1

words = list(words)
words.sort(key=lambda d:d)
#v1 = sample(words, 20)
#v2 = sample(words, 20)
#print v1
#print v2

#for i, j in zip(v1, v2):
    #vectors1.append(model[i])
    #vectors2.append(model[j])

#vectors1 = np.asarray(vectors1)
#vectors2 = np.asarray(vectors2)

#print ((model[words[2]]-model[words[10]])**2).mean()
