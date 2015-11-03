#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : preprocess.py
import cPickle as pickle
from nltk import FreqDist
from os import path

_VOCABULARY_SIZE = 10003
_ONEHOT_SIZE = 603
RAW_PATH = "data/without_unk"
PROCESSED_PATH = "data/char_level"
vocab = []
sentences = []

# load raw corpus and form raw dictionary
print "Loading raw data...."
with open(path.join(RAW_PATH, "ptb.trn")) as f:
    for line in f:
        line = line.strip().split()
        words = [word for word in line]
        vocab.extend(words)
        sentences.append(words)

print "%d sentences loaded" % len(sentences)

# count occurrence of each word
word_freq = FreqDist(vocab)
print "%d unique words" % len(word_freq.items())

# input vocab
dictionary = word_freq.most_common(_VOCABULARY_SIZE-1)
expression = "The least frequent word is '%s', and it occurs %d times" 
print expression % (dictionary[-1][0], dictionary[-1][1])
vocab = [x[0] for x in dictionary]

# output vocab
dictionary = word_freq.most_common(_ONEHOT_SIZE-1)

expression = "The least frequent word for output is '%s', and it occurs %d times" 
print expression % (dictionary[-1][0], dictionary[-1][1])

onehot = [x[0] for x in dictionary]
onehot.insert(0, "UNK")

# store unknown words not in output vocab
unknown = list(set([x[0] for x in word_freq.items()]) - set(onehot))
with open(path.join(PROCESSED_PATH, "unknown.txt"), "w") as f:
        pickle.dump(unknown, f)

# replace unknown words with UNK in ptb.trn
with open(path.join(PROCESSED_PATH, "ptb.trn"), "w") as f:
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] not in vocab:
                sentences[i][j] = "UNK"
        f.write(" ".join(sentences[i]))
        f.write("\n")

# replace unknown words with UNK in ptb.tst
with open(path.join(RAW_PATH, "ptb.tst")) as fr:
    with open(path.join(PROCESSED_PATH, "ptb.tst"), "w") as fw:
        for line in fr:
            line = line.strip().split()
            for j in range(len(line)):
                if line[j] not in vocab:
                    line[j] = "UNK"
            fw.write(" ".join(line))
            fw.write("\n")

# replace unknown words with UNK in ptb.dev
with open(path.join(RAW_PATH, "ptb.dev")) as fr:
    with open(path.join(PROCESSED_PATH, "ptb.dev"), "w") as fw:
        for line in fr:
            line = line.strip().split()
            for j in range(len(line)):
                if line[j] not in vocab:
                    line[j] = "UNK"
            fw.write(" ".join(line))
            fw.write("\n")

# save output vocabulary to disk
with open(path.join(PROCESSED_PATH, "onehot_output.txt"), "w") as f:
    pickle.dump(onehot, f)
