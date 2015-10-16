#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : preprocess.py
from nltk import FreqDist
from os import path

RAW = "without_unk"
PROCESSED = "with_unk"
vocab = []
sentences = []

with open(path.join(RAW, "ptb.trn")) as f:
    for line in f:
        line = line.strip().split()
        words = [word for word in line]
        vocab.extend(words)
        sentences.append(words)

word_freq = FreqDist(vocab)
dictionary = word_freq.most_common(5003-1)
vocab = [x[0] for x in dictionary]

with open(path.join(PROCESSED, "ptb.trn"), "w") as f:
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] not in vocab:
                sentences[i][j] = "UNK"
        f.write(" ".join(sentences[i]))
        f.write("\n")

with open(path.join(RAW, "ptb.tst")) as fr:
    with open(path.join(PROCESSED, "ptb.tst"), "w") as fw:
        for line in fr:
            line = line.strip().split()
            for j in range(len(line)):
                if line[j] not in vocab:
                    line[j] = "UNK"
            fw.write(" ".join(line))
            fw.write("\n")

with open(path.join(RAW, "ptb.dev")) as fr:
    with open(path.join(PROCESSED, "ptb.dev"), "w") as fw:
        for line in fr:
            line = line.strip().split()
            for j in range(len(line)):
                if line[j] not in vocab:
                    line[j] = "UNK"
            fw.write(" ".join(line))
            fw.write("\n")
