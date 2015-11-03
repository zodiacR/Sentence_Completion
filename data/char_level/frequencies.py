#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : frequencies.py
vocab_trn = set()
vocab_tst = set()
vocab_dev = set()

with open("./ptb.trn") as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            vocab_trn.add(word)
            #word = word.lower()
            #vocab_trn.setdefault(word, 0)
            #vocab_trn[word] += 1

#unk = vocab_trn["UNK"]
with open("./ptb.dev") as f:
    for line in f:
        words = line.split()
        for word in words:
            #word = word.lower()
            vocab_tst.add(word)

with open("./ptb.tst") as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            #word = word.lower()
            vocab_tst.add(word)

#frequencies = []
#for word, count in vocab_trn.iteritems():
    #frequencies.append((word, count))
    #frequencies.append(count)

#plt.hist(frequencies)
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.show()
#frequencies.sort(key=lambda word:word[1])

#print frequencies[1000:5000]
#print frequencies[-4900:]
#print len(vocab_trn.keys())

print vocab_tst-vocab_trn
print vocab_dev-vocab_trn

