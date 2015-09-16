#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : OneHot.py
import cPickle as pickle
import numpy as np
import theano
from os import path
class OneHot(object):
    """
    Construct a representation matrix of vocabulary
    """
    def __init__(self, raw_path, per_path):
        """
        Set basic environments
        """
        self.raw_path = raw_path
        self.per_path = per_path
        self.vocabulary = []

        self.LoadVoca()
        self.size = len(self.vocabulary)
        print len(self.vocabulary)


    def LoadVoca(self):
        """
        Load vocabulary list to memory
        """
        # if vocabulary list does not exist, construct a
        # new one
        if not path.exists(self.per_path):
            self.ReadRawData()
            self.DumpVoca()
        else:
            with open(self.per_path) as fin:
                self.vocabulary = pickle.load(fin)

    def ReadRawData(self):
        """
        Read raw data from the given file
        """
        vocabulary = set()

        with open(self.raw_path) as fin:
            # preprocess raw data
            for line in fin:
                # remove spaces at the end of each line
                # and split the line into several words
                words = line.strip().split()

                # to lower case
                #words = [word.lower() for word in line]

                # record words in vocabulary list
                words = set(words)
                vocabulary |= words

        self.vocabulary = list(vocabulary)

    def DumpVoca(self):
        """
        Dump vocabulary list to a file
        """
        with open(self.per_path, "wb") as fout:
            pickle.dump(self.vocabulary, fout)

    def Word2Vec(self, line):
        """
        translate words of a line to vectors
        """
        vectors = []
        for word in line:
            idx = self.vocabulary.index(word)
            vector = np.zeros(self.size, dtype=theano.config.floatX)
            vector[idx] = 1
            vectors.append(vector)

        return np.asarray(vectors)

    def Word2Index(self, line):
        """
        translate words of line to indices
        """
        indices = []
        for word in line:
            indices.append(self.vocabulary.index(word))

        return np.asarray(indices, dtype=np.int)
