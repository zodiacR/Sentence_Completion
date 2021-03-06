#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : Word2Vec.py
import cPickle as pickle
import gensim
import numpy as np
import theano
from os import path
class Word2Vec(object):
    """
    Construct a representation matrix of vocabulary
    """
    def __init__(self, raw_path, vector_path, level="word", size=100):
        """
        Set basic environments
        """
        self.raw_path = raw_path
        self.vector_path = vector_path
        self.sentences = []
        self.size = size

        self.LoadVoca()

        # load output vocabulary
        if level == "char":
            with open("data/char_level/onehot_output.txt") as f:
                self.vocabulary = pickle.load(f)
        else:
            self.vocabulary = self.dict.index2word

        self.output_size = len(self.vocabulary)


    def LoadVoca(self):
        """
        Load vocabulary list to memory
        """
        # if vocabulary list does not exist, construct a
        # new one
        if not path.exists(self.vector_path):
            self.ReadRawData()
            self.Word2Vec()
        else:
            self.dict = gensim.models.Word2Vec.load_word2vec_format(self.vector_path,
                                                                    binary=False)

    def ReadRawData(self):
        """
        Read raw data from the given file
        """

        with open(self.raw_path) as fin:
            # preprocess raw data
            for line in fin:
                # remove spaces at the end of each line
                # and split the line into several words
                line = line.strip().split()

                # append a sentence to sentence list
                self.sentences.append(line)

    def Word2Vec(self):
        """
        translate sentences to vectors
        """
        self.dict = gensim.models.Word2Vec(sentences=self.sentences,
                                         size=self.size,
                                         window=5,
                                         min_count=3,
                                         workers=4)
        self.dict.save_word2vec_format(self.vector_path, binary=False)

    def Lookup(self, line):
        """
        Lookup vectors of a sentence
        """
        vectors = [self.dict[word] for word in line]

        return np.asarray(vectors, dtype=theano.config.floatX)

    def Word2Index(self, line):
        """
        translate words of line to indices
        """
        indices = []
        for word in line:
            indices.append(self.vocabulary.index(word))

        return np.asarray(indices, dtype="int32")
