#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : C2W.py
import numpy as np
import string
import theano

class C2W(object):
    """
    construct a vector to represent each character in a corpus
    """
    def __init__(self):
        """
        Set basic environments
        """
        characters = string.letters + string.digits + string.punctuation
        self.table = list(characters)
        self.size = len(self.table)

    def Lookup(self, characters):
        """
        Lookup vectors of a sentence
        """
        vectors  = []

        for c in characters:
            idx = self.table.index(c)
            vector = np.zeros(self.size, dtype=theano.config.floatX)
            vector[idx] = 1
            vectors.append(vector)

        return np.asarray(vectors)

    def Char2Index(self, characters):
        """
        translate words of line to indices
        """
        indices = []
        for char in characters:
            indices.append(self.table.index(char))

        return np.asarray(indices, dtype="int32")
