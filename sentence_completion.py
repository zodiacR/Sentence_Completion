#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : sentence_completion.py
import logging
import numpy as np
from optparse import OptionParser
from random import sample
#import sys
import theano
import theano.tensor as T
from Word2Vec import Word2Vec
from RNN import RNN


mode = theano.Mode(linker="cvm")

class SentenceCompletion(object):
    """
    Read raw data from 
    """
    def __init__(self, n_in, n_hidden, n_out,
                 learning_rate=0.01,
                 learning_rate_decay=1,
                 L2_reg=0.00, n_epochs=100):
        """
        Initialise basic variables
        """
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.L2_reg = float(L2_reg)
        self.epochs = int(n_epochs)

        self.ready()

    def ready(self):
        """
        Load all inputs and parameters to train RNN
        """
        # input sentence
        self.x = T.matrix(name="x", dtype=theano.config.floatX)
        #target
        #self.y = T.matrix(name="y", dtype=theano.config.floatX)
        self.y = T.vector(name="y", dtype="int32")

        # initial hidden state of the RNN
        self.h0 = T.vector()
        #learning rate
        self.lr = T.scalar()

        self.rnn =RNN(input=self.x,
                      n_in=self.n_in,
                      n_hidden=self.n_hidden,
                      n_out=self.n_out)

        ## probabilities
        #self.predict_proba = theano.function(inputs=[self.x,],
                                             #outputs=self.rnn.p_y_given_x,
                                             #mode=mode)
        # index with the highest probability
        self.predict = theano.function(inputs=[self.x,],
                                       outputs=self.rnn.y_out[-1][-5:],
                                       mode=mode)

    def shared_dataset(self, data_xy):
        """
        Load the dataset into shared variables
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX))

        return shared_x, T.cast(shared_y, "int32")

    def fit(self, word2vec, vocab,samples, X_train, Y_train, X_test=None, Y_test=None,
            validation=10000):
        """
        Fit model
        Pass in X_test, Y_test to compute test error and report during
        training
        """
        #train_set_x, train_set_y = self.shared_dataset((X_train,
                                                        #Y_train))
        #n_train = train_set_x.get_value(borrow=True).shape[0]
        n_train = len(X_train)

        #####################
        #   Build model     #
        #####################
        #index = T.lscalar("index")
        train_set_x = T.matrix()
        #train_set_y = T.matrix(dtype=theano.config.floatX)
        train_set_y = T.vector(dtype="int32")

        l_r = T.scalar("l_r", dtype=theano.config.floatX)
        cost = self.rnn.loss(self.y) + self.L2_reg * self.rnn.L2_sqr

        compute_train_error = theano.function(inputs=[train_set_x, train_set_y],
                                              outputs=self.rnn.loss(self.y),
                                              givens={
                                                  self.x: train_set_x,
                                                  self.y: train_set_y
                                                  },
                                              mode=mode)
        # test config
        n_test = len(X_test)
        test_set_x = T.matrix()
        test_set_y = T.vector(dtype="int32")
        compute_test_error = theano.function(inputs=[test_set_x, test_set_y],
                                             outputs=self.rnn.loss(self.y),
                                             givens={
                                                self.x: test_set_x,
                                                self.y: test_set_y
                                                 },
                                             mode=mode)

        # compute gradient of cost with respect to theta = (W, W_in,
        # W_out, h0, bh, by)
        # gradients on the weights using BPTT
        updates = []
        for param in self.rnn.params:
            gparam = T.grad(cost, param)
            #gparams.append(gparam)
            updates.append((param, param - l_r * gparam))

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameters of the
        # model based on the rules defined in `updates`
        train_model = theano.function(inputs=[train_set_x, train_set_y, l_r],
                                      outputs=cost,
                                      updates=updates,
                                      givens={
                                          self.x: train_set_x,
                                          self.y: train_set_y
                                          },
                                      mode=mode)
        ##############
        # Train model#
        ##############

        epoch = 0

        while (epoch < self.epochs):
            epoch += 1

            for idx in xrange(n_train):
                train_model(X_train[idx], Y_train[idx],
                            self.learning_rate)


                # validate learnt weights on training set
                iter = (epoch-1) * n_train + idx + 1

                if iter % validation == 0:
                    train_losses = [compute_train_error(X_train[i], Y_train[i])
                                    for i in sample(xrange(n_train), samples)]
                    this_train_loss = np.mean(train_losses)

                    test_losses = [compute_test_error(X_test[i], Y_test[i])
                                    for i in xrange(n_test)]
                    this_test_loss = np.mean(test_losses)

                    fmt = "epoch %i, seq %i/%i, train loss %f, test loss %f, lr: %f"
                    logging.debug(fmt % (epoch, idx+1, n_train,
                                         this_train_loss, this_test_loss,
                                         self.learning_rate))


            self.learning_rate *= self.learning_rate_decay

            if epoch % 10 == 0:
                filename = "rnn-100_%e-%d.npz" % (self.L2_reg ,epoch)

                np.savez(filename,
                         W=self.rnn.W.get_value(),
                         W_in=self.rnn.W_in.get_value(),
                         W_out=self.rnn.W_out.get_value(),
                         h0=self.rnn.h0.get_value(),
                         bh=self.rnn.bh.get_value(),
                         by=self.rnn.by.get_value())

                # pred data
                pred_path = "data/ptb.dev"
                with open(pred_path) as fin:
                    for i, line in enumerate(fin):
                        if i > 10:
                            break
                        line = line.strip().split()
                        # pred_set.append(word2vec.Lookup(line[:-1]))

                        result = self.predict(word2vec.Lookup(line[:-1]))

                        #loss = 1
                        #output = ""

                        #for word in word2vec.vocabulary:
                            #diff = ((result-word2vec.dict[word])**2).mean()
                            #if diff < loss:
                                #output = word
                                #loss = diff

                        logging.debug(" ".join(line[:-1]))
                        for j in result:
                            logging.debug(vocab[j])

def Completion(n_hidden, n_epochs=100,lamb=1e-8):
    """
    load raw data from a file and train them, finally
    complete incomplete sentences
    """
    # initialise onehot class
    raw_path = "data/ptb.trn"
    per_path = "data/vectors.txt"
    word2vec = Word2Vec(raw_path,
                         per_path)
    # units of layers
    n_in = word2vec.size
    n_hidden = 100
    n_out = word2vec.output_size

    # training data
    train_set = []
    target_set = []

    with open(raw_path) as fin:
        for i, line in enumerate(fin):
            #if i > 10:
                #break
            #line = [word.lower() for word in line.strip().split()]
            line = line.strip().split()
            #vectors = onehot.Word2Vec(line)
            train_set.append(word2vec.Lookup(line[:-1]))
            target_set.append(word2vec.Word2Index(line[1:]))

    # test data
    test_set = []
    test_actual = []
    test_path = "data/ptb.tst"
    with open(test_path) as fin:
        for line in fin:
            line = line.strip().split()
            #line = [word.lower() for word in line.strip().split()]
            test_set.append(word2vec.Lookup(line[:-1]))
            test_actual.append(word2vec.Word2Index(line[1:]))
    
    # construct a model for training
    model = SentenceCompletion(n_in, n_hidden, n_out,
                            learning_rate_decay=0.999,
                            L2_reg=lamb,
                            n_epochs=n_epochs)
    # train and test data
    model.fit(word2vec, word2vec.vocabulary, 4000,
              train_set, target_set,
              test_set, test_actual,
              validation=len(train_set)) 

if __name__ == "__main__":
    # parse command line options and arguments
    parser = OptionParser()
    parser.add_option("--hidden",
                      action="store",
                      dest="n_hidden",
                      type="int",
                      default=100)
    parser.add_option("-l", "--lamb",
                      action="store",
                      dest="lamb",
                      type="float",
                      default=1e-7)
    options, args = parser.parse_args()
    logging.basicConfig(filename="prediction_100_%e.txt" % options.lamb,
                        level=logging.DEBUG)

    # pass basic settings
    Completion(options.n_hidden, lamb=options.lamb, n_epochs=100)

