#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : RNN.py
import numpy as np
import theano
import theano.tensor as T


class RNN(object):
    """
    Recurrent neural network model
    """
    def __init__(self, input, n_in, n_hidden, n_out):
        """
        Initialise basic settings related to
        RNN
        """
        self.input = input
        self.activation = T.tanh
        self.softmax = T.nnet.softmax

        # recurrent weights as a shared variable
        # hidden to hidden weights
        W_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                                 low=-.01, high=.01),
                                                 dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name="W")

        # input to hidden layer weights
        W_in_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
                                                 low=-.01, high=.01),
                                                 dtype=theano.config.floatX)
        self.W_in = theano.shared(value=W_in_init, name="W_in")

        # hidden to output layer weights
        W_out_init = np.asarray(np.random.uniform(size=(n_out, n_hidden),
                                                 low=-.01, high=.01),
                                                 dtype=theano.config.floatX)
        self.W_out = theano.shared(value=W_out_init, name="W_out")

        # initial hidden layer
        h0_init = np.zeros(n_hidden, dtype=theano.config.floatX)
        self.h0 = theano.shared(value=h0_init, name="h0")

        # bias of hidden layers
        bh_init = np.zeros(n_hidden, dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name="bh")

        # bias of output layer
        by_init = np.zeros(n_out, dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init, name="by")


        self.params = [self.W, self.W_in, self.W_out,
                       self.h0, self.bh, self.by]

        # for every parameter, we main its last update
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        # recurrent function and linear output activation function
        def step(x_t, h_tm1):
            # from input layer and previous hidden layer to current
            # hidden layer
            h_t = self.activation(T.dot(x_t, self.W_in) +
                                  T.dot(h_tm1, self.W) + self.bh)
            # from hidden layer to output layer
            y_t = T.dot(h_t, self.W_out) + self.by

            return h_t, y_t

        # store each hidden layer and output layer
        [self.h, self.y_pred], updates = theano.scan(step,
                                                     sequences=self.input,
                                                     outputs_info=[self.h0,
                                                                   None])

        # square of L2 norm
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.W_in ** 2).sum()
        self.L2_sqr += (self.W_out ** 2).sum()

        # push through softmax, computing vector of class-membership
        # probabilities in symbolic form
        self.p_y_given_x = self.softmax(self.y_pred)

        # compute prediction as class whose probability is maximal
        self.y_out = T.argmax(self.p_y_given_x, axis=-1)
        self.loss = lambda y: self.nll_multiclass(y)


    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        # y.shape[0] is the number if rows in y, i.e.,
        # number of times steps in the sequence
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,..,n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP(T.arange[y.shape[0], y]) is a vector
        # v containing [LP[0, y[0]], LP[1, y[1]], LP[2, y[2]]], ...,
        # LP[n-1, y[n-1]] and T.mean(LP[T.arange(y.shape[0]), y]) is
        # the mean (across minibatch examples) of the elements is v,
        # i.e., the mean log-likelihood across the minibatch
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
