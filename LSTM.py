#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : LSTM.py
import cPickle as pickle
from C2W import C2W
import logging
import numpy as np
from optparse import OptionParser
from random import sample
from theano import tensor as T
import theano

mode = theano.Mode(linker="cvm")

class LSTM(object):
    """
    Long short-time memory in recurrent neural network
    """
    def __init__(self, n_in, n_hidden,n_out,
                 learning_rate=0.01,
                 learning_rate_decay=1,
                 L2_reg=0.00, n_epochs=100):
        """
        Initialise basic variables in LSTM
        """
        self.softmax = T.nnet.softmax
        self.sigmoid = T.nnet.sigmoid
        self.tanh = T.tanh
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.L2_sqr = 0
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.L2_reg = float(L2_reg)
        self.epochs = int(n_epochs)

        self.ready()

    def uniform(self, n, m):
        """
        type n: int
        type m: int
        rtype : np.array

        return a n*m matrix with each value deriving from a uniform
        function
        """
        return np.asarray(np.random.uniform(size=(n, m),
                                                 low=-.01, high=.01),
                          dtype=theano.config.floatX)

    def LSTM(self, input):
        """
        forward LSTM
        """

        # lstm weights as shared values

        ##############
        # Input gate #
        ##############
        # input to input gate weights
        self.fWix = theano.shared(self.uniform(self.n_in, self.n_hidden),
                                 name="fWix")

        # previous hidden to input gate weights
        self.fWih = theano.shared(self.uniform(self.n_hidden, self.n_hidden),
                                 name="fWih")

        # bias of input gate
        fbi_init = np.zeros(self.n_hidden, dtype=theano.config.floatX)
        self.fbi = theano.shared(value=fbi_init, name="fbi")

        ###############
        # Forget gate #
        ###############
        # input to forget gate weights
        self.fWfx = theano.shared(self.uniform(self.n_in, self.n_hidden),
                                  name="fWfx")

        # previous hidden to forget gate weights
        self.fWfh = theano.shared(self.uniform(self.n_hidden, self.n_hidden),
                                 name="fWfh")

        # bias of forget gate
        fbf_init = np.zeros(self.n_hidden, dtype=theano.config.floatX)
        self.fbf = theano.shared(value=fbf_init, name="fbf")

        ################
        # Output gate  #
        ################
        # input to output gate weights
        self.fWox = theano.shared(self.uniform(self.n_in, self.n_hidden),
                                 name="fWox")

        # previous hidden to output gate weights
        self.fWoh = theano.shared(self.uniform(self.n_hidden, self.n_hidden),
                                  name="fWoh")

        # bias of output gate
        fbo_init = np.zeros(self.n_hidden, dtype=theano.config.floatX)
        self.fbo = theano.shared(value=fbo_init, name="fbo")

        ################
        # Cell gate  #
        ################
        # input to cell mem weights
        self.fWcx = theano.shared(self.uniform(self.n_in, self.n_hidden),
                                 name="fWcx")

        # previous hidden to cell mem weights
        self.fWch = theano.shared(self.uniform(self.n_hidden, self.n_hidden), name="fWch")

        # bias of cell mem
        fbc_init = np.zeros(self.n_hidden, dtype=theano.config.floatX)
        self.fbc = theano.shared(value=fbc_init, name="fbc")

        #####################
        #  hidden to output #
        #####################
        self.fWout = theano.shared(self.uniform(self.n_hidden, self.n_out), name="fWout")
        # bias of output
        fby_init = np.zeros(self.n_out, dtype=theano.config.floatX)
        self.fby = theano.shared(value=fby_init, name="fby")

        # initial hidden layer
        fh0_init = np.zeros(self.n_hidden, dtype=theano.config.floatX)
        self.fh0 = theano.shared(value=fh0_init, name="fh0")

        # initial cell mem
        fc0_init = np.zeros(self.n_hidden, dtype=theano.config.floatX)
        self.fc0 = theano.shared(value=fc0_init, name="fc0")

        self.fparams = [self.fWix, self.fWih, self.fbi,
                        self.fWfx, self.fWfh, self.fbf,
                        self.fWox, self.fWoh, self.fbo,
                        self.fWcx, self.fWch, self.fbc,
                        self.fWout, self.fby,
                        self.fh0, self.fc0
                         ]

        # square of L2 norm
        self.L2_sqr +=(self.fWix ** 2).sum()
        self.L2_sqr +=(self.fWih ** 2).sum()
        self.L2_sqr +=(self.fWfx ** 2).sum()
        self.L2_sqr +=(self.fWfh ** 2).sum()
        self.L2_sqr +=(self.fWox ** 2).sum()
        self.L2_sqr +=(self.fWoh ** 2).sum()
        self.L2_sqr +=(self.fWcx ** 2).sum()
        self.L2_sqr +=(self.fWch ** 2).sum()
        self.L2_sqr +=(self.fWout ** 2).sum()

        # lstm function
        def step(x_t, h_t, c_t):
            # input gate
            it = self.sigmoid(T.dot(x_t, self.fWix) + 
                              T.dot(h_t, self.fWih) + 
                              self.fbi)
            # forget gate
            ft = self.sigmoid(T.dot(x_t, self.fWfx) + 
                              T.dot(h_t, self.fWfh) + 
                              self.fbf)
            # output gate
            ot = self.sigmoid(T.dot(x_t, self.fWox) + 
                              T.dot(h_t, self.fWoh) + 
                              self.fbo)
            # cell mem
            ct = ft*c_t + it*self.tanh(
                                        T.dot(x_t, self.fWcx)+
                                        T.dot(h_t, self.fWch)+
                                        self.fbc
                                        )
            # hidden
            ht = ot * self.tanh(ct)

            # output
            y_t = T.dot(ht, self.fWout) + self.fby

            return ht, ct, y_t

        # store each cell mem and output
        [self.fh, self.fc, self.y_pred], updates = theano.scan(
                                                       step,
                                                       sequences=input,
                                                       outputs_info=[self.fh0, self.fc0, None])

        # push through softmax, computing vector of class-membership
        # probabilities in symbolic form
        self.p_y_given_x = self.softmax(self.y_pred)

        # compute prediction as class whose probability is maximal
        self.y_out = T.argsort(self.p_y_given_x, axis=-1)
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

    def ready(self):
        """
        Load all inputs and parameters to train RNN
        """
        # input sentence
        self.x = T.matrix(name="x", dtype=theano.config.floatX)
        #target
        #self.y = T.matrix(name="y", dtype=theano.config.floatX)
        self.y = T.vector(name="y", dtype="int32")

        self.LSTM(self.x)

    def fit(self, samples, X_train, Y_train, validation=10000):
        """
        Fit model
        Pass in X_test, Y_test to compute test error and report during
        training
        """
        n_train = len(X_train)

        #####################
        #   Build model     #
        #####################
        #index = T.lscalar("index")
        train_set_x = T.matrix()
        #train_set_y = T.matrix(dtype=theano.config.floatX)
        train_set_y = T.vector(dtype="int32")

        l_r = T.scalar("l_r", dtype=theano.config.floatX)
        cost = self.loss(self.y) + self.L2_reg * self.L2_sqr

        compute_train_error = theano.function(inputs=[train_set_x, train_set_y],
                                              outputs=self.loss(self.y),
                                              givens={
                                                  self.x: train_set_x,
                                                  self.y: train_set_y
                                                  },
                                              mode=mode)

        # compute gradient of cost with respect to
        # gradients on the weights using BPTT
        updates = []
        for param in self.fparams:
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

                    fmt = "epoch %i, seq %i/%i, train loss %f, lr: %f"
                    logging.debug(fmt % (epoch, idx+1, n_train,
                                         this_train_loss, 
                                         self.learning_rate))


            self.learning_rate *= self.learning_rate_decay

            if epoch % 10 == 0:
                filename = "lstm-100_%e-%d.npz" % (self.L2_reg ,epoch)

                np.savez(filename,
                        fWix=self.fWix.get_value(),
                        fWih=self.fWih.get_value(),
                        fbi=self.fbi.get_value(),
                        fWfx=self.fWfx.get_value(),
                        fWfh=self.fWfh.get_value(),
                        fbf=self.fbf.get_value(),
                        fWox=self.fWox.get_value(),
                        fWoh=self.fWoh.get_value(),
                        fbo=self.fbo.get_value(),
                        fWcx=self.fWcx.get_value(),
                        fWch=self.fWch.get_value(),
                        fbc=self.fbc.get_value(),
                        fWout=self.fWout.get_value(),
                        fby=self.fby.get_value(),
                        fh0=self.fh0.get_value(),
                        fc0=self.fc0.get_value())

def Train(lamb=1e-8):
    """
    Train a word character by character
    """
    UNK_PATH = "data/char_level/unknown.txt"

    c2w = C2W()
    with open("data/char_level/onehot_output.txt") as f:
        vocabulary = pickle.load(f)

    lstm = LSTM(c2w.size, len(vocabulary), c2w.size,
                learning_rate_decay=0.99, L2_reg=lamb)

    with open(UNK_PATH) as f:
        words = pickle.load(f)

    train_set = []
    target_set = []

    for word in words:
        word = list(word)
        word.insert(0,"^")
        word.append("$")
        train_set.append(c2w.Lookup(word[:-1]))
        target_set.append(c2w.Char2Index(word[1:]))

    lstm.fit(4000, train_set, target_set, validation=len(train_set))

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-l", "--lamb",
                      action="store",
                      dest="lamb",
                      type="float",
                      default=1e-8)
    options, args = parser.parse_args()
    logging.basicConfig(filename="char-%e.txt" % options.lamb,
                        level=logging.DEBUG)

    Train(options.lamb)
