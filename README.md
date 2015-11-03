#Language Model 

## Programme description
This programme uses RNN to simulate language model, and uses learnt parameters
to conduct sentence completion

System Requirements: 
- Python2.7, python-pip, php

## Setting up basic environment 
``` bash
# Mac OS
pip install Theano numpy gensim

# Ubuntu & Debian
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install Theano gensim

# CentOS 6
sudo yum install python-devel python-nose python-setuptools gcc gcc-gfortran gcc-c++ blas-devel lapack-devel atlas-devel
sudo easy_install pip
sudo pip install numpy==1.6.1
sudo pip install scipy==0.10.1
sudo pip install Theano gensim
```

## Programme Script
- language_model.py: main programme. training and output learnt parameters 
- prediction.py: online prediction programme. 
- prediction_local.py: command-line prediction programme. 
- C2W.py: charaters to vectors
- LSTM.py: charater-based language model
- RNN.py: recurrent neural network based language model
- Word2Vec.py: words to vectors
- preprocess.py: preprocess original corpus, including tokenization, replacement of least frequent words


## Run scripts
1. Train corpus: python language_model.py (takes 3-4 days without GPU acceleration.
                                           So I provied well-trained parameters for completion)
2. Local completion: python prediction_local.py
   - use sentences in prediction/ptb.dev (note: input should be without '<s>'.
     For example, "AMR slid $ UNK ,", not "<s> AMR slid $ UNK ,")
   - you can also type your own sentences (input should follow format in prediction/ptb.dev)
3. Online completion: please setup php environment and copy all files
   under prediction/web to php home directory. Then `python prediction.py`.
   - follow content described in 2 to enter partial sentences for completion
