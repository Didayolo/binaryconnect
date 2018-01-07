# -*- coding: utf-8 -*-

# Source : https://github.com/DingKe/nn_playground/tree/master/binarynet

from __future__ import absolute_import
import keras.backend as K
import random
import numpy as np

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)
    
    
def deterministic_binarization(W):
    """
    Perform a deterministic binarization (like numpy.sign but it returns integers)
        +1 if w >= 0
        -1 otherwise
    Inputs: W: the weights - list of ndarray
    Outputs: W: the binary weights - list of ndarray
    """
    binarization = lambda w: 1 if w >= 0 else -1
    return np.vectorize(binarization)(W)
    
    
def hard_sigmoid(x):
    return np.clip((x+1.)/2.,0,1)


def stbin(w):
    p = hard_sigmoid(w)
    if p > random.uniform(0, 1):
        return 1
    else:
        return -1


def stochastic_binarization(W):
    """
    Perform a deterministic binarization
        +1 with probability p = hard_sigmoid(w)
        -1 with probability 1 - p
    Inputs: W: the weights - list of ndarray
    Outputs: W: the binary weights - list of ndarray
    """
    return np.vectorize(stbin)(W)


def binary_sigmoid(x):
    '''Binary hard sigmoid for training binarized neural network.

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    '''Binary hard sigmoid for training binarized neural network.
     The neurons' activations binarization function
     It behaves like the sign function during forward propagation
     And like:
        hard_tanh(x) = 2 * _hard_sigmoid(x) - 1 
        clear gradient when |x| > 1 during back propagation

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return 2 * round_through(_hard_sigmoid(x)) - 1

def binarize(W, H=1, deterministic=False):
    '''The weights' binarization function, 

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    
    return H * binary_tanh(W / H)
    """
    if deterministic:
        Wb = deterministic_binarization(W)
        
    else:
        # [-H, H] -> -H or H
        #Wb = H * binary_tanh(W / H)
        stochastic_binarization(W)
        
    return Wb
    """

def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False, deterministic=False):
    Wb = binarize(W, H, deterministic=determinstic)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb
