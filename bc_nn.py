"""
Source Code by Allison Fenichel, Francisco Arceo, Michael Bisaha
ECBM E6040, Spring 2016, Columbia University


This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
[4] https://github.com/MatthieuCourbariaux/BinaryConnect
[5] https://github.com/hantek/BinaryConnect
"""

from __future__ import print_function

import timeit
import inspect
import sys
import numpy
from scipy import ndimage

import os
import csv
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
# Dis slow
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.bn import batch_normalization

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, rng, n_in, n_out, stochastic=False, binary=True):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        # self.W = theano.shared(
        #     value=numpy.zeros(
        #         (n_in, n_out),
        #         dtype=theano.config.floatX
        #     ),
        #     name='W',
        #     borrow=True
        # )
        W_values = numpy.asarray(
            rng.uniform(
                low= -1.,#numpy.sqrt(6. / (n_in + n_out)),
                high= 1.,#numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.gamma = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), name='gamma')
        self.beta = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX), name='beta')
        
        self.high = 2. #numpy.float32(numpy.sqrt(6. / (n_in + n_out)))
        self.W0 = numpy.float32(self.high/2)
        #srng = RandomStreams(seed=420) 
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(420)
        
        def hard_sigma(w):
            return T.clip((w+1.)/2,0,1)

        if binary:
            if stochastic:
                Wb = hard_sigma(self.W/self.W0)
                # using numpy was insanely slow and it caused issues with having to evaluate the function
                #Wb = T.cast(numpy.random.binomial(n=1, p=Wb, size=(n_in, n_out)),  theano.config.floatX)
                Wb = srng.binomial(n=1, p=Wb, size=(n_in, n_out) )         # This works much better

            else:
                # T.ge is greater than or equal to
                #Wb = T.ge(Wb, 0)
                Wb = T.ge(self.W, 0)
                #Wb = T.round(Wb)

            Wb = T.switch(Wb, self.W0, -self.W0)
            self.Wb = Wb

            # The code below was way slower
            #Wb = T.cast(T.switch(Wb,self.W0, -self.W0), dtype=theano.config.floatX)
            #Wb = theano.shared(Wb.eval(), name='Wb', borrow=True)

        else:
            self.Wb = self.W
        
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        
        self.linear=T.dot(input, self.Wb) + self.b

        bn_output = batch_normalization(inputs = self.linear,
                    gamma = self.gamma, beta = self.beta, mean = self.linear.mean((0,), keepdims=True),
                    std = T.ones_like(self.linear.var((0,), keepdims = True)), mode='high_mem')
        
        self.p_y_given_x = T.nnet.softmax(bn_output)
                          
                          
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        if binary:
            self.params = [self.W, self.Wb, self.gamma, self.beta, self.b]

        elif not binary:
            self.params = [self.Wb, self.gamma, self.beta, self.b]

        self.len_params = len(self.params)
        
        self.n_in=n_in
        
        # keep track of model input
        self.input = input

    def cost(self, y): # The cost here is the negative_log_likelihood
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class SVMLayer(object):
    def __init__(self, input, rng, n_in, n_out, stochastic=False, binary=True):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.high = 2.#numpy.float32(numpy.sqrt(6. / (float(n_in) + float(n_out) )))
        self.W0 = numpy.float32(self.high/2.)        
        #self.W = theano.shared(value=numpy.zeros((n_in, n_out),
        #                         dtype=theano.config.floatX),
        #                         name='W', borrow=True)
        #self.high = numpy.float32(2.)
        #self.W0 = numpy.float32(self.high/2.)
        W_values = numpy.asarray(
            rng.uniform(
                low= -1.,#numpy.sqrt(6. / (n_in + n_out)),
                high= 1.,#numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        #srng = RandomStreams(seed=420) 
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(420)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                dtype=theano.config.floatX),
                                name='b', borrow=True)

        self.gamma = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), name='gamma')
        self.beta = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX), name='beta')
        
        def hard_sigma(w):
            p=T.clip((w+1)/2,0,1)
            return p

        if binary:
            Wb = hard_sigma(self.W/self.W0)
            if stochastic:
                #Wb = T.cast(numpy.random.binomial(n=1, p=T.ge(Wb), size=(n_in, n_out)),  theano.config.floatX)
                Wb = srng.binomial(n=1, p=Wb, size=(n_in, n_out) )

            else:
                Wb = T.round(Wb)

            # Leave below alone
            Wb = T.switch(Wb,self.W0, -self.W0)
            #Wb = T.cast(T.switch(Wb,self.W0, -self.W0), dtype=theano.config.floatX)
            #Wb = theano.shared(Wb.eval(), name='Wb', borrow=True)
            self.Wb = Wb

        else:
            self.Wb = self.W

        # parameters of the model
        self.linear = T.dot(input, self.Wb) + self.b

        bn_output = batch_normalization(inputs = self.linear,
                    gamma = self.gamma, beta = self.beta, mean = self.linear.mean((0,), keepdims=True),
                    std = T.ones_like(self.linear.var((0,), keepdims = True)), mode='high_mem')

        self.linear_output = bn_output
        self.y_pred = T.argmax(bn_output, axis=1)

        if binary:
            self.params = [self.W, self.Wb, self.gamma, self.beta, self.b]

        elif binary==False:
            self.params = [self.Wb, self.gamma, self.beta, self.b]

        self.len_params = len(self.params)
        self.n_in=n_in
        # keep track of model input
        self.input = input

    # svm_cost/hinge loss  -- using this loss function based off of Matthieu Courbariaux's loss function
        # There are other versions of hinge multiclass loss functions though.
    def cost(self, y): 
        return T.mean(T.sqr(T.maximum(0., 1.-y*self.linear_output )) )

    def errors(self, y):
        return T.mean(T.neq(T.argmax(y,axis=1), self.y_pred))


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, stochastic=False, binary=True, W=None, b=None,
                 activation=T.nnet.relu):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        #srng = RandomStreams(seed=420) 
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(420)
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low= -1.,#numpy.sqrt(6. / (n_in + n_out)),
                    high= 1.,#numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
           
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

       
        self.high = numpy.float32(2.)
        self.W0 = numpy.float32(self.high/2)

        #self.high = numpy.float32(numpy.sqrt(6. / (float(n_in) + float(n_out) )))
        #self.W0 = numpy.float32(self.high/2)
   
        def hard_sigma(w):
            p=T.clip((w+1)/2,0,1)
            return p

        if binary:
            if stochastic:
                Wb = hard_sigma(W/self.W0)
                #Wb = T.cast(numpy.random.binomial(n=1, p=Wb, size=(n_in, n_out)),  theano.config.floatX)
                Wb = srng.binomial(n=1, p=Wb, size=(n_in, n_out) )

            else:
                #Wb = T.ge(Wb, 0)
                Wb = T.ge(W, 0)

            Wb = T.switch(Wb, self.W0, -self.W0)
            self.Wb = Wb

        else:
            self.Wb = W
            
        self.W = W
        self.b = b
        self.n_in=n_in
         
        self.gamma = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), name='gamma')
        self.beta = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX), name='beta')

        lin_output = T.dot(input, self.Wb) + self.b
        
        bn_output = batch_normalization(inputs = lin_output,
            gamma = self.gamma, beta = self.beta, mean = lin_output.mean((0,), keepdims=True),
            std = lin_output.std((0,), keepdims = True),
                        mode='low_mem')
        
        self.output = (
            bn_output if activation is None
            else activation(bn_output)
        )
            
        # parameters of the model
        if binary:
            self.params = [self.W, self.Wb, self.gamma, self.beta, self.b]

        elif binary==False:
            self.params = [self.Wb, self.gamma, self.beta, self.b]
        
        
class myMLP(object):
    """Multi-Layefr Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hiddenLayers, 
                    stochastic, binary, activation, outputlayer='Logistic'):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int or list of ints
        :param n_hidden: number of hidden units. If a list, it specifies the
        number of units in each hidden layers, and its length should equal to
        n_hiddenLayers.

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        :type n_hiddenLayers: int
        :param n_hiddenLayers: number of hidden layers
        """

        # If n_hidden is a list (or tuple), check its length is equal to the
        # number of hidden layers. If n_hidden is a scalar, we set up every
        # hidden layers with same number of units.
        if hasattr(n_hidden, '__iter__'):
            assert(len(n_hidden) == n_hiddenLayers)
        else:
            n_hidden = (n_hidden,)*n_hiddenLayers

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function.
        self.hiddenLayers = []
        for i in xrange(n_hiddenLayers):
            h_input = input if i == 0 else self.hiddenLayers[i-1].output
            h_in = n_in if i == 0 else n_hidden[i-1]
            self.hiddenLayers.append(
                HiddenLayer(
                    rng=rng,
                    input=h_input,
                    n_in=h_in,
                    n_out=n_hidden[i],
                    activation=T.nnet.relu,
                    stochastic=stochastic,
                    binary=binary
            )
        )

        if outputlayer=='logistic':
            outputRegressionFunction = LogisticRegression
            if binary:
                if stochastic:
                    print('Using Stochastic Binary Connect with Logistic Output Layer')
                else:
                    print('Using Deterministic Binary Connect with Logistic Output Layer')
            else:
                print("Using Logistic regression")

        if outputlayer=='svm':
            outputRegressionFunction = SVMLayer
            if binary:
                if stochastic:
                    print('Using Stochastic Binary Connect with Support Vector Machine Output Layer')
                else:
                    print('Using Deterministic Binary Connect with Support Vector Machine Output Layer')
            else:
                print("Using Support Vector Machine")

        self.OutputLayer = outputRegressionFunction(
            rng=rng,
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden[-1],
            n_out=n_out,
            stochastic=stochastic,
            binary=binary
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + abs(self.OutputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
            + (self.OutputLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.cost = (
            self.OutputLayer.cost
        )
        # same holds for the function computing the number of errors
        self.errors = self.OutputLayer.errors
        self.y_pred = self.OutputLayer.y_pred
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        
        self.len_params = self.OutputLayer.len_params
        self.params = sum([x.params for x in self.hiddenLayers], []) + self.OutputLayer.params
        if binary:
            self.W0=self.OutputLayer.W0
        # keep track of model input
        self.input = input

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),
        pool_ignore_border=True, stochastic=False, binary=True):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)


        ## adding code from Ali's work here
        
        self.high = numpy.float32(numpy.sqrt(6. / (fan_in + fan_out)))
        self.W0 = numpy.float32(self.high/2)
        
        def hard_sigma(w):
                p=T.clip((w+1)/2,0,1)
                return p
            
        if stochastic:
            p = hard_sigma(self.W/self.W0)
            p_mask = T.cast(numpy.random.binomial(n=1, p=p.eval(), size=filter_shape), theano.config.floatX)
            Wb = T.switch(p_mask,self.W0,-self.W0).eval()
        else:        
            Wb = T.switch(T.ge(self.W.get_value(),0),self.W0,-self.W0).eval()
       
        if binary:
            Wb = theano.shared(Wb, name='Wb', borrow=True)
            self.Wb = Wb
        else:
            self.Wb=self.W

        self.gamma = theano.shared(value = numpy.ones((image_shape[0], filter_shape[0], (image_shape[3]-2)/poolsize[0], 
                                                       (image_shape[3]-2)/poolsize[0]), dtype=theano.config.floatX), name='gamma')
        self.beta = theano.shared(value = numpy.zeros((image_shape[0], filter_shape[0], (image_shape[3]-2)/poolsize[0], 
                                                       (image_shape[3]-2)/poolsize[0]), dtype=theano.config.floatX), name='beta')
       
        # convolve input feature maps with filters
        conv_out = conv2d(
           input=input,
           filters=self.Wb,
           filter_shape=filter_shape,
           image_shape=image_shape
       )
        
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=pool_ignore_border
        )

        bn_output = batch_normalization(inputs = pooled_out,
                   gamma = self.gamma, beta = self.beta, mean = pooled_out.mean((0,2,3), keepdims=True),
                   std = pooled_out.var((0,2,3), keepdims = True), mode='high_mem')

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(bn_output + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        if binary:
            self.params = [self.W, self.Wb, self.gamma, self.beta, self.b]
        elif not binary:
            self.params = [self.Wb, self.gamma, self.beta, self.b]

        self.len_params = len(self.params)
        # keep track of model input
        self.input = input

def train_nn(decay_learning_rate, train_model, train_model_perf, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs, learning_rate, which_data,
            stochastic, binary, outputlayer, verbose, early_stopping):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
                         
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                training_losses = [train_model_perf(i) for i in range(n_valid_batches)]
                this_training_loss = numpy.mean(training_losses)
                
                test_losses = [ test_model(i) for i in range(n_test_batches) ]
                test_score = numpy.mean(test_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, training error %f %%, validation error %f %%, test error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_training_loss * 100.,
                         this_validation_loss * 100.,
                         test_score* 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_score = numpy.mean(test_losses)

                    # if verbose:
                    #     print(('     epoch %i, minibatch %i/%i, test error of '
                    #            'best model %f %%'
                    #           'learning rate %f') %
                    #           (epoch, minibatch_index + 1,
                    #            n_train_batches,
                    #            test_score * 100.,
                    #           learning_rate.get_value()))

            if (patience <= iter) & early_stopping:
                done_looping = True
                break
        
        filename=which_data
        if binary and stochastic:
            filename=filename+'_bin_stochastic_'
        elif binary and not stochastic:
            filename=filename+'_bin_deterministic_'

        if outputlayer=='logistic':
            filename=filename+'LRout'

        elif outputlayer=='svm':
            filename=filename+'SVMout'
        
        filename=filename+'_perf.csv'

        if not os.path.exists('./output'):
            os.mkdir('./output')
            
        if epoch==1:
            with open('./output/'+filename, 'wb') as f:
                csv_writer=csv.writer(f)
                csv_writer.writerow(['epoch', 'learning_rate', 'stochastic', 'binary', 'which_data', 'outputlayer', 'this_training_loss', 'this_validation_loss', 'test_score'])
                csv_writer.writerow([epoch, learning_rate.get_value(), stochastic, binary, which_data, outputlayer, this_training_loss, this_validation_loss, test_score])
        else:
            with open('./output/'+filename, 'ab') as f:
                csv_writer=csv.writer(f)
                csv_writer.writerow([epoch, learning_rate.get_value(), stochastic, binary, which_data, outputlayer,  this_training_loss, this_validation_loss, test_score])

        new_learning_rate = decay_learning_rate()

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
