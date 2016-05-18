"""
Source Code by Francisco Javier Arceo
Assistance from Allison Fenichel, Michael Bisaha
Columbia University


This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
[4] https://github.com/MatthieuCourbariaux/BinaryConnect
[5] https://github.com/hantek/BinaryConnect
"""
import numpy 

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt
from bc_utils import shared_dataset, load_svhn, load_mnist, load_cifar10
from bc_nn import myMLP, HiddenLayer, SVMLayer, LogisticRegression, train_nn

def mlp(initial_learning_rate=0.3, final_learning_rate=0.01, 
             L1_reg=0.00, L2_reg=0.000, n_epochs=100,
             batch_size=200, n_hidden=1024, n_hiddenLayers=3,
             verbose=False, stochastic=False, binary=True, 
             which_data='svhn', seedval=12345, outputlayer='Logistic', early_stopping=False):
    """
    Wrapper function for training and testing MLP

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    which_data = which_data.lower()
    outputlayer = outputlayer.lower()

    if which_data not in ('mnist','cifar10','svhn'):
        return 'Need to choose corrrect dataset either "mnist", "svhn", or "cifar10"'

    print ('Loading %s data...' % which_data.upper())
    if which_data=='mnist':
        datasets = load_mnist(outputlayer)
        nins = 28*28*1

    elif which_data=='svhn':
        datasets = load_svhn(outputlayer)
        nins = 32*32*3

    elif which_data=='cifar10':
        datasets = load_cifar10(outputlayer)
        nins = 32*32*3
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('Building the model...')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    if outputlayer=='svm':
        y = T.imatrix('y')
    if outputlayer=='logistic':
        y = T.ivector('y')

    learning_rate_decay = (float(final_learning_rate)/float(initial_learning_rate))**(1./n_epochs)
    learning_rate = theano.shared(numpy.asarray(initial_learning_rate, dtype=theano.config.floatX))
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})
   
    rng = numpy.random.RandomState(seedval)
    
    classifier = myMLP(rng=rng, 
                       input=x, 
                       n_in=nins, 
                       n_hidden=n_hidden, 
                       n_out=10, 
                       n_hiddenLayers=n_hiddenLayers, 
                       stochastic=stochastic,
                       binary=binary,
                       activation=T.nnet.relu,
                       outputlayer=outputlayer)
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.cost(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    train_model_perf = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]

    # This is the forward progagation
    if binary:
        W0=theano.shared(classifier.W0, name='W0', borrow=True)    
        updates=[]
        for param in classifier.params:
            if param.name not in ('beta','gamma','b', 'Wb', 'W'):
                continue

            u = param - learning_rate * T.grad(cost, param)
            if param.name=='W':
                u = T.clip(u, -1., 1.)
            updates.append((param, u))

    # if binary:
    #     W0=theano.shared(classifier.W0, name='W0', borrow=True)    
    #     updates=[]
    #     for i in range(classifier.len_params):
    #         for j in range(n_hiddenLayers+1):
    #             p=classifier.params[j*(classifier.len_params)+i]

    #             if p.name in ('beta','gamma','b', 'Wb'):
    #                 u = p - learning_rate * T.grad(cost, p)
    #                 updates.append((p, u))

    #             if p.name=='W':
    #                 n_Wb = classifier.params[j*(classifier.len_params)+i+2]
    #                 u = p - learning_rate * T.grad(cost, n_Wb)
    #                 u = T.clip(u, -W0, W0)
    #                 updates.append((p, u))
    if not binary:
        gparams = T.grad(cost, classifier.params)
        updates = [(p, p - learning_rate * gp) for p, gp in zip(classifier.params, gparams)]

    # compiling a Theano function `train_model` that returns the cost, but
    # at the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(decay_learning_rate, train_model, train_model_perf, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, learning_rate, which_data,
        stochastic, binary, outputlayer, verbose, early_stopping)
    
    # print('W =', classifier.hiddenLayers[0].W.eval())
    # print('Wb =', classifier.hiddenLayers[0].Wb.eval())
    return classifier

if __name__ == "__main__":
    mlp(initial_learning_rate=0.01, final_learning_rate=0.001,
             L1_reg=0.000, L2_reg=0.000, n_epochs=5, batch_size=200,
             n_hidden=1024, n_hiddenLayers=3, verbose=True, 
             stochastic=True, binary=True, which_data='mnist', 
             seedval=420, outputlayer='svm', early_stopping=True)