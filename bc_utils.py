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
import os
import sys
import numpy
import scipy.io

import theano
import theano.tensor as T
from keras.datasets import mnist
from keras.datasets import cifar10

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def shared_svm(data_xy, borrow=True):
    # For an SVM output layer we have to make the outputs coded into a +1/-1 matrix 
    # where each row corresponds to the individual label for each image and the columns represents 
    # whether it belongs to the class
    data_x, data_y = data_xy

    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

    n_classes = len(numpy.unique(data_y)) 
    y1 = -1 * numpy.ones((data_y.shape[0], n_classes))  # Making all -1

    y1[numpy.arange(data_y.shape[0]), data_y] = 1       # Placing 1 for the correct class

    shared_y1 = theano.shared(numpy.asarray(y1, dtype=theano.config.floatX), borrow=borrow)

    return shared_x, T.cast(shared_y1,  'int32')

def load_svhn(outputlayer='svm', theano_shared=True):
    ''' Loads the SVHN dataset, here we leverage the code from the homework and download it if it's not installed

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    # Download the SVHN dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'http://ufldl.stanford.edu/housenumbers/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    train_dataset = check_dataset('train_32x32.mat')
    test_dataset = check_dataset('test_32x32.mat')

    # Load the dataset
    train_set = scipy.io.loadmat(train_dataset)
    test_set = scipy.io.loadmat(test_dataset)

    # Convert data format
    def convert_data_format(data):
        X = data['X'].transpose(2,0,1,3)
        X = X.reshape((numpy.prod(X.shape[:-1]), X.shape[-1]),order='C').T / 255.
        y = data['y'].flatten()
        y[y == 10] = 0
        return (X,y)

    train_set = convert_data_format(train_set)
    test_set = convert_data_format(test_set)

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.


    if outputlayer=='svm':
        test_set_x, test_set_y = shared_svm(test_set)
        valid_set_x, valid_set_y = shared_svm(valid_set)
        train_set_x, train_set_y = shared_svm(train_set)

    if outputlayer=='logistic':
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def load_mnist(outputlayer='svm', theano_shared=True):
    ''' Loads the MNIST dataset directly from the Keras library
        I originally tried to load this from Lasagne, but that proved to be very difficult, 
                same for SVHN and CIFAR10

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    ntrn = len(X_train)
    ntst = len(X_test)

    train_set = X_train[0:ntrn,:,:].reshape((ntrn, 28*28*1)).astype('float32'), y_train[0:ntrn,]
    test_set =  X_test[:,:,:].reshape((ntst, 28*28*1)).astype('float32'), y_test
    
    # Downsample the training dataset if specified
    # Extract validation dataset from train dataset
    
    valid_set = train_set[0][50000:,:], train_set[1][50000:,]
    train_set = train_set[0][0:50000,:], train_set[1][0:50000,]
    test_set = test_set[0], test_set[1]

    if outputlayer=='svm':
        test_set_x, test_set_y = shared_svm(test_set)
        valid_set_x, valid_set_y = shared_svm(valid_set)
        train_set_x, train_set_y = shared_svm(train_set)

    if outputlayer=='logistic':
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def load_cifar10(outputlayer='svm', theano_shared=True):
    ''' Loads the cifar dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    train_set = X_train[0:45000,:,:].reshape((45000, 32*32*3)), y_train[0:45000,:].flatten()
    valid_set = X_train[45000:,:,:].reshape((5000, 32*32*3)), y_train[45000:,:].flatten()
    test_set =  X_test.reshape((len(X_test), 32*32*3)), y_test.flatten()

    if outputlayer=='svm':
        test_set_x, test_set_y = shared_svm(test_set)
        valid_set_x, valid_set_y = shared_svm(valid_set)
        train_set_x, train_set_y = shared_svm(train_set)

    if outputlayer=='logistic':
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval