from __future__ import print_function

import argparse
import sys
import os
import timeit

import numpy

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import theano.tensor as T

from LogisticRegression import LogisticRegression
from Lenet5 import Lenet5
import input_pipeline as in_pip

# script parser
parser = argparse.ArgumentParser(description='Training Lenet model')
parser.add_argument('--data_augm', action='store_true',
                    help='Enable data augmentation')
parser.add_argument('--dataset', choices=['mnist', 'cifar10'],
                    help='Choose input dataset', default='cifar10')
parser.add_argument('--nepochs', type=int,
                    help='Choose nb epochs', default=100)
parser.add_argument('--lr', type=float,
                    help='Choose learning rate', default=0.01)
parser.add_argument('--mom', type=float, help='Choose momentum', default=0.9)
parser.add_argument('--l2_pen', type=float,
                    help='Choose penalization strength for l2 regularization', default=0.01)
parser.add_argument('--lwidth', nargs='+', type=int,
                    help='Choose layer width for each layer. Use : --lwidth 1 2 3 4 '
                    + 'max 4 arguments. First 3 are conv layer and last is fc')
parser.add_argument('--ncrops', type=int, default=2,
                    help='Number of random crops in data augmentation')
args = parser.parse_args()


def get_parser_argument(start_msg):
    '''
    Load argument from parser and place default values if needed
    '''

    if args.dataset == "mnist":
        start_msg += " mnist"
    elif args.dataset == "cifar10":
        start_msg += " cifar10"

    print(start_msg)

    if args.lwidth is None:
        lwidth = [16, 32, 64, 128]
    else:
        lwidth = args.lwidth

    return (args.data_augm, args.dataset, args.nepochs,
            args.lr, args.mom, args.l2_pen, lwidth)


def sgd_momentum(cost, params, learning_rate, momentum, grads):
    '''
    Updates parameters with respect to stochastic gradient descent with a momentum
    :type cost : Theano compatible function
    :param cost : model cost function computation

    :type params : List of shared theano variable
    :param params : model parameters

    :type learning_rate : theano.tensor.scalar
    :param learning_rate : learning rate step for SGD optimization

    :type momentum : theano.tensor.scalar
    :param momentum : Control velocity strength

    :type grads : theano variable
    :param grads : model parameters gradient
    '''

    updates = []

    for param_i, grad_i in zip(params, grads):
        # Store previous iteration for momentum. Initilization to 0
        previous_step = theano.shared(
            param_i.get_value() * 0., broadcastable=param_i.broadcastable)

        # Step towards the gradient taking into consideration the velocity
        step = momentum * previous_step - learning_rate * grad_i

        # Store update tuples into updates list
        updates.append((param_i, param_i + step))

        # store update rules of previous step
        updates.append((previous_step, step))

    return updates


def evaluate_lenet5(lwidth, learning_rate, momentum, n_epochs,
                    l2_pen, data_aug, dataset, batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type momentum: float
    :param momentum: strength controlling velocity in SGD updates rule 

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type lwidth: list of ints
    :param lWidth: number of kernels on each layer (both convolutional and fully connected)
    """

    rng = numpy.random.RandomState(23455)

    #####################
    # DATA CONSTRUCTION #
    #####################

    if (dataset == 'mnist'):
        datasets = in_pip.load_mnist()
    elif (dataset == 'cifar10'):
        datasets = in_pip.load_cifar10(data_aug=data_aug, n_crops=args.ncrops)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    ################################
    # SYMBOLIC VARIABLE ALLOCATION #
    ################################

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # allocate symbolic variables keeping track of learning rate and momentum
    l_r = T.scalar('l_r', dtype=theano.config.floatX)
    mom = T.scalar('mom', dtype=theano.config.floatX)

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Create Lenet5 class
    if dataset == "mnist":
        model = Lenet5(input=x, label=y, lwidth=lwidth,
                       batch_size=batch_size, rng=rng)
    elif dataset == "cifar10":
        model = Lenet5(input=x, label=y, lwidth=lwidth,
                       batch_size=batch_size, rng=rng, size=32, channels=3,
                       l2_pen=l2_pen)

    # Grab values from model
    cost = model.cost
    params = model.params
    errors = model.errors

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        errors,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        errors,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = sgd_momentum(cost, params, l_r, mom, grads)

    train_model = theano.function(
        [index, l_r, mom],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############

    print('... training with learning rate ' + str(learning_rate))
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
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
        if epoch % 10 == 0 and epoch != 0:
            learning_rate *= 0.95
            momentum *= 0.95
            print("\t new learing rate : " + str(learning_rate) +
                  " and new momentum : " + str(momentum))

        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index, learning_rate, momentum)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    start_msg = "Starting training of LeNet5"
    data_aug, dataset, n_epochs, lr, mom, l2_pen, lwidth = get_parser_argument(
        start_msg)
    evaluate_lenet5(lwidth=lwidth, n_epochs=n_epochs, dataset=dataset,
                    learning_rate=lr, momentum=mom, l2_pen=l2_pen, data_aug=data_aug)
