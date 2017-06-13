import six.moves.cPickle as pickle
from six.moves import urllib
import gzip
import os
import random
import numpy as np
import theano
import theano.tensor as T


def load_mnist(dataset="../data/mnist.pkl.gz"):
    ''' Loads the MNIST dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # Check if mnist dataset present or download it
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a np.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # np.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_cifar10(root="../data/cifar-10-batches-py/", data_aug=False, n_crops=2):
    ''' Loads Cifar-10 dataset '''
    train_val_x = []
    train_val_y = []

    test_set_x = []
    test_set_y = []

    def unpickle(file):
        ''' Extract data from bathces provided for CIFAR '''

        with open(file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = np.array(batch[b'data'] / 255.0)
            labels = np.array(batch[b'labels'])

        return images, labels

    def download_cifar10(target_dir='../data'):
        '''
        Download cifar10 dataset if not present
        '''

        target_path = os.path.join(target_dir, "cifar-10-python.tar.gz")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if not os.path.exists(target_path):
            origin = (
                "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, os.path.join(
                target_dir, "cifar-10-python.tar.gz"))

        print("Extracting ...")
        os.system("tar xzvf " + os.path.join(target_dir, "cifar-10-python.tar.gz")
                  + " -C " + target_dir)

        print("done.")

    # Check if directory exist, toherwise download
    if not os.path.exists(root):
        download_cifar10()

    # We iterate over the batches for cifar 10 for the validation and training set
    for filename in os.listdir(root):
        if filename.startswith("data_batch"):
            images, labels = unpickle(os.path.join(root, filename))
            train_val_x.append(images)
            train_val_y.append(labels)

        elif filename.startswith("test_batch"):
            test_x, test_y = unpickle(os.path.join(root, filename))

    # Split train set into train and validation set with split of 80/20
    train_xy = (np.concatenate(np.array(train_val_x[0:-1]), axis=0),
                np.concatenate(np.array(train_val_y[0:-1]), axis=0))
    valid_xy = (train_val_x[-1], train_val_y[-1])

    if data_aug:
        # Perform data augmentation on training set
        train_xy = data_augmentation(train_xy, 32, 3, n_crops=n_crops)

    # Produce shared variables
    train_set_x, train_set_y = shared_dataset(train_xy)
    valid_set_x, valid_set_y = shared_dataset(valid_xy)
    test_set_x, test_set_y = shared_dataset((test_x, test_y))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
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


def data_augmentation(data_xy, im_size, im_chan, pad=4, n_crops=2):
    '''
    Artificial training set augmentation. We create new noisy data from original dataset. 
    It virtually increase our pool of available data and present altered version with similar label

    Note : Decision was made to augment input data before training instead of adding a layer. 
    It reduces training time overload.
    '''

    print("... augmenting data")

    input, labels = data_xy
    train_size, _ = input.shape
    # Input reshaped
    input = np.reshape(input, (train_size, im_chan, im_size, im_size))

    # We first add a mirrored vesion of our input to our dataset
    output = np.concatenate([input, input[:, :, ::-1, :]], axis=0)
    in_mirr_labels = np.concatenate([labels, labels], axis=0)
    output_labels = in_mirr_labels

    # First, we add padding to height and width to firther to a random crop
    # of the same image size
    input_padded = np.zeros(
        (train_size, im_chan, im_size + 2 * pad, im_size + 2 * pad))
    input_padded[:, :, pad:-pad, pad:-pad] = input

    # Then we produce a mirror version of our batch and concatenate it to the input
    # It will be used for choosing randomly if an image is horizontally flipped or not.
    mirror_input = input_padded[:, :, ::-1, :]
    possible_output = np.concatenate([input_padded, mirror_input], axis=0)

    def gen_random_tuples(max, n=3):
        curr_list = []
        while len(curr_list) < 3:
            tuple = (random.randint(0, max), random.randint(0, max))
            if tuple not in curr_list:
                curr_list.append(tuple)

        return curr_list
    # We decide to take n_crops similar random crops both from our padded input and
    # its mirrored version
    crops_top_coord = gen_random_tuples(2 * pad, n_crops)

    for x_crop, y_crop in crops_top_coord:
        output = np.concatenate(
            [output, possible_output[:, :, x_crop:x_crop + im_size, y_crop:y_crop + im_size]], axis=0)
        output_labels = np.concatenate([output_labels, in_mirr_labels], axis=0)

    train_size, _, _, _ = output.shape
    output = output.reshape((train_size, -1))
    return output, output_labels
