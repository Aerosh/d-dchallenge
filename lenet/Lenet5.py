from Layer import ConvPoolLayer, HiddenLayer, ConvLayer
from LogisticRegression import LogisticRegression

import theano
import theano.tensor as T

class Lenet5(object):
    
    def __init__(self, input, label, rng, lwidth, size=28, channels=1, batch_size=500, l2_pen = 0.001):

        '''
        Construct Lenet CNN model based on Yan LeCun paper. 
        Activations have been changed to Relu 

        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type lwidth: tuple or list of length 5
        :param lwidth: width (nb of kernel) for each layer (convolutionnal and fully connected)

        :type size: int
        :param size: size of the original image input. Default is MNIST (28 x 28)

        :type channels: int
        :param channels: Nb channels of original images. Default is MNIST (1 channel) 
        '''


        # Reshape matrix of rasterized images of shape (batch_size, size * size * channels)
        # to a 4D tensor, compatible with our ConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = input.reshape((batch_size, channels, size, size))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, lwidth[0], 12, 12)
        layer0 = ConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, channels, size, size),
            filter_shape=(lwidth[0], channels, 5, 5),
            poolsize=(2, 2),
            activation=T.nnet.relu
        )

        size = int((size - 4)/2)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, lwidth[1], 4, 4)
        layer1 = ConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, lwidth[0], size, size),
            filter_shape=(lwidth[1], lwidth[0], 5, 5),
            poolsize=(2, 2),
            activation=T.nnet.relu
        )

        size = int((size - 4)/2)

        # Construct strictly convolutional layer
        # filtering reduces the image size to (4-3+1, 4-3+1) = (2, 2)
        # 4D output tensor is thus of shape (batch_size, lwidth[2], 2, 2)
        layer2 = ConvLayer(
            rng,
            input=layer1.output,
            image_shape=(batch_size, lwidth[1], size, size),
            filter_shape=(lwidth[2], lwidth[1], 3, 3),
            activation=T.nnet.relu
        )

        size = int(size - 2)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, lwidth[2] * 2 * 2),
        # or (500, 120 * 2 * 2) = (500, 480) with the default values.
        layer3_input = layer2.output.flatten(2)
        
        # construct a fully-connected sigmoidal layer
        layer3 = HiddenLayer(
            rng,
            input=layer3_input,
            n_in=lwidth[2] * size * size,
            n_out=lwidth[3],
            activation=T.nnet.relu
        )

        # classify the values of the fully-connected sigmoidal layer
        layer4 = LogisticRegression(input=layer3.output, n_in=lwidth[3], n_out=10)

        # create a list with all layers parameters
        self.params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params

        # Include weight decay regularization for weights W avoiding overfitting
        self.L2_sq = ((layer0.W **2).sum() + (layer1.W **2).sum()  
                    + (layer2.W **2).sum() + (layer3.W **2).sum())


        # the cost we minimize during training is the NLL of the model
        self.cost = layer4.negative_log_likelihood(label) + l2_pen*self.L2_sq

        self.errors = layer4.errors(label)
        
