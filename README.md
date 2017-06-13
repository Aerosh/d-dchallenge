#Custom Lenet CNN for image classification

## Description
Fully Lenet implementation for MNIST and CIFAR10 in Theano with Python 3.
Specifications :

* Depth : 5
* Layers : ConvPool-ConvPool-Conv-FC
* Activations : ReLu
* Data augmentation : Random Cropping with horizontal rotation
* Regularization : L2

## Structure

```
data						:			Store dataset
lenet						:			Contrain code
----Layer.py				:			Layer type classes
----LogisticRegression.py	:			Logistic Regression class
----Lenet5.py				:			Lenet5 model construction
----input_pipeline.py		:			Dataset loading/downloading and data augmentation
----train.py				:			Main run script to train/valid/test dataset with ogiven model
environment.yml				:			Virtual conda environment used
```

The code structure was made for an easy readability and an ease to expand to other model or other datasets. Mostly, group of utility function or classes have been regroup insto same python file.

## How to make it work
An Anaconda environment with Python3 was used all along the development. 

`environment.yml` can be used as virtual python environment under wich the code was tested with the following command

```
conda env create -f environment.yml
source activate theano_env:
```

Help :

```
usage: train.py [-h] [--data_augm] [--dataset {mnist,cifar10}]
                [--nepochs NEPOCHS] [--lr LR] [--mom MOM] [--l2_pen L2_PEN]
                [--lwidth LWIDTH [LWIDTH ...]] [--ncrops NCROPS]

Training Lenet model

optional arguments:
  -h, --help            show this help message and exit
  --data_augm           Enable data augmentation
  --dataset {mnist,cifar10}
                        Choose input dataset
  --nepochs NEPOCHS     Choose nb epochs
  --lr LR               Choose learning rate
  --mom MOM             Choose momentum
  --l2_pen L2_PEN       Choose penalization strength for l2 regularization
  --lwidth LWIDTH [LWIDTH ...]
                        Choose layer width for each layer. Use : --lwidth 1 2
                        3 4 max 4 arguments. First 3 are conv layer and last
                        is fc
  --ncrops NCROPS       Number of random crops in data augmentation
```
Default values :

```
data_augm = False 
dataset = "cifar10"
nepochs = 100
lr = 0.01
mom = 0.9
l2_pen = 0.001
lwidth = [16 31 64 128]
ncrops = 2
```

default run `python3 train.py` will rununder default arguments

## Problem approach
This repository is my first experience with theano, but not with deep learning frameworks.

I decided to go step by step instead of directly apply Lenet5 model on CIFAR 10. I first decided to simply understand how the logistic regression with MNIST was implemented and how it worked. This example was simple to understand and enabled me to directly test my understanding on how theano worked.

Then, I upgraded with a simple MLP with a single Hidden layer. I saw how elements were put together and which elements were essential. I also took some times to go through some reading on Theano documentation and understand its basic concepts.

The next step was the implementation of multiple convolutional layer with Lenet. It was an upgrade of the previous MLP, following nearly the same procesure, with a bit more of OOP involved. 

With this base architecture and understanding of Theano, I changed the structure to something I though was more clear and more easy to debug and improve. I tried to make things less dependant on values and more reliable on user input.

For the improvements and changes asked, I tried myself and started to look at some answers available on forums. I also find multiple github repositories with some theano code to validate my choices or to help into understanding even further.

Finally, I was also inspired by some paper, such as AlexNet, ResNet or LeNet5 for some intuition on data augmentation and architecture construction.

## Further improvements
(Suggestions)
### Dropout
Dropout is a common techniques is more recent CNN architectures. It has proven its efficiency to avoid overfitting by randomly remove a portion of the output data.

### Batch Normalization
Batch Normalization enable a normalization among layers and batches. It speed up computation and guarantee some uniform processing of the data
