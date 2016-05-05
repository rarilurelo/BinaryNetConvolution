# Binary CNN by chainer
## Binarized Neural Networks: Training Neural Networks with Weights and Activation Constrained to +1 or -1
I implement Binarized Neural Network by chainer.
There are three different point from ordinary CNN.

1. Using Binarized Weight
2. Using Binarized Input
3. Using weight clip that constraine gradient to -1 < x < 1

But I don't implement these below.

+ Shift Based Operation of
 + Batch Normalization
 + AdaMax
 + XNOR Dot
+ stochastic Binarization


## Usage
`./mnist_cnn.py`

`./cifar10_cnn.py`

You can choose options
+ gpu
+ epoch
+ batchsize

## code explanation
`link_binary_convolution.py` and `function_binary_convolution.py` define Link of chainer's object

`net.py` defines network

`weight_clip.py` constraines gradient to -1 < x < 1 at update step

## Reference
I implemented these codes hillbig/binary_net as reference
