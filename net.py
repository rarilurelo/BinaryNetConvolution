import chainer
import chainer.functions as F
import chainer.links as L
import link_binary_linear
import link_binary_convolution
import bst

class MnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=link_binary_linear.BinaryLinear(n_in, n_units),
            b1=L.BatchNormalization(n_units),
            l2=link_binary_linear.BinaryLinear(n_units, n_units),
            b2=L.BatchNormalization(n_units),
            l3=link_binary_linear.BinaryLinear(n_units, n_out),
            b3=L.BatchNormalization(n_out),
        )
        self.train = True

    def __call__(self, x):
        h1 = bst.bst(self.b1(self.l1(x), test=not self.train))
        h2 = bst.bst(self.b2(self.l2(h1), test=not self.train))
        return self.b3(self.l3(h2), test=not self.train)

class MnistCNN(chainer.Chain):
    """ An example of binary CNN for MNIST dataset."""
    def __init__(self):
        super(MnistCNN, self).__init__(
            c1=link_binary_convolution.BinaryConvolution2D(1, 32, 5),
            b1=L.BatchNormalization(32),
            c2=link_binary_convolution.BinaryConvolution2D(32, 64, 5),
            b2=L.BatchNormalization(64),
            c3=link_binary_convolution.BinaryConvolution2D(64, 128, 5),
            b3=L.BatchNormalization(128),
            l1=link_binary_linear.BinaryLinear(128*16*16, 256),
            b4=L.BatchNormalization(256),
            l2=link_binary_linear.BinaryLinear(256, 10),
            b5=L.BatchNormalization(10)
        )
        self.train = True

    def __call__(self, x):
        h1 = bst.bst(self.b1(self.c1(x), test=not self.train))
        h2 = bst.bst(self.b2(self.c2(h1), test=not self.train))
        h3 = bst.bst(self.b3(self.c3(h2), test=not self.train))
        h4 = bst.bst(self.b4(self.l1(h3), test=not self.train))
        return self.b5(self.l2(h4), test=not self.train)

class Cifar10CNN(chainer.Chain):
    """ binary CNN fot Cifar10"""
    def __init__(self):
        super(Cifar10CNN, self).__init__(
            c1=link_binary_convolution.BinaryConvolution2D(3, 32, 5),
            b1=L.BatchNormalization(32),
            c2=link_binary_convolution.BinaryConvolution2D(32, 64, 5),
            b2=L.BatchNormalization(64),
            c3=link_binary_convolution.BinaryConvolution2D(64, 128, 5),
            b3=L.BatchNormalization(128),
            l1=link_binary_linear.BinaryLinear(128*20*20, 256),
            b4=L.BatchNormalization(256),
            l2=link_binary_linear.BinaryLinear(256, 10),
            b5=L.BatchNormalization(10)
        )
        self.train = True

    def __call__(self, x):
        h1 = bst.bst(self.b1(self.c1(x), test=not self.train))
        h2 = bst.bst(self.b2(self.c2(h1), test=not self.train))
        h3 = bst.bst(self.b3(self.c3(h2), test=not self.train))
        h4 = bst.bst(self.b4(self.l1(h3), test=not self.train))
        return self.b5(self.l2(h4), test=not self.train)
        

