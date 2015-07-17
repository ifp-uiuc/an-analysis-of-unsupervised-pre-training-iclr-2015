import numpy

from anna.layers import layers, cc_layers
import anna.models


class CAELayer1Model(anna.models.UnsupervisedModel):
    batch = 128
    input = cc_layers.Input2DLayer(batch, 3, 32, 32)

    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*3)
    binit = 0.0

    def trec(x):
        return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.Conv2DNoBiasLayer(
        input,
        n_filters=96,
        filter_size=5,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=2)
    pool1 = cc_layers.Pooling2DLayer(conv1, 2, stride=2)
    unpool2 = cc_layers.Unpooling2DLayer(pool1, pool1)
    output = cc_layers.Deconv2DNoBiasLayer(
        unpool2, conv1, nonlinearity=layers.identity)


class CAELayer2Model(anna.models.UnsupervisedModel):
    batch = 128
    input = cc_layers.Input2DLayer(batch, 3, 32, 32)

    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*3)
    winit2 = k/numpy.sqrt(5*5*96)
    binit = 0.0

    def trec(x):
        return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.Conv2DNoBiasLayer(
        input,
        n_filters=96,
        filter_size=5,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=2)
    pool1 = cc_layers.Pooling2DLayer(conv1, 2, stride=2)
    conv2 = cc_layers.Conv2DNoBiasLayer(
        pool1,
        n_filters=144,
        filter_size=5,
        weights_std=winit2,
        nonlinearity=nonlinearity,
        pad=2)
    pool2 = cc_layers.Pooling2DLayer(conv2, 2, stride=2)
    unpool3 = cc_layers.Unpooling2DLayer(pool2, pool2)
    deconv3 = cc_layers.Deconv2DNoBiasLayer(
        unpool3, conv2, nonlinearity=layers.identity)
    unpool4 = cc_layers.Unpooling2DLayer(deconv3, pool1)
    output = cc_layers.Deconv2DNoBiasLayer(
        unpool4, conv1, nonlinearity=layers.identity)


class CAELayer3Model(anna.models.UnsupervisedModel):
    batch = 128
    input = cc_layers.Input2DLayer(batch, 3, 32, 32)

    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*3)
    winit2 = k/numpy.sqrt(5*5*96)
    winit3 = k/numpy.sqrt(5*5*144)
    binit = 0.0

    def trec(x):
        return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.Conv2DNoBiasLayer(
        input,
        n_filters=96,
        filter_size=5,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=2)
    pool1 = cc_layers.Pooling2DLayer(conv1, 2, stride=2)
    conv2 = cc_layers.Conv2DNoBiasLayer(
        pool1,
        n_filters=144,
        filter_size=5,
        weights_std=winit2,
        nonlinearity=nonlinearity,
        pad=2)
    pool2 = cc_layers.Pooling2DLayer(conv2, 2, stride=2)
    conv3 = cc_layers.Conv2DNoBiasLayer(
        pool2,
        n_filters=192,
        filter_size=3,
        weights_std=winit3,
        nonlinearity=nonlinearity,
        pad=1)
    deconv4 = cc_layers.Deconv2DNoBiasLayer(
        conv3, conv3, nonlinearity=layers.identity)
    unpool5 = cc_layers.Unpooling2DLayer(deconv4, pool2)
    deconv5 = cc_layers.Deconv2DNoBiasLayer(
        unpool5, conv2, nonlinearity=layers.identity)
    unpool6 = cc_layers.Unpooling2DLayer(deconv5, pool1)
    output = cc_layers.Deconv2DNoBiasLayer(
        unpool6, conv1, nonlinearity=layers.identity)


class CNNModel(anna.models.SupervisedModel):
    batch = 128
    input = cc_layers.Input2DLayer(batch, 3, 32, 32)

    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*3)
    winit2 = k/numpy.sqrt(5*5*96)
    winit3 = k/numpy.sqrt(5*5*144)
    binit = 0.0

    def trec(x):
        return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.Conv2DNoBiasLayer(
        input,
        n_filters=96,
        filter_size=5,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=2)
    pool1 = cc_layers.Pooling2DLayer(conv1, 2, stride=2)
    conv2 = cc_layers.Conv2DNoBiasLayer(
        pool1,
        n_filters=144,
        filter_size=5,
        weights_std=winit2,
        nonlinearity=nonlinearity,
        pad=2)
    pool2 = cc_layers.Pooling2DLayer(conv2, 2, stride=2)
    conv3 = cc_layers.Conv2DNoBiasLayer(
        pool2,
        n_filters=192,
        filter_size=3,
        weights_std=winit3,
        nonlinearity=nonlinearity,
        pad=1)

    winitD1 = k/numpy.sqrt(numpy.prod(conv3.get_output_shape()))
    winitD2 = k/numpy.sqrt(300)

    conv3_shuffle = cc_layers.ShuffleC01BToBC01Layer(conv3)
    fc4 = layers.DenseLayer(
        conv3_shuffle,
        n_outputs=300,
        weights_std=winitD1,
        init_bias_value=1.0,
        nonlinearity=layers.rectify,
        dropout=0.0)
    output = layers.DenseLayer(
        fc4,
        n_outputs=10,
        weights_std=winitD2,
        init_bias_value=0.0,
        nonlinearity=layers.softmax)
