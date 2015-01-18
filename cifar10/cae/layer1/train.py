import os
import sys
sys.path.append('../..')

import numpy
import theano
import theano.tensor as T

from anna import util
from anna.datasets import unsupervised_dataset

import checkpoints
from models import CAELayer1Model


def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    dim = numpy.min((dim1, dim2))
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2, dim1))
    s = s/s
    S[:dim, :dim] = numpy.diag(s)
    w = numpy.dot(u, numpy.dot(S, v))
    w = numpy.float32(w)
    return w


def conv_orthogonalize(w, k=1.0):
    # Reshape filters into a matrix
    channels, width, height, filters = w.shape
    w = w.reshape(channels*width*height, filters).transpose(1, 0)

    # Orthogonalize the matrix
    w = orthogonalize(w)

    # Contruct 2D hamming window
    hamming1 = numpy.hamming(width)
    hamming2 = numpy.hamming(height)
    hamming = numpy.outer(hamming1, hamming2)

    # Use it to mask the input to w
    mask = numpy.tile(hamming[None, :, :], (channels, 1, 1))
    mask = mask.reshape(channels*width*height)*k
    m = numpy.diag(mask)
    w = numpy.dot(w, m)

    # Reshape the matrix into filters
    w = w.transpose(1, 0)
    w = w.reshape(channels, width, height, filters)
    w = numpy.float32(w)
    return w

print('Start')

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid', 'wb')
f.write(str(pid)+'\n')
f.close()

model = CAELayer1Model('experiment', './', learning_rate=1e-4)
monitor = util.Monitor(model, save_steps=200)

# Function to compute sparsity
output = theano.function([model.input.output()],
                         T.mean(T.sum(model.conv1.output() > 0, axis=0)))

# Loading CIFAR-10 dataset
print('Loading Data')
train_iterator = util.get_cifar_iterator('train',
                                         mode='random_uniform',
                                         batch_size=128,
                                         num_batches=100000,
                                         rescale=True)

test_iterator = util.get_cifar_iterator('test',
                                        mode='sequential',
                                        batch_size=128,
                                        rescale=True)

normer = util.Normer2(filter_size=5, num_channels=3)

# Grab batch for patch extraction.
x_batch, y_batch = train_iterator.next()
x_batch = normer.run(x_batch)
# Grab some patches to initialize weights.
patch_grabber = util.PatchGrabber(96, 5)
patches = patch_grabber.run(x_batch)*0.01
model.conv1.W.set_value(patches)

# Grab test data to give to NormReconVisualizer.
test_x_batch, test_y_batch = test_iterator.next()
test_x_batch = normer.run(test_x_batch)
recon_visualizer = util.NormReconVisualizer(model, test_x_batch, steps=100)
recon_visualizer.run()

# Create object to display first layer filter weights.
filter_visualizer = util.FilterVisualizer(model, steps=100)
filter_visualizer.run()

#model.learning_rate_symbol.set_value(0.000005/10)
print('Training Model')
count = 0
for x_batch, y_batch in train_iterator:
    monitor.start()
    x_batch = normer.run(x_batch)
    error = model.train(x_batch)
    monitor.stop(error)
    recon_visualizer.run()
    filter_visualizer.run()
    if count % 50 == 0:
        sparsity = output(x_batch)
        print 'Sparsity: %.3f' % sparsity
    count += 1
