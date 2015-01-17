import os
import sys
sys.path.append('../..')

import numpy
import theano
import theano.tensor as T

from anna import util
from anna.datasets import unsupervised_dataset

import checkpoints
from models import CAELayer2Model


def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    dim = numpy.min((dim1, dim2))
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:dim,:dim] = numpy.diag(s)
    w = numpy.dot(u,numpy.dot(S,v))
    w = numpy.float32(w)
    return w

def conv_orthogonalize(w, k=1.0):
    # Reshape filters into a matrix
    channels, width, height, filters = w.shape
    w = w.reshape(channels*width*height, filters).transpose(1,0)

    # Orthogonalize the matrix
    w = orthogonalize(w)

    # Contruct 2D hamming window
    hamming1 = numpy.hamming(width)
    hamming2 = numpy.hamming(height)
    hamming = numpy.outer(hamming1, hamming2)

    # Use it to mask the input to w
    mask = numpy.tile(hamming[None,:,:], (channels,1,1))
    mask = mask.reshape(channels*width*height)*k
    m = numpy.diag(mask)
    w = numpy.dot(w, m)

    # Reshape the matrix into filters
    w = w.transpose(1,0)
    w =  w.reshape(channels, width, height, filters)
    w = numpy.float32(w)
    return w

print('Start')

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid', 'wb')
f.write(str(pid)+'\n')
f.close()

model = CAELayer2Model('experiment', './', learning_rate=1e-5)
checkpoint = checkpoints.unsupervised_greedy
util.set_parameters_from_unsupervised_model(model, checkpoint)
monitor = util.Monitor(model, save_steps=200)

model.conv1.trainable = False
model._compile()

# Function to compute sparsity
output1 = theano.function([model.input.output()], T.mean(T.sum(model.conv1.output()>0, axis=0)))
output2 = theano.function([model.input.output()], T.mean(T.sum(model.conv2.output()>0, axis=0)))

# Loading STL-10 dataset
print('Loading Data')
data = numpy.load('/data/stl10_matlab/unsupervised.npy')
data = numpy.float32(data)
data /= 255.0
data *= 2.0
train_data = data[0:90000, :, :, :]
test_data = data[90000::, :, :, :]

train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=128, num_batches=100000)
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)

# Create object to local contrast normalize a batch.
# Note: Every batch must be normalized before use.
normer = util.Normer2(filter_size=5, num_channels=3)

# Orthogonalize second layer weights.
W2 = model.conv2.W.get_value()
W2 = conv_orthogonalize(W2)
# Scale second layer weights.
s=2.5
model.conv2.W.set_value(W2*s)

# Grab test data to give to NormReconVisualizer.
test_x_batch = test_iterator.next()
test_x_batch = test_x_batch.transpose(1, 2, 3, 0)
test_x_batch = normer.run(test_x_batch)
recon_visualizer = util.NormReconVisualizer(model, test_x_batch, steps=200)
recon_visualizer.run()   

# Create object to display first layer filter weights.
filter_visualizer = util.FilterVisualizer(model, steps=200)
filter_visualizer.run()

print('Training Model')
count = 0
for x_batch in train_iterator:
    x_batch = x_batch.transpose(1, 2, 3, 0)    
    monitor.start()
    x_batch = normer.run(x_batch)
    error = model.train(x_batch)
    monitor.stop(error) 
    recon_visualizer.run()   
    filter_visualizer.run()
    if count % 100 == 0:
        sparsity_layer_1 = output1(x_batch)
        sparsity_layer_2 = output2(x_batch)
        print 'Sparsity (Layer 1): %.3f' % sparsity_layer_1
        print 'Sparsity (Layer 2): %.3f' % sparsity_layer_2
    count += 1
