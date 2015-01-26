# An Analysis of Unsupervised Pre-training in Light of Recent Advances
This repo contains the experiment files for the paper "An Analysis of Unsupervised Pre-training in Light of Recent Advances", available here: http://arxiv.org/abs/1412.6597

The experiments are split into two sections:
+ cifar10
+ stl10

The `README.md` file in each folder will give you more information about running experiments.

The experiments are written in python 2.7, and require open source software to run, including:
+ [numpy][numpy], a standard numerical computing library for python.
+ [anna][anna], our library for quickly design new neural networks, which itself depends on [theano][theano] and [pylearn2][pylearn2]. The pylearn dependencies are relatively small, and we may remove them to limit the number of dependencies.

[numpy]:http://www.numpy.org/
[theano]:http://deeplearning.net/software/theano/
[pylearn2]:http://deeplearning.net/software/pylearn2/
[anna]:https://github.com/ifp-uiuc/anna

## Status
All experiments are added. Just need to finalize documentation.
