import os
import urllib
import gzip
import struct
import array
import numpy as np

datadir = os.path.expanduser('~/repos/Kayak/examples/data')

def download(url, filename):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    out_file = os.path.join(datadir, filename)
    if not os.path.isfile(out_file):
        urllib.urlretrieve(url, out_file)

def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images(os.path.join(datadir, 'train-images-idx3-ubyte.gz'))
    train_labels = parse_labels(os.path.join(datadir, 'train-labels-idx1-ubyte.gz'))
    test_images  = parse_images(os.path.join(datadir, 't10k-images-idx3-ubyte.gz'))
    test_labels  = parse_labels(os.path.join(datadir, 't10k-labels-idx1-ubyte.gz'))

    return train_images, train_labels, test_images, test_labels

def load_data(normalize=False):
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    if normalize:
        train_mean = np.mean(train_images, axis=0)
        train_images = train_images - train_mean
        test_images = test_images - train_mean
    return train_images, train_labels, test_images, test_labels, N_data

