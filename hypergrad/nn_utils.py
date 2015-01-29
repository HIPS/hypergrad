import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict

class WeightsParser(object):
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def set(self, vect, name, val):
        idxs, shape = self.idxs_and_shapes[name]
        if isinstance(val, np.ndarray):
            vect[idxs] = val.ravel()
        else:
            vect[idxs] = val  # Can't unravel a float.

class VectorParser(object):
    def __init__(self,):
        self.idxs_and_shapes = OrderedDict()
        self.vect = np.zeros((0,))

    def add_shape(self, name, shape):
        start = len(self.vect)
        size = np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, start + size), shape)
        self.vect = np.concatenate((self.vect, np.zeros(size)), axis=0)

    def new_vect(self, vect):
        new_parser = self.copy()
        new_parser.vect = vect
        return new_parser

    def copy(self):
        """Creates a parser with a blank vector."""
        new_parser = VectorParser()
        for name, (idxs, shape) in self.idxs_and_shapes.iteritems():
            new_parser.add_shape(name, shape)
        return new_parser

    @property
    def names(self):
        return self.idxs_and_shapes.keys()

    def __getitem__(self, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(self.vect[idxs], shape)

    def __setitem__(self, name, val):
        if isinstance(val, list): val = np.array(val)
        assert isinstance(val, np.ndarray)
        if name not in self.idxs_and_shapes:
            self.add_shape(name, val.shape)

        idxs, shape = self.idxs_and_shapes[name]
        assert val.shape == shape
        self.vect[idxs] = val.ravel()

class BatchList(list):
    def __init__(self, N_total, N_batch):
        start = 0
        while start < N_total:
            self.append(slice(start, start + N_batch))
            start += N_batch
        self.all_idxs = slice(0, N_total)

def logsumexp(X, axis):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def logit(x): return 1 / (1 + np.exp(-x))
def inv_logit(y): return -np.log( 1/y - 1)
def d_logit(x): return logit(x) * (1 - logit(x))

def make_nn_funs(layer_sizes):
    parser = VectorParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_shape(('weights', i), shape)
        parser.add_shape(('biases', i), (1, shape[1]))

    def predictions(W_vect, X):
        """Outputs normalized log-probabilities."""
        W = parser.new_vect(W_vect)
        cur_units = X
        N_iter = len(layer_sizes) - 1
        for i in range(N_iter):
            cur_W = W[('weights', i)]
            cur_B = W[('biases',  i)]
            cur_units = np.dot(cur_units, cur_W) + cur_B
            if i == (N_iter - 1):
                cur_units = cur_units - logsumexp(cur_units, axis=1)
            else:
                cur_units = np.tanh(cur_units)
        return cur_units

    def loss(W_vect, X, T, L2_reg=0.0):
        # TODO: consider treating L2_reg as a matrix
        log_prior = -np.dot(W_vect * L2_reg, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T) / X.shape[0]
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        preds = np.argmax(predictions(W_vect, X), axis=1)
        return np.mean(np.argmax(T, axis=1) != preds)

    return parser, predictions, loss, frac_err


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28,28)):
    N_images = images.shape[0]
    N_rows = np.ceil(float(N_images) / ims_per_row)
    concat_images = np.zeros(((digit_dimensions[0] + padding) * N_rows + padding,
                              (digit_dimensions[0] + padding) * ims_per_row + padding))
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i / ims_per_row  # Integer division.
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0])*row_ix
        col_start = padding + (padding + digit_dimensions[0])*col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[0]] \
            = cur_image
    ax.matshow(concat_images, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

plot_mnist_images = plot_images
