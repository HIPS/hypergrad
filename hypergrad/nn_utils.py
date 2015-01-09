import numpy as np
from funkyyak import grad

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

def make_nn_funs(layer_sizes, L2_reg):
    parser = WeightsParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_weights(('weights', i), shape)
        parser.add_weights(('biases', i), (1, shape[1]))

    def predictions(W_vect, X):
        cur_units = X
        for i in range(len(layer_sizes) - 1):
            cur_W = parser.get(W_vect, ('weights', i))
            cur_B = parser.get(W_vect, ('biases', i))
            cur_units = np.tanh(np.dot(cur_units, cur_W) + cur_B)
        return cur_units - logsumexp(cur_units, axis=1)

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T) / X.shape[0]
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        preds = np.argmax(pred_fun(getval(W_vect), X), axis=1)
        return np.mean(np.argmax(T, axis=1) != preds)

    return parser.N, predictions, loss, frac_err

