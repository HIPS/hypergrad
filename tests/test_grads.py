import numpy as np
import numpy.random as npr
import itertools as it
from copy import copy

from hypergrad.nn_utils import BatchList
from hypergrad.optimizers import sgd, sgd2

npr.seed(0)

def nd(f, *args):
    unary_f = lambda x : f(*x)
    return unary_nd(unary_f, args)

def unary_nd(f, x):
    eps = 1e-2
    if isinstance(x, np.ndarray):
        nd_grad = np.zeros(x.shape)
        for dims in it.product(*map(range, x.shape)):
            nd_grad[dims] = unary_nd(indexed_function(f, x, dims), x[dims])
        return nd_grad
    elif isinstance(x, tuple):
        return tuple([unary_nd(indexed_function(f, list(x), i), x[i])
                      for i in range(len(x))])
    elif isinstance(x, dict):
        return {k : unary_nd(indexed_function(f, x, k), v) for k, v in x.iteritems()}
    elif isinstance(x, list):
        return [unary_nd(indexed_function(f, x, i), v) for i, v in enumerate(x)]
    else:
        return (f(x + eps/2) - f(x - eps/2)) / eps

def indexed_function(fun, arg, index):
    local_arg = copy(arg)
    def partial_function(x):
        local_arg[index] = x
        return fun(local_arg)
    return partial_function

def test_sgd():
    N_weights = 5
    W0 = 0.1 * npr.randn(N_weights)
    V0 = 0.1 * npr.randn(N_weights)
    N_data = 12
    batch_size = 4
    num_epochs = 3
    batch_idxs = BatchList(N_data, batch_size)
    N_iter = num_epochs * len(batch_idxs)
    alphas = 0.1 * npr.rand(len(batch_idxs) * num_epochs)
    betas = 0.5 + 0.2 * npr.rand(len(batch_idxs) * num_epochs)

    A = npr.randn(N_data, N_weights)
    def loss_fun(W, idxs):
        sub_A = A[idxs, :]
        return np.dot(np.dot(W, np.dot(sub_A.T, sub_A)), W)

    result = sgd(loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas)
    d_x = result['d_x']
    d_v = result['d_v']
    d_alphas = result['d_alphas']
    d_betas = result['d_betas']

    def full_loss(W0, V0, alphas, betas):
        result = sgd(loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas)
        x_final = result['x_final']
        return loss_fun(x_final, batch_idxs.all_idxs)

    d_an = (d_x, d_v, d_alphas, d_betas)
    d_num = nd(full_loss, W0, V0, alphas, betas)
    for i, (an, num) in enumerate(zip(d_an, d_num)):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)

def test_sgd2():
    N_weights = 5
    W0 = 0.1 * npr.randn(N_weights)
    V0 = 0.1 * npr.randn(N_weights)
    N_data = 12
    batch_size = 4
    num_epochs = 3
    batch_idxs = BatchList(N_data, batch_size)
    N_iter = num_epochs * len(batch_idxs)
    alphas = 0.1 * npr.rand(len(batch_idxs) * num_epochs)
    betas = 0.5 + 0.2 * npr.rand(len(batch_idxs) * num_epochs)
    meta = 0.1 * npr.randn(N_weights*2)

    A = npr.randn(N_data, N_weights)
    def loss_fun(W, meta, idxs):
        sub_A = A[idxs, :]
        return np.dot(np.dot(W + meta[:N_weights] + meta[N_weights:], np.dot(sub_A.T, sub_A)), W)

    def meta_loss_fun(meta):
        return np.dot(meta, meta)

    def full_loss(W0, V0, alphas, betas, meta):
        result = sgd2(loss_fun, meta_loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas, meta)
        return result['L_final']

    def meta_loss(W0, V0, alphas, betas, meta):
        result = sgd2(loss_fun, meta_loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas, meta)
        return result['M_final']

    result = sgd2(loss_fun, meta_loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas, meta)

    d_an = (result['dLd_x'], result['dLd_v'], result['dLd_alphas'], result['dLd_betas'], result['dLd_meta'])
    d_num = nd(full_loss, W0, V0, alphas, betas, meta )
    for i, (an, num) in enumerate(zip(d_an, d_num)):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)

    d_an = (result['dMd_x'], result['dMd_v'], result['dMd_alphas'], result['dMd_betas'], result['dMd_meta'])
    d_num = nd(meta_loss, W0, V0, alphas, betas, meta)
    for i, (an, num) in enumerate(zip(d_an, d_num)):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)
