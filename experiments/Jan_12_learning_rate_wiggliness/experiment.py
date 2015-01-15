"""Gradient wrt learning rate"""
import numpy as np
import numpy.random as npr
import pickle
from functools import partial
import itertools as it
import socket

from hypergrad.data import load_data
from hypergrad.nn_utils import make_nn_funs, BatchList
from hypergrad.optimizers import sgd
from hypergrad.odyssey import omap

layer_sizes = [784, 200, 50, 10]
L2_reg = 0.0
log_param_scale = -2.0
velocity_scale = 0.0
batch_size = 250
alpha_0 = 1.0
beta_0 = 0.9
num_meta_iters = 1
all_N_iters = [0, 1, 5, 10] 
N_oiter = 4
all_log_param_scale = np.linspace(-4, 1, N_oiter)
N_data = 10**2

def d_log_loss(x, dx):
    return np.sum(x * dx)

def run(oiter):
    print "Running job {0} on {1}".format(oiter + 1, socket.gethostname())
    train_images, train_labels, _, _, _ = load_data()
    train_images = train_images[:N_data, :]
    train_labels = train_labels[:N_data, :]
    batch_idxs = BatchList(N_data, batch_size)
    iter_per_epoch = len(batch_idxs)
    N_weights, _, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
    def indexed_loss_fun(w, idxs):
        return loss_fun(w, X=train_images[idxs], T=train_labels[idxs])

    V0 = npr.randn(N_weights) * velocity_scale
    losses = []
    d_losses = []
    for N_iters in all_N_iters:
        alphas = np.full(N_iters, alpha_0)
        betas = np.full(N_iters, beta_0)
        log_param_scale = all_log_param_scale[oiter]
        npr.seed(1)
        W0 = npr.randn(N_weights) * np.exp(log_param_scale)
        results = sgd(indexed_loss_fun, batch_idxs, N_iters, W0, V0, alphas, betas)
        losses.append(results['loss_final'])
        d_losses.append(d_log_loss(W0, results['d_x']))

    return losses, d_losses

def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        losses, d_losses = zip(*pickle.load(f))

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((8,8))
    ax = fig.add_subplot(211)
    for loss_curve, N_iter in zip(losses, all_N_iters):
        ax.plot(all_log_param_scale, loss_curve, 'o-', label="{0} iters".format(N_iter))
    ax.set_title("Loss vs param scale")
    ax.set_ylim([-0, 3])
    ax.set_ylabel("Negative log loss per datum")
    ax.set_xlabel("Log param scale")
    ax.legend(loc=4)
    ax = fig.add_subplot(212)
    for d_loss_curve, N_iter in zip(d_losses, all_N_iters):
        ax.plot(all_log_param_scale, d_loss_curve, 'o-', label="{0} iters".format(N_iter))
    ax.set_title("Grad loss vs param scale")
    ax.set_ylim([-2, 2])
    ax.set_ylabel("Negative log loss per datum")
    ax.set_xlabel("Log param scale")
    ax.legend(loc=2)
    plt.savefig("/tmp/fig.png")
    plt.savefig("fig.png")

if __name__ == '__main__':
    results = omap(run, range(N_oiter))
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()
