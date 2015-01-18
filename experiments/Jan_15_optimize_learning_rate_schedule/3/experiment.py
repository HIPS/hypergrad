"""Gradient descent to optimize learning rate schedule"""
"""Expt 3: average over epochs"""
import numpy as np
import numpy.random as npr
import pickle
from functools import partial
import itertools as it
import socket

from hypergrad.data import load_data
from hypergrad.nn_utils import make_nn_funs, BatchList
from hypergrad.optimizers import sgd
# from hypergrad.odyssey import omap, collect_results

layer_sizes = [784, 200, 50, 10]
L2_reg = 0.0
log_param_scale = -2.0
velocity_scale = 0.0
batch_size = 250
log_alpha_0 = 0.0
beta_0 = 0.9
num_meta_iters = 1
log_param_scale = -2
N_data = 3 * 10**4
N_iters = 1200

# ----- Variable for this run -----
N_meta_iter = 4
meta_alpha = 2000

def step_smooth(x, L):
    # Turns x into a stepwise constant function with step length L
    N = len(x)
    y = np.zeros(N)
    i = 0
    while i < N:
        idxs = slice(i, i+L)
        y[idxs] = np.mean(x[idxs])
        i += L
    return y

def run():
    train_images, train_labels, _, _, _ = load_data()
    train_images = train_images[:N_data, :]
    train_labels = train_labels[:N_data, :]
    batch_idxs = BatchList(N_data, batch_size)
    iter_per_epoch = len(batch_idxs)
    N_weights, _, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
    def indexed_loss_fun(w, idxs):
        return loss_fun(w, X=train_images[idxs], T=train_labels[idxs])

    log_alphas = np.full(N_iters, log_alpha_0)
    betas      = np.full(N_iters, beta_0)
    npr.seed(1)
    V0 = npr.randn(N_weights) * velocity_scale
    W0 = npr.randn(N_weights) * np.exp(log_param_scale)
    output = []
    for i in range(N_meta_iter):
        print "Meta iteration {0}".format(i)
        results = sgd(indexed_loss_fun, batch_idxs, N_iters,
                      W0, V0, np.exp(log_alphas), betas, record_learning_curve=True)
        learning_curve = results['learning_curve']
        d_log_alphas = np.exp(log_alphas) * results['d_alphas']
        output.append((learning_curve, log_alphas, d_log_alphas))
        log_alphas = log_alphas - meta_alpha * step_smooth(d_log_alphas, iter_per_epoch)

    return output

def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        all_learning_curves, all_log_alphas, all_d_log_alphas = zip(*pickle.load(f))

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((8,12))

    ax = fig.add_subplot(311)
    ax.set_title("Step size schedule")
    for i, log_alphas in enumerate(all_log_alphas):
        ax.plot(log_alphas, 'o-')
    ax.set_ylabel("Log alpha")
    ax.set_xlabel("Step number")
    # ax.legend(loc=4)

    ax = fig.add_subplot(312)
    ax.set_title("Gradient wrt step size schedule")
    for i, d_log_alphas in enumerate(all_d_log_alphas):
        ax.plot(d_log_alphas, 'o-', label="{0} meta iters".format(i))
    ax.set_ylabel("Deriveative wrt log alpha")
    ax.set_xlabel("Step number")
    # ax.legend(loc=4)

    ax = fig.add_subplot(313)
    ax.set_title("Learning curve")
    for i, learning_curve in enumerate(all_learning_curves):
        ax.plot(learning_curve, 'o-', label="{0} meta iters".format(i))
    ax.set_ylabel("Negative log training loss")
    ax.set_xlabel("Epoch number")
    # ax.legend(loc=4)
    # plt.show()
    plt.savefig("/tmp/fig.png")
    plt.savefig("fig.png")

if __name__ == '__main__':
    # results = run()
    # with open('results.pkl', 'w') as f:
    #     pickle.dump(results, f)
    plot()
