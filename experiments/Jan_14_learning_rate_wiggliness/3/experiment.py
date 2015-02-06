"""Gradient wrt learning rate"""
import numpy as np
import numpy.random as npr
import pickle

from hypergrad.data import load_data
from hypergrad.nn_utils import make_nn_funs, BatchList
from hypergrad.optimizers import sgd
from hypergrad.odyssey import omap

layer_sizes = [784, 200, 50, 10]
L2_reg = 0.0
log_param_scale = -2.0
velocity_scale = 0.0
batch_size = 250
log_alpha_0 = 0.0
beta_0 = 0.9
num_meta_iters = 1
log_param_scale = -2
N_iters = 50
N_data = 10**3

N_meta_iter = 1000
log_stepsizes = np.linspace(0.5, 2.5, N_meta_iter)

def d_log_loss(x, dx):
    return np.sum(x * dx)

def run():
    train_images, train_labels, _, _, _ = load_data()
    train_images = train_images[:N_data, :]
    train_labels = train_labels[:N_data, :]
    batch_idxs = BatchList(N_data, batch_size)

    parser, _, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = len(parser.vect)
    def indexed_loss_fun(w, idxs):
        return loss_fun(w, X=train_images[idxs], T=train_labels[idxs], L2_reg=L2_reg)

    losses = []
    d_losses = []
    for log_alpha_0 in log_stepsizes:
        npr.seed(0)
        V0 = npr.randn(N_weights) * velocity_scale
        alpha_0 = np.exp(log_alpha_0)
        alphas = np.full(N_iters, alpha_0)
        betas = np.full(N_iters, beta_0)
        W0 = npr.randn(N_weights) * np.exp(log_param_scale)
        results = sgd(indexed_loss_fun, batch_idxs, N_iters, W0, V0, alphas, betas)
        losses.append(results['loss_final'])
        d_losses.append(d_log_loss(alpha_0, results['d_alphas']))

    return losses, d_losses

def plot():
    xlabel = "Log step size"

    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font',**{'family':'serif'})

    with open('results.pkl') as f:
        losses, d_losses = pickle.load(f)

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((5,4))
    ax = fig.add_subplot(211)
    ax.plot(log_stepsizes, losses, 'b-', label="loss")
    ax.set_ylim([0, 0.1])
    ax.set_yticks([0.0,])
    #ax.set_xticks([np.min(log_stepsizes), np.max(log_stepsizes)])
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_ylabel("Training loss")
    #ax.set_xlabel(xlabel)
    # ax.legend(loc=4)

    ax = fig.add_subplot(212)
    ax.plot(log_stepsizes, d_losses, 'k-', label="Gradient of loss")
    #ax.set_title("Grad loss vs step size")
    ax.set_ylim([-0.5, 0.5])
    ax.set_ylabel("Gradient")
    ax.set_yticks([0.0,])
    #ax.set_xticks([np.min(log_stepsizes), np.max(log_stepsizes)])
    ax.set_xticks([1.0, 1.5, 2.0])
    #ax.set_xlabel(xlabel)
    #ax.legend(loc=2)
    #plt.savefig("/tmp/fig.png")
    plt.savefig("fig.png")
    plt.savefig('chaos.pdf', pad_inches=0.1, bbox_inches='tight')

if __name__ == '__main__':
    #results = run()
    #with open('results.pkl', 'w') as f:
    #    pickle.dump(results, f)
    plot()
