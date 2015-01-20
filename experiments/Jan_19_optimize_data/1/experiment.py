"""Gradient descent to optimize dataset inputs to a neural network."""
import numpy as np
import numpy.random as npr
import pickle

from hypergrad.data import load_data
from hypergrad.nn_utils import make_nn_funs, BatchList
from hypergrad.optimizers import sgd2

layer_sizes = [784, 10]
L2_reg = 0.1
velocity_scale = 0.0
log_alpha_0 = 0.0
beta_0 = 0.9
log_param_scale = -2
N_real_data = 1000
N_fake_data = 10
batch_size = 10
N_iters = 20

# ----- Variable for this run -----
data_stepsize = 0.004
N_meta_iter = 40

def run():
    train_images, train_labels, _, _, _ = load_data(normalize=True)
    train_images = train_images[:N_real_data, :]
    train_labels = train_labels[:N_real_data, :]
    batch_idxs = BatchList(N_fake_data, batch_size)
    parser, _, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg, return_parser=True)
    N_weights = parser.N

    fake_data = npr.zeros(train_images[:N_fake_data, :].shape)
    fake_labels = np.range(0, 10)  # One of each label.

    def indexed_loss_fun(x, meta_params, idxs):   # To be optimized by SGD.
        return loss_fun(x, X=meta_params[idxs], T=fake_labels[idxs])
    def meta_loss_fun(x):                         # To be optimized in the outer loop.
        return loss_fun(x, X=train_images, T=train_labels)
    log_alphas = np.full(N_iters, log_alpha_0)
    betas      = np.full(N_iters, beta_0)
    npr.seed(0)
    v0 = npr.randn(N_weights) * velocity_scale
    x0 = npr.randn(N_weights) * np.exp(log_param_scale)

    output = []
    for i in range(N_meta_iter):
        print "Meta iteration {0}".format(i)
        results = sgd2(indexed_loss_fun, meta_loss_fun, batch_idxs, N_iters,
                       x0, v0, np.exp(log_alphas), betas)

        learning_curve = results['learning_curve']
        output.append((learning_curve, fake_data))
        fake_data += results['dMd_meta'] * data_stepsize   # Update data with one gradient step.

    return output


def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        all_learning_curves, all_bindicts = zip(*pickle.load(f))

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((8,12))

    N_figs = len(all_bindicts[0]) + 1

    ax = fig.add_subplot(N_figs, 1, 1)
    ax.set_title("Learning Curve")
    for i, log_alphas in enumerate(all_learning_curves):
        ax.plot(log_alphas, 'o-')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step number")
    #ax.legend(loc=4)


    for i, layer in enumerate(all_bindicts[0]):
        ax = fig.add_subplot(N_figs, 1, i + 2)
        ax.set_title("Weight initialization PDF " + layer.__repr__() )
        ax.set_ylabel("PDF Height")
        #ax.set_xlabel("Weight range")

        for bindict in all_bindicts:
            bins = bindict[layer]
            x = np.linspace( np.min(bins), np.max(bins), 1000)
            ax.plot(x, binpdf(x, bins), '-', label="{0} meta iters".format(i))

    plt.savefig("/tmp/fig.png")
    plt.savefig("fig.png")

    plt.show()

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f: pickle.dump(results, f)
    plot()
