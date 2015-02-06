"""Gradient descent to optimize regularization per-pixel.
This is a lot like automatic relevance determination."""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pickle

from hypergrad.mnist import load_data_subset
from hypergrad.nn_utils import make_nn_funs, BatchList, plot_mnist_images, WeightsParser
from hypergrad.optimizers import sgd2

# Not going to learn:
velocity_scale = 0.0
log_alpha_0 = 0.0
beta_0 = 0.9
log_param_scale = -4
log_L2_reg_scale = np.log(0.01)

# ----- Discrete training hyper-parameters -----
layer_sizes = [784, 20, 10]
batch_size = 200
N_iters = 50

# ----- Variables for meta-optimization -----
N_train_data = 10000
N_val_data = 10000
N_test_data = 10000
meta_stepsize = 1000
N_meta_iter = 50
meta_L2_reg = 0.01

one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

def run():
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) \
        = load_data_subset(N_train_data, N_val_data, N_test_data)

    batch_idxs = BatchList(N_train_data, batch_size)
    parser, _, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = parser.N

    hyperparser = WeightsParser()
    hyperparser.add_weights('log_L2_reg', (N_weights,))
    metas = np.zeros(hyperparser.N)
    print "Number of hyperparameters to be trained:", hyperparser.N

    npr.seed(0)
    hyperparser.set(metas, 'log_L2_reg', log_L2_reg_scale + np.ones(N_weights))

    def indexed_loss_fun(x, meta_params, idxs):   # To be optimized by SGD.
        L2_reg=np.exp(hyperparser.get(meta_params, 'log_L2_reg'))
        return loss_fun(x, X=train_images[idxs], T=train_labels[idxs], L2_reg=L2_reg)
    def meta_loss_fun(x, meta_params):            # To be optimized in the outer loop.
        L2_reg=np.exp(hyperparser.get(meta_params, 'log_L2_reg'))
        log_prior = -meta_L2_reg * np.dot(L2_reg.ravel(), L2_reg.ravel())
        return loss_fun(x, X=val_images, T=val_labels) - log_prior
    def test_loss_fun(x):                         # To measure actual performance.
        return loss_fun(x, X=test_images, T=test_labels)

    log_alphas = np.full(N_iters, log_alpha_0)
    betas      = np.full(N_iters, beta_0)

    v0 = npr.randn(N_weights) * velocity_scale
    x0 = npr.randn(N_weights) * np.exp(log_param_scale)

    output = []
    for i in range(N_meta_iter):
        results = sgd2(indexed_loss_fun, meta_loss_fun, batch_idxs, N_iters,
                       x0, v0, np.exp(log_alphas), betas, metas)

        learning_curve = results['learning_curve']
        validation_loss = results['M_final']
        test_loss = test_loss_fun(results['x_final'])
        weightparser = parser.new_vect(results['x_final'])
        l2parser = parser.new_vect(np.exp(hyperparser.get(metas, 'log_L2_reg')))
        output.append((learning_curve, validation_loss, test_loss,
                       weightparser[('weights', 0)],
                       l2parser[('weights', 0)]))
        metas -= results['dMd_meta'] * meta_stepsize
        print "Meta iteration {0} Valiation loss {1} Test loss {2}"\
            .format(i, validation_loss, test_loss)
    return output


def plot():
    with open('results.pkl') as f:
        all_learning_curves, all_val_loss, all_test_loss, all_weights, all_L2 = zip(*pickle.load(f))

    fig = plt.figure(0)
    fig.clf()
    N_figs = 2
    ax = fig.add_subplot(N_figs, 1, 1)
    ax.set_title("Learning Curves")
    subsample = np.ceil(float(len(all_learning_curves)) / 50)
    for i, log_alphas in enumerate(all_learning_curves):
        if i % subsample == 0:
            ax.plot(log_alphas, 'o-')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step number")
    #ax.legend(loc=4)

    ax = fig.add_subplot(N_figs, 1, 2)
    ax.set_title("Meta Learning Curve")
    all_train_loss = [curve[-1] for curve in all_learning_curves]
    ax.plot(all_train_loss, 'o-', label="Train Loss")
    ax.plot(all_val_loss, 'o-', label="Validation Loss")
    ax.plot(all_test_loss, 'o-', label="Test Loss")
    #ax.plot(all_L2, 'o-', label="L2_regularization")
    ax.set_ylabel("Validation Loss")
    ax.set_xlabel("Meta Iteration Number")
    ax.legend(loc=2)

    plt.savefig("fig.png")


    fig = plt.figure(0)
    fig.clf()
    N_figs = 1

    ax = fig.add_subplot(N_figs, 1, 1)
    images = all_weights[-1].T
    plot_mnist_images(images, ax, ims_per_row=10)
    fig.set_size_inches((8,12))

    fig.tight_layout()
    plt.savefig("weights.png")
    plt.savefig("weights.pdf", pad_inches=0.05, bbox_inches='tight')

    fig = plt.figure(0)
    fig.clf()
    N_figs = 1

    ax = fig.add_subplot(N_figs, 1, 1)
    images = all_L2[-1].T
    plot_mnist_images(images, ax, ims_per_row=10)
    fig.set_size_inches((8,12))

    fig.tight_layout()
    plt.savefig("penalties.png")
    plt.savefig("penalties.pdf", pad_inches=0.05, bbox_inches='tight')


if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f: pickle.dump(results, f)
    plot()
