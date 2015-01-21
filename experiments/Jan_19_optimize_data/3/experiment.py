"""Gradient descent to optimize dataset inputs to a neural network."""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pickle

from hypergrad.data import load_data, one_hot
from hypergrad.nn_utils import make_nn_funs, BatchList, plot_mnist_images
from hypergrad.optimizers import sgd2

# ----- Variables for regular optimization -----
layer_sizes = [784, 10]
L2_reg = 0.01
velocity_scale = 0.0
log_alpha_0 = 0.0
beta_0 = 0.9
log_param_scale = -4
batch_size = 10
N_iters = 40
N_classes = 10

# ----- Variables for meta-optimization -----
N_fake_data = 10
fake_data_L2_reg = 0.0
N_val_data = 10000
N_test_data = 1000
data_stepsize = 1
N_meta_iter = 20
init_fake_data_scale = 0.01


def run():
    val_images, val_labels, test_images, test_labels, _ = load_data(normalize=True)
    val_images = val_images[:N_val_data, :]
    val_labels = val_labels[:N_val_data, :]
    truedatasize = np.std(val_images)

    test_images = test_images[:N_test_data, :]
    test_labels = test_labels[:N_test_data, :]
    batch_idxs = BatchList(N_fake_data, batch_size)
    parser, _, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg, return_parser=True)
    N_weights = parser.N

    fake_data = npr.randn(*(val_images[:N_fake_data, :].shape)) * init_fake_data_scale
    fake_labels = one_hot(np.array(range(N_fake_data)) % N_classes, N_classes)  # One of each.

    def indexed_loss_fun(x, meta_params, idxs):   # To be optimized by SGD.
        return loss_fun(x, X=meta_params[idxs], T=fake_labels[idxs])
    def meta_loss_fun(x, meta_params):                         # To be optimized in the outer loop.
        log_prior = -fake_data_L2_reg * np.dot(meta_params.ravel(), meta_params.ravel())
        return loss_fun(x, X=val_images, T=val_labels) - log_prior
    def test_loss_fun(x):                         # To measure actual performance.
        return loss_fun(x, X=test_images, T=test_labels)
    log_alphas = np.full(N_iters, log_alpha_0)
    betas      = np.full(N_iters, beta_0)
    npr.seed(0)
    v0 = npr.randn(N_weights) * velocity_scale
    x0 = npr.randn(N_weights) * np.exp(log_param_scale)

    output = []
    for i in range(N_meta_iter):
        results = sgd2(indexed_loss_fun, meta_loss_fun, batch_idxs, N_iters,
                       x0, v0, np.exp(log_alphas), betas, fake_data)
        learning_curve = results['learning_curve']
        validation_loss = results['M_final']
        fakedatasize = np.std(fake_data) / truedatasize
        test_loss = test_loss_fun(results['x_final'])
        output.append((learning_curve, validation_loss, test_loss, fake_data, fakedatasize))
        fake_data -= results['dMd_meta'] * data_stepsize   # Update data with one gradient step.
        print "Meta iteration {0} Valiation loss {1} Test loss {2}"\
            .format(i, validation_loss, test_loss)
    return output


def plot():
    with open('results.pkl') as f:
        all_learning_curves, all_val_loss, all_test_loss,\
        all_fakedata, all_fakedatasize = zip(*pickle.load(f))

    fig = plt.figure(0)
    fig.clf()
    N_figs = 3
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
    ax.plot(all_fakedatasize, 'o-', label="Fake Data Scale")
    ax.set_ylabel("Validation Loss")
    ax.set_xlabel("Meta Iteration Number")
    ax.legend(loc=2)

    ax = fig.add_subplot(N_figs, 1, 3)
    ax.set_title("Fake Data")
    images = all_fakedata[-1]
    plot_mnist_images(images, ax, ims_per_row=10)
    fig.set_size_inches((8,12))

    plt.savefig("/tmp/fig.png")
    plt.savefig("fig.png")

    plt.show()


if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f: pickle.dump(results, f)
    plot()
