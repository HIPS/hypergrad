"""Gradient descent to optimize dataset inputs to a neural network."""
import numpy as np
import numpy.random as npr
import pickle
import matplotlib

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
N_iters = 10

# ----- Variable for this run -----
data_stepsize = 0.04
N_meta_iter = 50

def run():
    train_images, train_labels, _, _, _ = load_data(normalize=True)
    train_images = train_images[:N_real_data, :]
    train_labels = train_labels[:N_real_data, :]
    batch_idxs = BatchList(N_fake_data, batch_size)
    parser, _, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg, return_parser=True)
    N_weights = parser.N

    #fake_data = npr.randn(*(train_images[:N_fake_data, :].shape))
    fake_data = np.zeros(train_images[:N_fake_data, :].shape)
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    fake_labels = one_hot(np.array(range(0, 10)), 10)  # One of each label.

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
                       x0, v0, np.exp(log_alphas), betas, fake_data)

        learning_curve = results['learning_curve']
        output.append((learning_curve, fake_data))
        fake_data -= results['dMd_meta'] * data_stepsize   # Update data with one gradient step.

    return output


def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        all_learning_curves, all_fakedata = zip(*pickle.load(f))

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((8,12))

    N_figs = 2

    ax = fig.add_subplot(N_figs, 1, 1)
    ax.set_title("Learning Curve")
    for i, log_alphas in enumerate(all_learning_curves):
        ax.plot(log_alphas, 'o-')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step number")
    #ax.legend(loc=4)

    ax = fig.add_subplot(N_figs, 1, 2)
    ax.set_title("Fake Data")
    images = all_fakedata[-1]
    concat_images = np.zeros((28, 0))
    for i in range(N_fake_data):
        cur_image = np.reshape(images[i, :], (28, 28))
        concat_images = np.concatenate((concat_images, cur_image, np.zeros((28, 5))), axis=1)
    ax.matshow(concat_images, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

    plt.savefig("/tmp/fig.png")
    plt.savefig("fig.png")




if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f: pickle.dump(results, f)
    plot()
