"""Gradient descent to optimize dataset inputs to a neural network,
   as well as the regularization and learning parameters."""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pickle

from hypergrad.data import load_data
from hypergrad.nn_utils import make_nn_funs, BatchList, plot_mnist_images, WeightsParser
from hypergrad.optimizers import sgd2



# ----- Initial values of continuous hyper-parameters -----
init_log_L2_reg = np.log(0.01)

# Not going to learn:
velocity_scale = 0.0
log_alpha_0 = 0.0
beta_0 = 0.9
log_param_scale = -4

# ----- Discrete training hyper-parameters -----
layer_sizes = [784, 10]
batch_size = 10
N_iters = 20
N_classes = 10

# ----- Variables for meta-optimization -----
N_fake_data = 10
fake_data_L2_reg = 0.0
N_val_data = 10000
N_test_data = 1000
meta_stepsize = 1
N_meta_iter = 40
init_fake_data_scale = 0.01


def run():
    val_images, val_labels, test_images, test_labels, _ = load_data(normalize=True)
    val_images = val_images[:N_val_data, :]
    val_labels = val_labels[:N_val_data, :]
    true_data_scale = np.std(val_images)

    test_images = test_images[:N_test_data, :]
    test_labels = test_labels[:N_test_data, :]
    batch_idxs = BatchList(N_fake_data, batch_size)
    parser, _, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = parser.N

    npr.seed(0)
    init_fake_data = npr.randn(*(val_images[:N_fake_data, :].shape)) * init_fake_data_scale
    fake_labels = one_hot(np.array(range(N_fake_data)) % N_classes, N_classes)  # One of each.

    hyperparser = WeightsParser()
    hyperparser.add_weights('log_L2_reg', (1,))
    hyperparser.add_weights('fake_data', init_fake_data.shape)
    metas = np.zeros(hyperparser.N)
    print "Number of hyperparameters to be trained:", hyperparser.N
    hyperparser.set(metas, 'log_L2_reg', init_log_L2_reg)
    hyperparser.set(metas, 'fake_data', init_fake_data)

    def indexed_loss_fun(x, meta_params, idxs):   # To be optimized by SGD.
        L2_reg=np.exp(hyperparser.get(meta_params, 'log_L2_reg')[0])
        fake_data=hyperparser.get(meta_params, 'fake_data')
        return loss_fun(x, X=fake_data[idxs], T=fake_labels[idxs], L2_reg=L2_reg)
    def meta_loss_fun(x, meta_params):            # To be optimized in the outer loop.
        fake_data=hyperparser.get(meta_params, 'fake_data')
        log_prior = -fake_data_L2_reg * np.dot(fake_data.ravel(), fake_data.ravel())
        return loss_fun(x, X=val_images, T=val_labels) - log_prior
    def test_loss_fun(x):                         # To measure actual performance.
        return loss_fun(x, X=test_images, T=test_labels)

    log_alphas = np.full(N_iters, log_alpha_0)
    betas      = np.full(N_iters, beta_0)

    output = []
    for i in range(N_meta_iter):
        print "L2 reg is ", np.exp(hyperparser.get(metas, 'log_L2_reg')[0]), "| ",

        npr.seed(0)
        v0 = npr.randn(N_weights) * velocity_scale
        x0 = npr.randn(N_weights) * np.exp(log_param_scale)

        results = sgd2(indexed_loss_fun, meta_loss_fun, batch_idxs, N_iters,
                       x0, v0, np.exp(log_alphas), betas, metas)

        learning_curve = results['learning_curve']
        validation_loss = results['M_final']
        fake_data_scale = np.std(hyperparser.get(metas, 'fake_data')) / true_data_scale
        test_loss = test_loss_fun(results['x_final'])
        output.append((learning_curve, validation_loss, test_loss,
                       hyperparser.get(metas, 'fake_data'), fake_data_scale,
                       np.exp(hyperparser.get(metas, 'log_L2_reg')[0])))

        metas -= results['dMd_meta'] * meta_stepsize
        print "Meta iteration {0} Valiation loss {1} Test loss {2}"\
            .format(i, validation_loss, test_loss)
    return output


def plot():
    with open('results.pkl') as f:
        all_learning_curves, all_val_loss, all_test_loss,\
        all_fakedata, all_fakedatasize, all_L2 = zip(*pickle.load(f))


    # Fake data
    fig = plt.figure(0)
    fig.clf()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    images = all_fakedata[-1]
    plot_mnist_images(images, ax, ims_per_row=5, padding=2)
    plt.savefig('fake_data.pdf', bbox_inches='tight')

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
    ax.plot(all_L2, 'o-', label="L2_regularization")
    ax.set_ylabel("Validation Loss")
    ax.set_xlabel("Meta Iteration Number")
    ax.legend(loc=2)

    ax = fig.add_subplot(N_figs, 1, 3)
    ax.set_title("Fake Data")
    images = all_fakedata[-1]
    plot_mnist_images(images, ax, ims_per_row=10)
    fig.set_size_inches((8,12))
    plt.savefig("fig.png")




if __name__ == '__main__':
    #results = run()
    #with open('results.pkl', 'w') as f: pickle.dump(results, f)
    plot()
