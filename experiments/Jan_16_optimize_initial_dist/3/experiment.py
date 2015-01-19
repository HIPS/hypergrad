"""Gradient descent to optimize initial weight distribution,
with a separate weight distribution per layer."""
import numpy as np
import numpy.random as npr
import pickle

from hypergrad.data import load_data
from hypergrad.nn_utils import make_nn_funs, BatchList
from hypergrad.optimizers import sgd

layer_sizes = [784, 10]
L2_reg = 0.0
velocity_scale = 0.0
batch_size = 250
log_alpha_0 = 0.0
beta_0 = 0.9
num_meta_iters = 1
log_param_scale = -2
N_data = 10**3
N_iters = 20

# ----- Variable for this run -----
N_bins = 20
bin_stepsize = 0.01
N_meta_iter = 100

def run():
    train_images, train_labels, _, _, _ = load_data()
    train_images = train_images[:N_data, :]
    train_labels = train_labels[:N_data, :]
    batch_idxs = BatchList(N_data, batch_size)
    iter_per_epoch = len(batch_idxs)
    parser, _, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg, return_parser=True)
    N_weights = parser.N
    def indexed_loss_fun(w, idxs):
        return loss_fun(w, X=train_images[idxs], T=train_labels[idxs])
    log_alphas = np.full(N_iters, log_alpha_0)
    betas      = np.full(N_iters, beta_0)
    npr.seed(2)
    V0 = npr.randn(N_weights) * velocity_scale
    #W0 = npr.randn(N_weights) * np.exp(log_param_scale)
    X_uniform = npr.rand(N_weights)  # Weights are uniform passed through an inverse cdf.
    bindict = {k : np.linspace(-1,1,N_bins) * np.exp(log_param_scale)  # Different cdf per layer.
                   for k, v in parser.idxs_and_shapes.iteritems()}
    output = []
    for i in range(N_meta_iter):
        print "Meta iteration {0}".format(i)
        #X0, dX_dbins = bininvcdf(W_uniform, bins)
        X0 = np.zeros(N_weights)
        dX_dbins = {}
        for k, cur_bins in bindict.iteritems():
            cur_slice, cur_shape = parser.idxs_and_shapes[k]
            cur_xs = X_uniform[cur_slice]
            cur_X0, cur_dX_dbins = bininvcdf(cur_xs, cur_bins)
            X0[cur_slice] = cur_X0
            dX_dbins[k] = cur_dX_dbins
        results = sgd(indexed_loss_fun, batch_idxs, N_iters,
                      X0, V0, np.exp(log_alphas), betas, record_learning_curve=True)
        dL_dx = results['d_x']

        learning_curve = results['learning_curve']
        output.append((learning_curve, bindict))

        # Update bins with one gradient step.
        for k, bins in bindict.iteritems():
            dL_dbins = np.dot(parser.get(dL_dx, k).flatten(), dX_dbins[k])
            bins = bins - dL_dbins * bin_stepsize
            bins[[0,-1]] = bins[[0,-1]] - dL_dbins[[0,1]] * bin_stepsize
            bins.sort()  # Sort in place.
            bindict[k] = bins

    return output

def bininvcdf(x, bins):
    """Returns height of inverse cdf, and gradients of output w.r.t. bin heights.
       this is a matrix, rows are different inputs, columns are different bins."""
    assert np.all(np.sort(bins) == bins)  # Bin heights must be sorted.
    edges = np.linspace(0, 1, len(bins))
    dy_dbin = np.zeros((len(x), len(bins)))
    y = np.zeros(len(x))
    for ix, (left, right, bottom, top) \
            in enumerate(zip(edges[:-1], edges[1:], bins[:-1], bins[1:])):
        cur = np.where((left <= x) * (x < right))
        frac = (x[cur] - left) / (right - left)
        y[cur] = bottom + frac  * (top - bottom)
        dy_dbin[cur, ix    ] = 1 - frac
        dy_dbin[cur, ix + 1] = frac
    return y, dy_dbin

    # Divide points into
    return [np.mean(x[i:(i+L)]) for i in range(0, len(x), L)]

def test_bininvcdf():
    x = npr.rand(10)
    bins = np.array([-3.0, -1.0, 5.0, 6.0])
    _, dy_dbin = bininvcdf(x, bins)
    eps = 1e-3
    for i in range(len(bins)):
        dbin = np.zeros(len(bins))
        dbin[i] = eps / 2
        y_1, _ = bininvcdf(x, bins - dbin)
        y_2, _ = bininvcdf(x, bins + dbin)
        nd = (y_2 - y_1) / eps
        assert np.allclose(nd, dy_dbin[:, i])

def binpdf(x, bins):
    """x is a set of locations at which to evaluate the pdf."""
    assert np.all(np.sort(bins) == bins)  # Bin heights must be sorted.
    edges = np.linspace(0, 1, len(bins))
    y = np.zeros(len(x))
    for ix, (left, right, bottom, top) \
            in enumerate(zip(edges[:-1], edges[1:], bins[:-1], bins[1:])):
        cur = np.where((bottom <= x) * (x < top))
        y[cur] = (right - left) / (top - bottom)
    return y

def smoothbins(x, L):
    return [np.mean(x[i:(i+L)]) for i in range(0, len(x), L)]

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

    #ax = fig.add_subplot(N_figs, 1, 2)
    #ax.set_title("Inverse CDF Height")
    #for i, bins in enumerate(all_bins):
    #    ax.plot(np.linspace(0, 1, len(bins)), bins, 'o-', label="{0} meta iters".format(i))
    #ax.set_ylabel("Inverse CDF Height")
    #ax.set_xlabel("Uniform")
    #ax.legend(loc=4)

    #for i, bindict in enumerate(all_bindicts):
    for i, (k, bins) in enumerate(all_bindicts[-1].iteritems()):
        ax = fig.add_subplot(N_figs, 1, i + 2)
        ax.set_title("Weight initialization PDF " + k.__repr__() )
        x = np.linspace( np.min(bins), np.max(bins), 1000)
        ax.plot(x, binpdf(x, bins), '-', label="{0} meta iters".format(i))
        ax.set_ylabel("PDF Height")
        ax.set_xlabel("Weight range")

    plt.savefig("/tmp/fig.png")
    plt.savefig("fig.png")

    plt.show()

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f: pickle.dump(results, f)
    plot()
