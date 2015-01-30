import numpy as np
import pickle
from collections import defaultdict
from functools import partial
from funkyyak import grad, kylist, getval
import itertools as it

import hypergrad.omniglot as omniglot
from hypergrad.nn_utils import make_nn_funs, VectorParser
from hypergrad.optimizers import sgd_meta_only as sgd, rms_prop
from hypergrad.util import RandomState, dictslice

# ----- Fixed params -----
layer_sizes = [784, 100, 55]
batch_size = 200
N_iters = 100
alpha = 0.1
beta = 0.9
seed = 1
N_test_alphabets = 20
N_valid_dpts = 100
N_alphabets_eval = 6
# ----- Superparameters -----
meta_alpha = 0.01
N_meta_iter = 301
N_hyper_thin = 50
initialization_scale = 0.1
# ----- Initial values of learned hyper-parameters -----
log_scale_init = 2.0
offset_init_std = 0.1

def run():
    RS = RandomState((seed, "top_rs"))
    all_alphabets = omniglot.load_data()
    RS.shuffle(all_alphabets)
    train_alphabets = all_alphabets[:-N_test_alphabets]
    tests_alphabets = all_alphabets[-N_test_alphabets:]
    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size
    hyperparams_0 = VectorParser()
    hyperparams_0['log_scale']  = log_scale_init * np.ones(N_weights)
    hyperparams_0['offset'] = offset_init_std * RS.randn(N_weights)

    def reg_loss_fun(W, data, hyperparam_vect, reg_penalty):
        hyperparams = hyperparams_0.new_vect(hyperparam_vect)
        Z = np.exp(hyperparams['log_scale']) * W + hyperparams['offset']
        return loss_fun(Z, **data) + np.dot(W, W) * reg_penalty

    def hyperloss(hyperparam_vect, i_hyper, alphabets, verbose=True, report_train_loss=False):
        RS = RandomState((seed, i_hyper, "hyperloss"))        
        alphabet = shuffle_alphabet(RS.choice(alphabets), RS)
        N_train = alphabet['X'].shape[0] - N_valid_dpts
        train_data = dictslice(alphabet, slice(None, N_train))
        if report_train_loss:
            valid_data = dictslice(alphabet, slice(None, N_valid_dpts))
        else:
            valid_data = dictslice(alphabet, slice(N_train, None))
        def primal_loss(W, hyperparam_vect, i_primal, reg_penalty=True):
            RS = RandomState((seed, i_hyper, i_primal))
            idxs = RS.permutation(N_train)[:batch_size]
            minibatch = dictslice(train_data, idxs)
            loss = reg_loss_fun(W, minibatch, hyperparam_vect, reg_penalty)
            if verbose and i_primal % 10 == 0: print "Iter {0}, loss, {1}".format(i_primal, getval(loss))
            return loss

        W0 = RS.randn(N_weights) * initialization_scale
        W_final = sgd(grad(primal_loss), hyperparam_vect, W0, alpha, beta, N_iters, callback=None)
        return reg_loss_fun(W_final, valid_data, hyperparam_vect, reg_penalty=False)

    results = defaultdict(list)
    def record_results(hyperparam_vect, i_hyper, g):
        print "Meta iter {0}. Recording results".format(i_hyper)
        # RS = RandomState((seed, i_hyper, "evaluation"))
        def loss_fun(alphabets, report_train_loss):
            RS = RandomState((seed, "evaluation")) # Same alphabet with i_hyper now
            return np.mean([hyperloss(hyperparam_vect, RS.int32(), alphabets=alphabets,
                                      verbose=False, report_train_loss=report_train_loss)
                            for i in range(N_alphabets_eval)])
        cur_hyperparams = hyperparams_0.new_vect(hyperparam_vect.copy())
        if i_hyper % N_hyper_thin == 0:
            # Storing O(N_weights) is a bit expensive so we thin it out and store in low precision
            for field in cur_hyperparams.names:
                results[field].append(cur_hyperparams[field].astype(np.float16))
        results['train_loss'].append(loss_fun(train_alphabets, report_train_loss=True))
        results['valid_loss'].append(loss_fun(train_alphabets, report_train_loss=False))
        results['tests_loss'].append(loss_fun(tests_alphabets, report_train_loss=False))
        print "Train:", results['train_loss']
        print "Valid:", results['valid_loss']
        print "Tests:", results['tests_loss']

    train_hyperloss = partial(hyperloss, alphabets=train_alphabets)
    rms_prop(grad(train_hyperloss), hyperparams_0.vect, record_results, N_meta_iter, meta_alpha, gamma=0)
    return results

def shuffle_alphabet(alphabet, RS):
    # Shuffles both data and label indices
    N_rows, N_cols = alphabet['T'].shape
    alphabet['T'] = alphabet['T'][:, RS.permutation(N_cols)]
    return dictslice(alphabet, RS.permutation(N_rows))

def ceil_div(a, b):
    return a / b + ((a % b) > 0)

def plot():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    def plot_filters(ax, parser, lims, N_cols=10, L_img=28, padding=2):
        bg_val = 0
        filters = parser[('weights', 0)]
        output_weights = parser[('weights', 1)]
        N_outputs = output_weights.shape[1]
        N_filters = filters.shape[1]
        N_rows = ceil_div(N_filters, N_cols)
        L_extra = ceil_div(N_outputs, L_img)
        output_weights_padded = np.full((N_filters, L_img * L_extra), bg_val)
        output_weights_padded[:, :N_outputs] = output_weights
        output_weights_padded = output_weights_padded.reshape((N_filters, L_extra, L_img))
        filters = filters.reshape((L_img, L_img, N_filters))
        row_height = L_img + L_extra + padding * 2
        col_width  = L_img + padding
        image = np.full((row_height * N_rows, col_width * N_cols), bg_val)
        def pix_range_x(i):
            offset = i * col_width
            return slice(offset, offset + L_img)
        def pix_range_y(i):
            offset = i * row_height
            return slice(offset, offset + L_img + L_extra + padding)
        for i_x, i_y in it.product(range(N_rows), range(N_cols)):
            i_filter = i_x + i_y * N_cols
            if i_filter < N_filters:
                cur_frame = np.concatenate((filters[:, :, i_filter],
                                            np.full((padding, L_img), bg_val),
                                            output_weights_padded[i, :, :]), axis=0)
                image[pix_range_y(i_y), pix_range_x(i_x)] = cur_frame

        img_min, img_max = lims
        image = (image - img_min) / (img_max - img_min)
        image = np.minimum(np.maximum(image, 0.0), 1.0)
        ax.imshow(image, cmap = mpl.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])

    with open('results.pkl') as f:
         results = pickle.load(f)

    fig = plt.figure(0)
    fig.set_size_inches((6,4))
    ax = fig.add_subplot(111)
    ax.set_title('Meta learning curves')
    losses = ['train_loss', 'valid_loss', 'tests_loss']
    for loss_type in losses:
        ax.plot(results[loss_type], 'o-', label=loss_type)
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Negative log prob')
    ax.legend(loc=1, frameon=False)
    plt.savefig('learning_curves.png')

    fig.clf()
    fig.set_size_inches((6,8))
    ax = fig.add_subplot(211)
    ax.set_title('Parameter scale')
    for i, log_scale in enumerate(results['log_scale']):
        ax.plot(np.sort(log_scale), label = "Meta iter {0}".format(i * N_hyper_thin))
    ax.legend(loc=2, frameon=False)

    ax = fig.add_subplot(212)
    ax.set_title('Parameter offset')
    for i, offset in enumerate(results['offset']):
        ax.plot(np.sort(offset), label = "Meta iter {0}".format(i * N_hyper_thin))
    plt.savefig('Learned regularization.png')

    w_parser, _, _, _ = make_nn_funs(layer_sizes)
    log_scales = w_parser.new_vect(np.exp(results['log_scale'])[-1])
    offset     = w_parser.new_vect(results['offset'][-1])
    fig.clf()
    fig.set_size_inches((6,6))
    ax = fig.add_subplot(111)
    plot_filters(ax, log_scales, [5, 10])
    ax.set_title("Scales")
    plt.savefig("L2_scale_filters.png")
    plt.savefig("L2_scale_filters.pdf")

    fig.clf()
    fig.set_size_inches((6,6))
    ax = fig.add_subplot(111)
    plot_filters(ax, offset, [-1, 1])
    ax.set_title("Offsets")
    plt.savefig("L2_mean_filters.png")
    plt.savefig("L2_mean_filters.pdf")

if __name__ == '__main__':
    # results = run()
    # with open('results.pkl', 'w') as f:
    #     pickle.dump(results, f, 1)
    plot()
