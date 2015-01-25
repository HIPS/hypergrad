"""Gradient descent to optimize everything"""
"""Aiming for smooth curves by running for a long time with small steps."""
import numpy as np
import numpy.random as npr
import pickle
from functools import partial
import itertools as it
import socket

from hypergrad.data import load_data_subset
from hypergrad.nn_utils import make_nn_funs, BatchList, VectorParser, logit, inv_logit
from hypergrad.optimizers import sgd3, rms_prop
from hypergrad.odyssey import omap, collect_results

# ----- Fixed params -----
layer_sizes = [784, 200, 10]
batch_size = 200
N_epochs = 5
N_classes = 10
N_train = 10**3
N_valid = 10**3
N_tests = 10**3
N_meta_thin = 5
# ----- Superparameters -----
meta_alpha = 0.05
meta_gamma = 0.0 # Setting this to zero makes things much more stable
N_meta_iter = 200
# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = -6.0
init_log_alphas = 0.0
init_invlogit_betas = inv_logit(0.9)
init_log_param_scale = -3.0

def d_logit(x):
    return logit(x) * (1 - logit(x))

def run():
    (train_images, train_labels),\
    (valid_images, valid_labels),\
    (tests_images, tests_labels) = load_data_subset(N_train, N_valid, N_tests)
    batch_idxs = BatchList(N_train, batch_size)
    N_iters = N_epochs * len(batch_idxs)
    parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weight_types = len(parser.names)
    hyperparams = VectorParser()
    hyperparams['log_L2_reg']      = np.full(N_weight_types, init_log_L2_reg)
    hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
    hyperparams['log_alphas']      = np.full(N_iters, init_log_alphas)
    hyperparams['invlogit_betas']  = np.full(N_iters, init_invlogit_betas)

    def train_loss_fun(w, log_L2_reg=0.0):
        return loss_fun(w, X=train_images, T=train_labels)

    def valid_loss_fun(w, log_L2_reg=0.0):
        return loss_fun(w, X=valid_images, T=valid_labels)

    def tests_loss_fun(w, log_L2_reg=0.0):
        return loss_fun(w, X=tests_images, T=tests_labels)

    all_learning_curves = []
    all_x = []
    def hyperloss_grad(hyperparam_vect, ii):
        learning_curve = []
        def callback(x, i):
            if i % len(batch_idxs) == 0:
                learning_curve.append(loss_fun(x, X=train_images, T=train_labels))

        def indexed_loss_fun(w, log_L2_reg, j):
            # idxs = batch_idxs[i % len(batch_idxs)]
            npr.seed(1000 * ii + j)
            idxs = npr.randint(N_train, size=len(batch_idxs))
            partial_vects = [np.full(parser[name].size, np.exp(log_L2_reg[i]))
                             for i, name in enumerate(parser.names)]
            L2_reg_vect = np.concatenate(partial_vects, axis=0)
            return loss_fun(w, X=train_images[idxs], T=train_labels[idxs], L2_reg=L2_reg_vect)

        npr.seed(ii)
        N_weights = parser.vect.size
        V0 = np.zeros(N_weights)

        cur_hyperparams = hyperparams.new_vect(hyperparam_vect)
        layer_param_scale = [np.full(parser[name].size, 
                                     np.exp(cur_hyperparams['log_param_scale'][i]))
                             for i, name in enumerate(parser.names)]
        W0 = npr.randn(N_weights) * np.concatenate(layer_param_scale, axis=0)
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas = logit(cur_hyperparams['invlogit_betas'])
        log_L2_reg = cur_hyperparams['log_L2_reg']
        results = sgd3(indexed_loss_fun, valid_loss_fun, W0, V0,
                       alphas, betas, log_L2_reg, callback=callback)
        hypergrads = hyperparams.copy()
        hypergrads['log_L2_reg']      = results['dMd_meta']
        weights_grad = parser.new_vect(W0 * results['dMd_x'])
        hypergrads['log_param_scale'] = [np.sum(weights_grad[name])
                                         for name in parser.names]
        hypergrads['log_alphas']      = results['dMd_alphas'] * alphas
        hypergrads['invlogit_betas']  = (results['dMd_betas'] *
                                         d_logit(cur_hyperparams['invlogit_betas']))
        all_x.append(results['x_final'])
        all_learning_curves.append(learning_curve)
        return hypergrads.vect

    add_fields = ['train_loss', 'valid_loss', 'tests_loss', 'iter_num']
    meta_results = {field : [] for field in add_fields + hyperparams.names}
    def meta_callback(hyperparam_vect, i):
        if i % N_meta_thin == 0:
            print "Meta iter {0}".format(i)
            x = all_x[-1]
            cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
            log_L2_reg = cur_hyperparams['log_L2_reg']
            for field in cur_hyperparams.names:
                meta_results[field].append(cur_hyperparams[field])

            meta_results['train_loss'].append(train_loss_fun(x))
            meta_results['valid_loss'].append(valid_loss_fun(x))
            meta_results['tests_loss'].append(tests_loss_fun(x))
            meta_results['iter_num'].append(i)

    final_result = rms_prop(hyperloss_grad, hyperparams.vect,
                            meta_callback, N_meta_iter, meta_alpha, meta_gamma)
    meta_results['all_learning_curves'] = all_learning_curves
    parser.vect = None # No need to pickle zeros
    return meta_results, parser

def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser = pickle.load(f)

    fig = plt.figure(0)
    fig.set_size_inches((6,8))
    # ----- Primal learning curves -----
    ax = fig.add_subplot(211)
    ax.set_title('Primal learning curves')
    for i, y in zip(results['iter_num'], results['all_learning_curves']):
        ax.plot(y, 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Negative log prob')
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(212)
    ax.set_title('Meta learning curves')
    losses = ['train_loss', 'valid_loss', 'tests_loss']
    for loss_type in losses:
        ax.plot(results[loss_type], 'o-', label=loss_type)
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Negative log prob')
    ax.set_ylim([0.6, 1.2])
    # ax.legend(loc=1, frameon=False)
    plt.savefig('learning_curves.png')

    # ----- Alpha and beta schedules -----
    fig.clf()
    ax = fig.add_subplot(211)
    ax.set_title('Alpha learning curves')
    for i, y in zip(results['iter_num'], results['log_alphas']):
        ax.plot(y, 'o-', label="Meta iter {0}".format(i))
    ax.set_xlabel('Primal iter number')
    ax.set_ylabel('Log alpha')
    # ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(212)
    ax.set_title('Beta learning curves')
    for y in results['invlogit_betas']:
        ax.plot(y, 'o-')
    ax.set_xlabel('Primal iter number')
    ax.set_ylabel('Inv logit beta')
    plt.savefig('alpha_beta_curves.png')

    # ----- Init scale and L2 reg -----
    fig.clf()
    ax = fig.add_subplot(211)
    ax.set_title('Init scale learning curves')
    for i, y in enumerate(zip(*results['log_param_scale'])):
        y = np.array(y) + 0.01 * i # Just distinguish overlapping lines
        ax.plot(y, 'o-', label=parser.names[i])
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Log param scale')
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(212)
    ax.set_title('L2 reg learning curves')
    for i, y in enumerate(zip(*results['log_L2_reg'])):
        y = np.array(y) + 0.01 * i # Just distinguish overlapping lines
        ax.plot(y, 'o-', label=parser.names[i])
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Log L2 reg')
    plt.savefig('scale_and_reg.png')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()
