"""Gradient descent to optimize everything"""
"""Now just seeing if we can clean this up a bit"""
import numpy as np
import numpy.random as npr
import pickle
from collections import defaultdict

from funkyyak import grad, kylist, getval

from hypergrad.util import memoize
from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, BatchList, VectorParser, logit, inv_logit
from hypergrad.optimizers import sgd4, rms_prop

# ----- Fixed params -----
layer_sizes = [784, 10]
batch_size = 200
N_epochs = 10
N_classes = 10
N_train = 10**3
N_valid = 10**3
N_tests = 10**3
N_batches = N_train / batch_size
N_iters = N_epochs * N_batches
# ----- Superparameters -----
meta_alpha = 0.1
N_meta_iter = 20
# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = 0.0
init_log_alphas = 0.0
init_invlogit_betas = inv_logit(0.9)
init_log_param_scale = 0.0

def fill_parser(parser, items):
    partial_vects = [np.full(parser[name].size, items[i])
                     for i, name in enumerate(parser.names)]
    return np.concatenate(partial_vects, axis=0)

def run():
    train_data, valid_data, tests_data = load_data_dicts(N_train, N_valid, N_tests)
    parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weight_types = len(parser.names)
    hyperparams = VectorParser()
    hyperparams['log_L2_reg']      = np.full(N_weight_types, init_log_L2_reg)
    hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
    hyperparams['log_alphas']      = np.full(N_iters, init_log_alphas)
    hyperparams['invlogit_betas']  = np.full(N_iters, init_invlogit_betas)

    def primal_optimizer(hyperparam_vect, i_hyper):
        def indexed_loss_fun(w, L2_vect, i_iter):
            seed = i_hyper * 10**6 + i_iter
            idxs = npr.RandomState(seed).randint(N_train, size=batch_size)
            return loss_fun(w, train_data['X'][idxs], train_data['T'][idxs], L2_vect)

        learning_curve = []
        def callback(x, i_iter):
            if i_iter % N_batches == 0:
                learning_curve.append(loss_fun(x, **train_data))

        cur_hyperparams = hyperparams.new_vect(hyperparam_vect)
        W0 = fill_parser(parser, np.exp(cur_hyperparams['log_param_scale']))
        W0 *= npr.RandomState(i_hyper).randn(W0.size)
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas  = logit(cur_hyperparams['invlogit_betas'])
        L2_reg = fill_parser(parser, np.exp(cur_hyperparams['log_L2_reg']))
        V0 = np.zeros(W0.size)
        W_opt = sgd4(grad(indexed_loss_fun), kylist(W0, alphas, betas, L2_reg), callback)
        return W_opt, learning_curve

    def hyperloss(hyperparam_vect, i_hyper):
        W_opt, _ = primal_optimizer(hyperparam_vect, i_hyper)
        return loss_fun(W_opt, **valid_data)
    hyperloss_grad = grad(hyperloss)

    meta_results = defaultdict(list)
    def meta_callback(hyperparam_vect, i_hyper):
        print "Epoch {0}".format(i_hyper)
        x, learning_curve = primal_optimizer(hyperparam_vect, i_hyper)
        cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
        for field in cur_hyperparams.names:
            meta_results[field].append(cur_hyperparams[field])
        meta_results['train_loss'].append(loss_fun(x, **train_data))
        meta_results['valid_loss'].append(loss_fun(x, **valid_data))
        meta_results['tests_loss'].append(loss_fun(x, **tests_data))
        meta_results['learning_curves'].append(learning_curve)

    final_result = rms_prop(hyperloss_grad, hyperparams.vect,
                            meta_callback, N_meta_iter, meta_alpha)
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
    for i, y in enumerate(results['learning_curves']):
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
    ax.legend(loc=1, frameon=False)
    plt.savefig('learning_curves.png')

    # ----- Alpha and beta schedules -----
    fig.clf()
    ax = fig.add_subplot(211)
    ax.set_title('Alpha learning curves')
    for i, y in enumerate(results['log_alphas']):
        ax.plot(y, 'o-', label="Meta iter {0}".format(i))
    ax.set_xlabel('Primal iter number')
    ax.set_ylabel('Log alpha')
    ax.legend(loc=1, frameon=False)

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
        ax.plot(y, 'o-', label=parser.names[i])
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Log param scale')
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(212)
    ax.set_title('L2 reg learning curves')
    for i, y in enumerate(zip(*results['log_L2_reg'])):
        ax.plot(y, 'o-', label=parser.names[i])
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Log L2 reg')
    plt.savefig('scale_and_reg.png')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()
