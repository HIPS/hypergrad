"""Gradient descent to optimize minimizing a parabola."""
import numpy as np
import numpy.random as npr
import pickle
from collections import defaultdict

from funkyyak import grad, kylist

from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, VectorParser, logit, inv_logit
from hypergrad.optimizers import sgd4, rms_prop, adam

# ----- Fixed params -----
N_iters = 60
dimension = 100
# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = 0.0
init_log_alphas = -2.0
init_invlogit_betas = inv_logit(0.9)
init_log_param_scale = -np.Inf #2.0
# ----- Superparameters -----
meta_alpha = 0.02
N_meta_iter = 100

npr.seed(0)

def fill_parser(parser, items):
    partial_vects = [np.full(parser[name].size, items[i])
                     for i, name in enumerate(parser.names)]
    return np.concatenate(partial_vects, axis=0)

def make_parabola(d):
    parser = VectorParser()
    parser.add_shape('weights', d)
    dimscale = np.exp(np.linspace(-3, 3, d))
    offset = npr.randn(d)

    def loss(w, X=0.0, T=0.0, L2_reg=0.0):
        return np.dot((w - offset) * dimscale, (w - offset))

    return parser, loss

def run():
    parser, loss_fun = make_parabola(dimension)
    N_weight_types = len(parser.names)
    hyperparams = VectorParser()
    hyperparams['log_L2_reg']      = np.full(N_weight_types, init_log_L2_reg)
    hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
    hyperparams['log_alphas']      = np.full(N_iters, init_log_alphas)
    hyperparams['invlogit_betas']  = np.full(N_iters, init_invlogit_betas)

    def primal_optimizer(hyperparam_vect, i_hyper):
        learning_curve = []
        def callback(x, i_iter):
            learning_curve.append(loss_fun(x))

        cur_hyperparams = hyperparams.new_vect(hyperparam_vect)
        W0 = fill_parser(parser, np.exp(cur_hyperparams['log_param_scale']))
        W0 *= npr.RandomState(hash(i_hyper)).randn(W0.size)
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas  = logit(cur_hyperparams['invlogit_betas'])
        L2_reg = fill_parser(parser, np.exp(cur_hyperparams['log_L2_reg']))
        W_opt = sgd4(grad(loss_fun), kylist(W0, alphas, betas, L2_reg), callback)
        callback(W_opt, N_iters)
        return W_opt, learning_curve

    def hyperloss(hyperparam_vect, i_hyper):
        W_opt, _ = primal_optimizer(hyperparam_vect, i_hyper)
        return loss_fun(W_opt)
    hyperloss_grad = grad(hyperloss)

    meta_results = defaultdict(list)
    def meta_callback(hyperparam_vect, i_hyper):
        print "Meta Epoch {0}".format(i_hyper)
        x, learning_curve = primal_optimizer(hyperparam_vect, i_hyper)
        cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
        for field in cur_hyperparams.names:
            meta_results[field].append(cur_hyperparams[field])
        meta_results['train_loss'].append(loss_fun(x))
        meta_results['learning_curves'].append(learning_curve)

    final_result = rms_prop(hyperloss_grad, hyperparams.vect,
                            meta_callback, N_meta_iter, meta_alpha, gamma=0.0)
    meta_callback(final_result, N_meta_iter)
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
    losses = ['train_loss']
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
