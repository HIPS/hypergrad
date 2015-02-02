"""Showing that the average hypergradient over many minibatches looks sensible."""
import numpy as np
import pickle
from collections import defaultdict

from funkyyak import grad, kylist

from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, VectorParser, logit, inv_logit
from hypergrad.optimizers import sgd_parsed
from hypergrad.util import RandomState

# ----- Fixed params -----
layer_sizes = [784, 50, 50, 50, 10]
batch_size = 200
N_iters = 100
N_classes = 10
N_train = 10000
N_valid = 10000
N_tests = 10000
N_learning_checkpoint = 1
thin = np.ceil(N_iters/N_learning_checkpoint)
init_log_L2_reg = -100.0

# ----- Initial values of learned hyper-parameters -----
init_log_alphas = -1.0
init_invlogit_betas = inv_logit(0.5)
init_log_param_scale = -3.0

# ----- Superparameters -----
N_meta_iter = 100
seed = 0

def fill_parser(parser, items):
    partial_vects = [np.full(parser[name].size, items[i])
                     for i, name in enumerate(parser.names)]
    return np.concatenate(partial_vects, axis=0)

def run():
    train_data, valid_data, tests_data = load_data_dicts(N_train, N_valid, N_tests)
    parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weight_types = len(parser.names)
    hyperparams = VectorParser()
    hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
    hyperparams['log_alphas']      = np.full((N_iters, N_weight_types), init_log_alphas)
    hyperparams['invlogit_betas']  = np.full((N_iters, N_weight_types), init_invlogit_betas)
    fixed_hyperparams = VectorParser()
    fixed_hyperparams['log_L2_reg'] = np.full(N_weight_types, init_log_L2_reg)

    def primal_optimizer(hyperparam_vect, i_hyper):
        def indexed_loss_fun(w, L2_vect, i_iter):
            rs = RandomState((seed, i_hyper, i_iter))  # Deterministic seed needed for backwards pass.
            idxs = rs.randint(N_train, size=batch_size)
            return loss_fun(w, train_data['X'][idxs], train_data['T'][idxs], L2_vect)

        learning_curve_dict = defaultdict(list)
        def callback(x, v, g, i_iter):
            if i_iter % thin == 0:
                learning_curve_dict['learning_curve'].append(loss_fun(x, **train_data))
                learning_curve_dict['grad_norm'].append(np.linalg.norm(g))
                learning_curve_dict['weight_norm'].append(np.linalg.norm(x))
                learning_curve_dict['velocity_norm'].append(np.linalg.norm(v))

        cur_hyperparams = hyperparams.new_vect(hyperparam_vect)
        rs = RandomState((seed, i_hyper))
        W0 = fill_parser(parser, np.exp(cur_hyperparams['log_param_scale']))
        W0 *= rs.randn(W0.size)
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas  = logit(cur_hyperparams['invlogit_betas'])
        L2_reg = fill_parser(parser, np.exp(fixed_hyperparams['log_L2_reg']))
        W_opt = sgd_parsed(grad(indexed_loss_fun), kylist(W0, alphas, betas, L2_reg),
                           parser, callback=callback)
        return W_opt, learning_curve_dict

    def hyperloss(hyperparam_vect, i_hyper):
        W_opt, _ = primal_optimizer(hyperparam_vect, i_hyper)
        return loss_fun(W_opt, **train_data)
    hyperloss_grad = grad(hyperloss)

    initial_hypergrad = hyperloss_grad( hyperparams.vect, 0)
    parsed_init_hypergrad = hyperparams.new_vect(initial_hypergrad.copy())
    avg_hypergrad = initial_hypergrad.copy()
    for i in xrange(1, N_meta_iter):
        avg_hypergrad += hyperloss_grad( hyperparams.vect, i)
        print i
    parsed_avg_hypergrad = hyperparams.new_vect(avg_hypergrad)

    parser.vect = None # No need to pickle zeros
    return parser, parsed_init_hypergrad, parsed_avg_hypergrad


def plot():

    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        parser, parsed_init_hypergrad, parsed_avg_hypergrad = pickle.load(f)

    # ----- Alpha and beta initial hypergradients -----
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(411)
    for cur_results, name in zip(parsed_init_hypergrad['log_alphas'].T, parser.names):
        if name[0] == 'weights':
            ax.plot(cur_results, 'o-', label=name)
    ax.set_ylabel('Step size Gradient', fontproperties='serif')
    ax.set_xticklabels([])
    ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
              prop={'family':'serif', 'size':'12'})

    ax = fig.add_subplot(412)
    for cur_results, name in zip(parsed_init_hypergrad['invlogit_betas'].T, parser.names):
        if name[0] == 'weights':
            ax.plot(cur_results, 'o-', label=name)
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum Gradient', fontproperties='serif')

    ax = fig.add_subplot(413)
    for cur_results, name in zip(parsed_init_hypergrad['log_alphas'].T, parser.names):
        if name[0] == 'biases':
            ax.plot(cur_results, 'o-', label=name)
    ax.set_ylabel('Step size Gradient', fontproperties='serif')
    ax.set_xticklabels([])
    ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
              prop={'family':'serif', 'size':'12'})

    ax = fig.add_subplot(414)
    for cur_results, name in zip(parsed_init_hypergrad['invlogit_betas'].T, parser.names):
        if name[0] == 'biases':
            ax.plot(cur_results, 'o-', label=name)
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum Gradient', fontproperties='serif')

    fig.set_size_inches((6,8))
    #plt.show()
    plt.savefig('initial_gradient.png')
    plt.savefig('initial_gradient.pdf', pad_inches=0.05, bbox_inches='tight')



    # ----- Alpha and beta initial hypergradients -----
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(411)
    for cur_results, name in zip(parsed_avg_hypergrad['log_alphas'].T, parser.names):
        if name[0] == 'weights':
            ax.plot(cur_results, 'o-', label=name)
    ax.set_ylabel('Step size Gradient', fontproperties='serif')
    ax.set_xticklabels([])
    ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
              prop={'family':'serif', 'size':'12'})

    ax = fig.add_subplot(412)
    for cur_results, name in zip(parsed_avg_hypergrad['invlogit_betas'].T, parser.names):
        if name[0] == 'weights':
            ax.plot(cur_results, 'o-', label=name)
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum Gradient', fontproperties='serif')

    ax = fig.add_subplot(413)
    for cur_results, name in zip(parsed_avg_hypergrad['log_alphas'].T, parser.names):
        if name[0] == 'biases':
            ax.plot(cur_results, 'o-', label=name)
    ax.set_ylabel('Step size Gradient', fontproperties='serif')
    ax.set_xticklabels([])
    ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
              prop={'family':'serif', 'size':'12'})

    ax = fig.add_subplot(414)
    for cur_results, name in zip(parsed_avg_hypergrad['invlogit_betas'].T, parser.names):
        if name[0] == 'biases':
            ax.plot(cur_results, 'o-', label=name)
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum Gradient', fontproperties='serif')

    fig.set_size_inches((6,8))
    #plt.show()
    plt.savefig('average_gradient.png')
    plt.savefig('average_gradient.pdf', pad_inches=0.05, bbox_inches='tight')



if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()