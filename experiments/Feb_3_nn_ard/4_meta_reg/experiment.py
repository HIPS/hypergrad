"""Learns an l2 penalty per-weight."""
import numpy as np
import pickle
from collections import defaultdict

from funkyyak import grad, kylist, getval

from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, VectorParser, logit, inv_logit, plot_mnist_images
from hypergrad.optimizers import adam, sgd_parsed
from hypergrad.util import RandomState

# ----- Fixed params -----
layer_sizes = [784, 300, 100, 10]
batch_size = 200
N_iters = 50
N_classes = 10
N_train = 1000
N_valid = 10000
N_tests = 10000
N_learning_checkpoint = 2
thin = np.ceil(N_iters/N_learning_checkpoint)

# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = -2.0
init_log_L2_reg_noise = 0.5
init_log_alphas = -1.0
init_invlogit_betas = inv_logit(0.5)
init_log_param_scale = -3.0

L2_meta_penalty_mean = init_log_L2_reg
L2_meta_penalty = 1.0

# ----- Superparameters -----
meta_alpha = 0.04
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
    N_weights = len(parser.vect)
    hyperparams = VectorParser()
    rs = RandomState((seed))
    hyperparams['log_L2_reg'] = np.full(N_weights, init_log_L2_reg)\
                              + rs.randn(N_weights) * init_log_L2_reg_noise
    fixed_hyperparams = VectorParser()
    fixed_hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
    fixed_hyperparams['log_alphas']      = np.full((N_iters, N_weight_types), init_log_alphas)
    fixed_hyperparams['invlogit_betas']  = np.full((N_iters, N_weight_types), init_invlogit_betas)

    cur_primal_results = {}

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
        W0 = fill_parser(parser, np.exp(fixed_hyperparams['log_param_scale']))
        W0 *= rs.randn(W0.size)
        alphas = np.exp(fixed_hyperparams['log_alphas'])
        betas  = logit(fixed_hyperparams['invlogit_betas'])
        L2_reg = np.exp(cur_hyperparams['log_L2_reg'])
        W_opt = sgd_parsed(grad(indexed_loss_fun), kylist(W0, alphas, betas, L2_reg),
                           parser, callback=callback)
        cur_primal_results['weights'] = getval(W_opt).copy()
        cur_primal_results['learning_curve'] = getval(learning_curve_dict)
        return W_opt, learning_curve_dict

    def hyperloss(hyperparam_vect, i_hyper):
        W_opt, _ = primal_optimizer(hyperparam_vect, i_hyper)
        L2_norm = np.sum((hyperparam_vect - L2_meta_penalty_mean)**2)
        return loss_fun(W_opt, **valid_data) + L2_norm * L2_meta_penalty
    hyperloss_grad = grad(hyperloss)

    meta_results = defaultdict(list)
    old_metagrad = [np.ones(hyperparams.vect.size)]
    def meta_callback(hyperparam_vect, i_hyper, metagrad=None):
        #x, learning_curve_dict = primal_optimizer(hyperparam_vect, i_hyper)
        x, learning_curve_dict = cur_primal_results['weights'], cur_primal_results['learning_curve']
        cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
        for field in cur_hyperparams.names:
            meta_results[field] = cur_hyperparams[field]
        meta_results['train_loss'].append(loss_fun(x, **train_data))
        meta_results['valid_loss'].append(loss_fun(x, **valid_data))
        meta_results['tests_loss'].append(loss_fun(x, **tests_data))
        meta_results['test_err'].append(frac_err(x, **tests_data))
        meta_results['learning_curves'].append(learning_curve_dict)
        meta_results['example_weights'] = x
        if metagrad is not None:
            meta_results['meta_grad_magnitude'].append(np.linalg.norm(metagrad))
            meta_results['meta_grad_angle'].append(np.dot(old_metagrad[0], metagrad) \
                                                   / (np.linalg.norm(metagrad)*
                                                      np.linalg.norm(old_metagrad[0])))
        old_metagrad[0] = metagrad
        print "Meta Epoch {0} Train loss {1:2.4f} Valid Loss {2:2.4f}" \
              " Test Loss {3:2.4f} Test Err {4:2.4f}".format(
            i_hyper, meta_results['train_loss'][-1], meta_results['valid_loss'][-1],
            meta_results['tests_loss'][-1], meta_results['test_err'][-1])

    initial_hypergrad = hyperloss_grad( hyperparams.vect, 0)
    parsed_init_hypergrad = hyperparams.new_vect(initial_hypergrad.copy())
    final_result = adam(hyperloss_grad, hyperparams.vect, meta_callback, N_meta_iter, meta_alpha)
    meta_callback(final_result, N_meta_iter)
    parser.vect = None # No need to pickle zeros
    return meta_results, parser, parsed_init_hypergrad


def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser, parsed_init_hypergrad = pickle.load(f)

    # Show final ARD in the first layer filters.
    L2s = results['log_L2_reg']
    L2parser = parser.empty_copy()
    L2parser.vect = L2s
    l2_images = L2parser[('weights', 0)].T

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((6,8))
    ax = fig.add_subplot(111)
    plot_mnist_images(l2_images, ax, ims_per_row=30, padding=0)
    fig.set_size_inches((8,12))

    #fig.tight_layout()
    plt.savefig("penalties.png")
    plt.savefig("penalties.pdf", pad_inches=0.05, bbox_inches='tight')


    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((6,8))
    ax = fig.add_subplot(111)
    plt.hist(l2_images.ravel(), 100)
    plt.savefig("penalty_histogram.png")



    # Show first layer filters from the last meta-iteration.
    weights = results['example_weights']
    parser.vect = weights
    weight_images = parser[('weights', 0)].T

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((6,8))
    ax = fig.add_subplot(111)
    plot_mnist_images(weight_images, ax, ims_per_row=30, padding=0)
    fig.set_size_inches((8,12))

    #fig.tight_layout()
    plt.savefig("weights.png")
    plt.savefig("weights.pdf", pad_inches=0.05, bbox_inches='tight')



    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((6,8))
    # ----- Primal learning curves -----
    ax = fig.add_subplot(311)
    ax.set_title('Primal learning curves')
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['learning_curve'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Negative log prob')
    #ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(312)
    ax.set_title('Meta learning curves')
    losses = ['train_loss', 'valid_loss', 'tests_loss']
    for loss_type in losses:
        ax.plot(results[loss_type], 'o-', label=loss_type)
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Negative log prob')
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(313)
    ax.set_title('Meta-gradient magnitude')
    ax.plot(results['meta_grad_magnitude'], 'o-', label='Meta-gradient magnitude')
    ax.plot(results['meta_grad_angle'], 'o-', label='Meta-gradient angle')
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Meta-gradient Magnitude')
    ax.legend(loc=1, frameon=False)

    plt.savefig('learning_curves.png')


    # ----- Learning curve info -----
    fig.clf()
    ax = fig.add_subplot(311)
    ax.set_title('Primal learning curves')
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['grad_norm'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    #ax.legend(loc=1, frameon=False)
    ax.set_title('Grad norm')

    ax = fig.add_subplot(312)
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['weight_norm'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    ax.legend(loc=1, frameon=False)
    ax.set_title('Weight norm')

    ax = fig.add_subplot(313)
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['velocity_norm'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    ax.set_title('Velocity norm')
    ax.legend(loc=1, frameon=False)
    plt.savefig('extra_learning_curves.png')




if __name__ == '__main__':
    #results = run()
    #with open('results.pkl', 'w') as f:
    #    pickle.dump(results, f)
    plot()