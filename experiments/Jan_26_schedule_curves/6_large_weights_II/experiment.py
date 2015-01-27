"""Gradient descent to optimize everything,
subject to the constraint that weight initialization can't be set to zero."""
import numpy as np
import numpy.random as npr
import pickle
from collections import defaultdict

from funkyyak import grad, kylist

from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, VectorParser, logit, inv_logit
from hypergrad.optimizers import sgd4, rms_prop, adam

# ----- Fixed params -----
layer_sizes = [784, 10]
batch_size = 200
N_iters = 60
N_classes = 10
N_train = 1000
N_valid = 10**3
N_tests = 10**3
N_batches = N_train / batch_size
#N_iters = N_epochs * N_batches
# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = -2.0
init_log_alphas = -1.0
init_invlogit_betas = inv_logit(0.99)
init_log_param_scale = -2.0
# ----- Superparameters -----
meta_alpha = 0.04
N_meta_iter = 100

global_seed = npr.RandomState(3).randint(1000)

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
    fixed_hyperparams = VectorParser()
    fixed_hyperparams['log_param_scale']  = np.full(N_iters, init_log_param_scale)

    # TODO: memoize
    def primal_optimizer(hyperparam_vect, i_hyper):
        def indexed_loss_fun(w, L2_vect, i_iter):
            rs = npr.RandomState(npr.RandomState(global_seed + i_hyper).randint(1000))
            seed = i_hyper * 10**6 + i_iter   # Deterministic seed needed for backwards pass.
            idxs = rs.randint(N_train, size=batch_size)
            return loss_fun(w, train_data['X'][idxs], train_data['T'][idxs], L2_vect)

        learning_curve_dict = defaultdict(list)
        def callback(x, v, g, i_iter):
            if i_iter % N_batches == 0:
                learning_curve_dict['learning_curve'].append(loss_fun(x, **train_data))
                learning_curve_dict['grad_norm'].append(np.linalg.norm(g))
                learning_curve_dict['weight_norm'].append(np.linalg.norm(x))
                learning_curve_dict['velocity_norm'].append(np.linalg.norm(v))

        cur_hyperparams = hyperparams.new_vect(hyperparam_vect)
        W0 = fill_parser(parser, np.exp(fixed_hyperparams['log_param_scale']))
        W0 *= npr.RandomState(global_seed + i_hyper).randn(W0.size)
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas  = logit(cur_hyperparams['invlogit_betas'])
        L2_reg = fill_parser(parser, np.exp(cur_hyperparams['log_L2_reg']))
        W_opt = sgd4(grad(indexed_loss_fun), kylist(W0, alphas, betas, L2_reg), callback)
        #callback(W_opt, N_iters)
        return W_opt, learning_curve_dict

    def hyperloss(hyperparam_vect, i_hyper):
        W_opt, _ = primal_optimizer(hyperparam_vect, i_hyper)
        return loss_fun(W_opt, **valid_data)
    hyperloss_grad = grad(hyperloss)

    meta_results = defaultdict(list)
    old_metagrad = np.ones(hyperparams.vect.size)
    def meta_callback(hyperparam_vect, i_hyper, metagrad):
        print "Meta Epoch {0}".format(i_hyper)
        x, learning_curve_dict = primal_optimizer(hyperparam_vect, i_hyper)
        cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
        for field in cur_hyperparams.names:
            meta_results[field].append(cur_hyperparams[field])
        meta_results['train_loss'].append(loss_fun(x, **train_data))
        meta_results['valid_loss'].append(loss_fun(x, **valid_data))
        meta_results['tests_loss'].append(loss_fun(x, **tests_data))
        meta_results['learning_curves'].append(learning_curve_dict)
        meta_results['meta_grad_magnitude'].append(np.linalg.norm(metagrad))
        meta_results['meta_grad_angle'].append(np.dot(old_metagrad, metagrad) / (np.linalg.norm(metagrad)*np.linalg.norm(old_metagrad)))

    final_result = rms_prop(hyperloss_grad, hyperparams.vect,
                            meta_callback, N_meta_iter, meta_alpha, gamma=0.0)
    #meta_callback(final_result, N_meta_iter)
    parser.vect = None # No need to pickle zeros
    return meta_results, parser

def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser = pickle.load(f)


    # ----- Nice versions of Alpha and beta schedules for paper -----
    # fig = plt.figure(0)
    # fig.set_size_inches((6,4))
    # fig.clf()
    # ax = fig.add_subplot(111)
    # #ax.set_title('Alpha learning curves')
    # ax.plot(np.exp(results['log_alphas'][-1]), 'o-', label="Step size")
    # ax.plot(logit(results['invlogit_betas'][-1]), 'o-', label="Momentum")
    # ax.set_xlabel('Learning Iteration', fontproperties='serif')
    # ax.set_ylabel('Log alpha')
    # ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
    #           borderaxespad=0.0, prop={'family':'serif', 'size':'12'})
    # plt.show()
    # plt.savefig('alpha_beta_paper.png')
    # plt.savefig('alpha_beta_paper.pdf', pad_inches=0.05, bbox_inches='tight')

    fig = plt.figure(0)
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
    #ax.legend(loc=1, frameon=False)

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
    #ax.legend(loc=1, frameon=False)
    ax.set_title('Weight norm')

    ax = fig.add_subplot(313)
    for i, y in enumerate(results['learning_curves']):
        ax.plot(y['velocity_norm'], 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Epoch number')
    ax.set_title('Velocity norm')
    #ax.legend(loc=1, frameon=False)
    plt.savefig('extra_learning_curves.png')

    # ----- Alpha and beta schedules -----
    fig.clf()
    ax = fig.add_subplot(211)
    ax.set_title('Alpha learning curves')
    for i, y in enumerate(results['log_alphas']):
        ax.plot(y, 'o-', label="Meta iter {0}".format(i))
    ax.set_xlabel('Primal iter number')
    #ax.set_ylabel('Log alpha')
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
    #results = run()
    #with open('results.pkl', 'w') as f:
    #    pickle.dump(results, f)
    plot()
