"""Find good initial values using Whetlab."""
import numpy as np
import pickle
from collections import defaultdict

from funkyyak import grad, kylist

from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, VectorParser, logit, inv_logit, fill_parser
from hypergrad.optimizers import simple_sgd, sgd_parsed
from hypergrad.util import RandomState

import whetlab
parameters = { 'init_log_alphas':{'min':-3, 'max':2, 'type':'float'},
               'init_invlogit_betas':{'min':inv_logit(0.1), 'max':inv_logit(0.999),'type':'float'},
               'init_log_param_scale':{'min':-5, 'max':1,'type':'float'}}
outcome = {'name':'Training Gain'}
scientist = whetlab.Experiment(name="ICML Hypergrad paper - optimize initial values - v2 no randomness",
                               description="Vanilla tuning of hyperparameters. v2",
                               parameters=parameters,
                               outcome=outcome)

# ----- Fixed params -----
layer_sizes = [784, 50, 50, 50, 10]
batch_size = 200
N_iters = 100
N_classes = 10
N_train = 10000
N_valid = 10000
N_tests = 10000
N_learning_checkpoint = 10
thin = np.ceil(N_iters/N_learning_checkpoint)
init_log_L2_reg = -100.0

# ----- Initial values of learned hyper-parameters -----
init_log_alphas = -1.0
init_invlogit_betas = inv_logit(0.5)
init_log_param_scale = -3.0

# ----- Superparameters -----
N_meta_iter = 200

seed = 0

def run():
    train_data, valid_data, tests_data = load_data_dicts(N_train, N_valid, N_tests)
    parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weight_types = len(parser.names)

    def build_hypervect(init_log_alphas, init_invlogit_betas, init_log_param_scale):
        hyperparams = VectorParser()
        hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
        hyperparams['log_alphas']      = np.full((N_iters, N_weight_types), init_log_alphas)
        hyperparams['invlogit_betas']  = np.full((N_iters, N_weight_types), init_invlogit_betas)
        return hyperparams

    hyperparams = build_hypervect(init_log_alphas, init_invlogit_betas, init_log_param_scale)  # Build just for parser.
    fixed_hyperparams = VectorParser()
    fixed_hyperparams['log_L2_reg'] = np.full(N_weight_types, init_log_L2_reg)

    def whetlab_optimize(loss, max_iters, callback):
        for i in xrange(max_iters):
            params = scientist.suggest()
            hyperparams = build_hypervect(**params)
            cur_loss = loss(hyperparams.vect, 0)  # No randomness
            scientist.update(params, -cur_loss)
            if callback: callback(hyperparams.vect, 0)

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

    meta_results = defaultdict(list)
    old_metagrad = [np.ones(hyperparams.vect.size)]
    def meta_callback(hyperparam_vect, i_hyper, metagrad=None):
        x, learning_curve_dict = primal_optimizer(hyperparam_vect, i_hyper)
        cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
        for field in cur_hyperparams.names:
            meta_results[field].append(cur_hyperparams[field])
        meta_results['train_loss'].append(loss_fun(x, **train_data))
        meta_results['valid_loss'].append(loss_fun(x, **valid_data))
        meta_results['tests_loss'].append(loss_fun(x, **tests_data))
        meta_results['test_err'].append(frac_err(x, **tests_data))
        meta_results['learning_curves'].append(learning_curve_dict)
        if metagrad is not None:
            meta_results['meta_grad_magnitude'].append(np.linalg.norm(metagrad))
            meta_results['meta_grad_angle'].append(np.dot(old_metagrad[0], metagrad) \
                                                   / (np.linalg.norm(metagrad)*
                                                      np.linalg.norm(old_metagrad[0])))
        old_metagrad[0] = metagrad
        print "Meta Epoch {0} Train loss {1:2.4f} Valid Loss {2:2.4f}" \
              " Test Loss {3:2.4f} Test Err {4:2.4f}".format(
            i_hyper, meta_results['train_loss'][-1], meta_results['valid_loss'][-1],
            meta_results['train_loss'][-1], meta_results['test_err'][-1])

    whetlab_optimize(hyperloss, N_meta_iter, meta_callback)
    best_params = scientist.best()
    print "best params:", best_params

    parser.vect = None # No need to pickle zeros
    return meta_results, parser, best_params


def plot():

    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser, best_params = pickle.load(f)

    print "best params:", best_params

    # ----- Nice versions of Alpha and beta schedules for paper -----
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(411)
    #ax.set_title('Alpha learning curves')
    for cur_results, name in zip(results['log_alphas'][-1].T, parser.names):
        if name[0] == 'weights':
            ax.plot(np.exp(cur_results), 'o-', label=name)
    #ax.set_xlabel('Learning Iteration', fontproperties='serif')
    low, high = ax.get_ylim()
    ax.set_ylim([0, high])
    ax.set_ylabel('Step size', fontproperties='serif')
    ax.set_xticklabels([])
    ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
              prop={'family':'serif', 'size':'12'})

    ax = fig.add_subplot(412)
    #ax.set_title('Alpha learning curves')
    for cur_results, name in zip(results['invlogit_betas'][-1].T, parser.names):
        if name[0] == 'weights':
            ax.plot(logit(cur_results), 'o-', label=name)
    low, high = ax.get_ylim()
    ax.set_ylim([0, 1])
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum', fontproperties='serif')

    ax = fig.add_subplot(413)
    #ax.set_title('Alpha learning curves')
    for cur_results, name in zip(results['log_alphas'][-1].T, parser.names):
        if name[0] == 'biases':
            ax.plot(np.exp(cur_results), 'o-', label=name)
    #ax.set_xlabel('Learning Iteration', fontproperties='serif')
    low, high = ax.get_ylim()
    ax.set_ylim([0, high])
    ax.set_ylabel('Step size', fontproperties='serif')
    ax.set_xticklabels([])
    ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
              prop={'family':'serif', 'size':'12'})

    ax = fig.add_subplot(414)
    #ax.set_title('Alpha learning curves')
    for cur_results, name in zip(results['invlogit_betas'][-1].T, parser.names):
        if name[0] == 'biases':
            ax.plot(logit(cur_results), 'o-', label=name)
    low, high = ax.get_ylim()
    ax.set_ylim([0, 1])
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum', fontproperties='serif')


    fig.set_size_inches((6,8))
    #plt.show()
    plt.savefig('alpha_beta_paper.png')
    plt.savefig('alpha_beta_paper.pdf', pad_inches=0.05, bbox_inches='tight')

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
    ax = fig.add_subplot(111)
    ax.set_title('Init scale learning curves')
    for i, y in enumerate(zip(*results['log_param_scale'])):
        if parser.names[i][0] == 'weights':
            ax.plot(y, 'o-', label=parser.names[i])
    ax.set_xlabel('Meta iter number')
    ax.set_ylabel('Log param scale')
    ax.legend(loc=1, frameon=False)


    plt.savefig('scale_and_reg.png')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()