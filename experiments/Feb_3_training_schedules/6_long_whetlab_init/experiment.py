"""Shows both an initial average gradient, as well as the final learned schedule."""
import numpy as np
import pickle
from collections import defaultdict

from funkyyak import grad, kylist, getval

from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, VectorParser, logit, inv_logit, nice_layer_name, fill_parser
from hypergrad.optimizers import adam, sgd_parsed
from hypergrad.util import RandomState

# ----- Fixed params -----
layer_sizes = [784, 50, 50, 50, 10]
batch_size = 200
N_iters = 1000
N_classes = 10
N_train = 10000
N_valid = 10000
N_tests = 10000
N_learning_checkpoint = 100
thin = np.ceil(N_iters/N_learning_checkpoint)

# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = -100.0
init_log_alphas = 0.887   # Based on 100 iterations of whetlab.  (Experiment ID 70189)
init_invlogit_betas = 0.403
init_log_param_scale = -1.63

# ----- Superparameters -----
N_meta_iter = 1
seed = 0

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
            if i_iter % thin == 0 or i_iter == N_iters or i_iter == 0:
                learning_curve_dict['learning_curve'].append(loss_fun(x, **train_data))
                learning_curve_dict['grad_norm'].append(np.linalg.norm(g))
                learning_curve_dict['weight_norm'].append(np.linalg.norm(x))
                learning_curve_dict['velocity_norm'].append(np.linalg.norm(v))
                learning_curve_dict['iteration'].append(i_iter + 1)
                print "iteration", i_iter

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
            meta_results['train_loss'][-1], meta_results['test_err'][-1])

    meta_callback(hyperparams.vect, N_meta_iter)
    parser.vect = None # No need to pickle zeros
    return meta_results, parser


def plot():

    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font',**{'family':'serif'})

    with open('results.pkl') as f:
        results, parser = pickle.load(f)

    with open('../3_adam_50/results.pkl') as f:
        opt_results, opt_parser, parsed_init_hypergrad = pickle.load(f)

    # ----- Nice learning curves for paper -----
    print "Plotting nice learning curves for paper..."
    fig = plt.figure(0)
    fig.clf()
    # ----- Primal learning curves -----
    ax = fig.add_subplot(111)
    #ax.set_title('Primal learning curves')
    ax.plot(opt_results['learning_curves'][0]['iteration'],
            opt_results['learning_curves'][0]['learning_curve'],  '-', label='Initial hypers')
    ax.plot(opt_results['learning_curves'][-1]['iteration'],
            opt_results['learning_curves'][-1]['learning_curve'],  '-', label='Final hypers')
    ax.plot(results['learning_curves'][0]['iteration'],
            results['learning_curves'][0]['learning_curve'],  '-', label='BayesOpt hypers')
    ax.set_xlabel('Training iteration')
    ax.set_ylabel('Training loss')
    plt.show()
    fig.set_size_inches((2.5,2.5))
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'10'})

    plt.savefig('learning_curves_paper.pdf', pad_inches=0.05, bbox_inches='tight')


    fig.clf()
    ax = fig.add_subplot(111)
    #ax.set_title('Meta learning curves')
    losses = ['train_loss', 'valid_loss', 'tests_loss']
    loss_names = ['Training loss', 'Validation loss', 'Test loss']
    for loss_type, loss_name in zip(losses, loss_names):
        ax.plot(results[loss_type], 'o-', label=loss_name)
    ax.set_xlabel('Meta iteration')
    ax.set_ylabel('Predictive loss')
    ax.legend(loc=1, frameon=False)
    fig.set_size_inches((2.5,2.5))
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'10'})
    plt.savefig('meta_learning_curve_paper.pdf', pad_inches=0.05, bbox_inches='tight')

if __name__ == '__main__':
    #results = run()
    #with open('results.pkl', 'w') as f:
    #    pickle.dump(results, f)
    plot()