"""Shows both the initial average gradient."""
import numpy as np
import pickle
from collections import defaultdict

from funkyyak import grad, kylist, getval

from hypergrad.data import load_data_dicts
from hypergrad.nn_utils import make_nn_funs, VectorParser, logit, inv_logit, nice_layer_name
from hypergrad.optimizers import adam, sgd_parsed
from hypergrad.util import RandomState

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

# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = -100.0
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
        W0 = fill_parser(parser, np.exp(cur_hyperparams['log_param_scale']))
        W0 *= rs.randn(W0.size)
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas  = logit(cur_hyperparams['invlogit_betas'])
        L2_reg = fill_parser(parser, np.exp(fixed_hyperparams['log_L2_reg']))
        W_opt = sgd_parsed(grad(indexed_loss_fun), kylist(W0, alphas, betas, L2_reg),
                           parser, callback=callback)
        cur_primal_results['weights'] = getval(W_opt).copy()
        cur_primal_results['learning_curve'] = getval(learning_curve_dict)
        return W_opt, learning_curve_dict

    def hyperloss(hyperparam_vect, i_hyper):
        W_opt, _ = primal_optimizer(hyperparam_vect, i_hyper)
        return loss_fun(W_opt, **train_data)
    hyperloss_grad = grad(hyperloss)

    meta_results = defaultdict(list)
    old_metagrad = [np.ones(hyperparams.vect.size)]
    def meta_callback(hyperparam_vect, i_hyper, metagrad=None):
        x, learning_curve_dict = cur_primal_results['weights'], cur_primal_results['learning_curve']
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


    initial_hypergrad = hyperloss_grad( hyperparams.vect, 0)
    hypergrads = np.zeros((N_meta_iter, len(initial_hypergrad)))
    for i in xrange(N_meta_iter):
        hypergrads[i] = hyperloss_grad( hyperparams.vect, i)
        print i
    avg_hypergrad = np.mean(hypergrads, axis=0)
    parsed_avg_hypergrad = hyperparams.new_vect(avg_hypergrad)

    parser.vect = None # No need to pickle zeros
    return parser, parsed_avg_hypergrad


def plot():

    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font',**{'family':'serif'})

    with open('results.pkl') as f:
        parser, parsed_avg_hypergrad = pickle.load(f)

    #rc('text', usetex=True)

   # ----- Small versions of stepsize schedules for paper -----
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    def layer_name(weight_key):
        return "Layer {num}".format(num=weight_key[1] + 1)
    for cur_results, name in zip(parsed_avg_hypergrad['log_alphas'].T, parser.names):
        if name[0] == 'weights':
            ax.plot(np.exp(cur_results), 'o-', label=layer_name(name))
    low, high = ax.get_ylim()
    #ax.set_ylim([0, high])
    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Schedule index')
    fig.set_size_inches((6,2.5))
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    plt.savefig('schedules_small.pdf', pad_inches=0.05, bbox_inches='tight')


    # ----- Alpha and beta initial hypergradients -----
    print "Plotting initial gradients..."
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(411)
    for cur_results, name in zip(parsed_avg_hypergrad['log_alphas'].T, parser.names):
        if name[0] == 'weights':
            ax.plot(cur_results, 'o-', label=nice_layer_name(name))
    ax.set_ylabel('Step size Gradient', fontproperties='serif')
    ax.set_xticklabels([])
    ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
              prop={'family':'serif', 'size':'12'})

    ax = fig.add_subplot(412)
    for cur_results, name in zip(parsed_avg_hypergrad['invlogit_betas'].T, parser.names):
        if name[0] == 'weights':
            ax.plot(cur_results, 'o-', label=nice_layer_name(name))
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum Gradient', fontproperties='serif')

    ax = fig.add_subplot(413)
    for cur_results, name in zip(parsed_avg_hypergrad['log_alphas'].T, parser.names):
        if name[0] == 'biases':
            ax.plot(cur_results, 'o-', label=nice_layer_name(name))
    ax.set_ylabel('Step size Gradient', fontproperties='serif')
    ax.set_xticklabels([])
    ax.legend(numpoints=1, loc=1, frameon=False, bbox_to_anchor=(1.0, 0.5),
              prop={'family':'serif', 'size':'12'})

    ax = fig.add_subplot(414)
    for cur_results, name in zip(parsed_avg_hypergrad['invlogit_betas'].T, parser.names):
        if name[0] == 'biases':
            ax.plot(cur_results, 'o-', label=nice_layer_name(name))
    ax.set_xlabel('Learning Iteration', fontproperties='serif')
    ax.set_ylabel('Momentum Gradient', fontproperties='serif')

    fig.set_size_inches((6,8))
    #plt.show()
    plt.savefig('initial_gradient.png')
    plt.savefig('initial_gradient.pdf', pad_inches=0.05, bbox_inches='tight')



if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()