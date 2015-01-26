"""Gradient descent to optimize everything"""
"""Aiming for smooth curves by running for a long time with small steps."""
import numpy as np
import pickle

from hypergrad.nn_utils import VectorParser
from hypergrad.nn_utils import logit, inv_logit, d_logit
from hypergrad.optimizers import sgd3, rms_prop, adam, simple_sgd

# ----- Fixed params -----
N_epochs = 10
N_meta_thin = 1  # How often to save meta-curve info.
init_param_scale = -1.8

# ----- Superparameters - aka meta-meta params that control metalearning -----
meta_alpha = 0.5
meta_gamma = 0.01 # Setting this to zero makes things much more stable
N_meta_iter = 4
# ----- Initial values of learned hyper-parameters -----
init_log_alphas = -6.0
init_invlogit_betas = inv_logit(0.9)
init_V0 = 0.0


def make_toy_funs():
    parser = VectorParser()
    parser.add_shape('weights', 2)

    def rosenbrock(x):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    def loss(W_vect, X=0.0, T=0.0, L2_reg=0.0):
        return 500 * logit(rosenbrock(W_vect) / 500)

    return parser, loss


def run():
    N_iters = N_epochs
    parser, loss_fun = make_toy_funs()
    N_weight_types = len(parser.names)
    N_weights = parser.vect.size
    hyperparams = VectorParser()
    hyperparams['log_alphas']      = np.full(N_iters, init_log_alphas)
    hyperparams['invlogit_betas']  = np.full(N_iters, init_invlogit_betas)
    hyperparams['V0']  = np.full(N_weights, init_V0)

    all_learning_curves = []
    all_param_curves = []
    all_x = []
    def hyperloss_grad(hyperparam_vect, ii):
        learning_curve = []
        params_curve = []
        def callback(x, i):
            params_curve.append(x)
            learning_curve.append(loss_fun(x))

        def indexed_loss_fun(w, log_L2_reg, j):
            return loss_fun(w)

        cur_hyperparams = hyperparams.new_vect(hyperparam_vect)
        W0 = np.ones(N_weights) * init_param_scale
        V0 = cur_hyperparams['V0']
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas = logit(cur_hyperparams['invlogit_betas'])
        log_L2_reg = 0.0
        results = sgd3(indexed_loss_fun, loss_fun, W0, V0,
                       alphas, betas, log_L2_reg, callback=callback)
        hypergrads = hyperparams.copy()
        hypergrads['V0']              = results['dMd_v']
        hypergrads['log_alphas']      = results['dMd_alphas'] * alphas
        hypergrads['invlogit_betas']  = (results['dMd_betas'] *
                                         d_logit(cur_hyperparams['invlogit_betas']))
        all_x.append(results['x_final'])
        all_learning_curves.append(learning_curve)
        all_param_curves.append(params_curve)
        return hypergrads.vect

    add_fields = ['train_loss', 'valid_loss', 'tests_loss', 'iter_num']
    meta_results = {field : [] for field in add_fields + hyperparams.names}
    def meta_callback(hyperparam_vect, i):
        if i % N_meta_thin == 0:
            print "Meta iter {0}".format(i)
            x = all_x[-1]
            cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
            for field in cur_hyperparams.names:
                meta_results[field].append(cur_hyperparams[field])
            meta_results['train_loss'].append(loss_fun(x))
            meta_results['iter_num'].append(i)

    final_result = rms_prop(hyperloss_grad, hyperparams.vect,
                            meta_callback, N_meta_iter, meta_alpha, meta_gamma)
    meta_results['all_learning_curves'] = all_learning_curves
    meta_results['all_param_curves'] = all_param_curves
    parser.vect = None # No need to pickle zeros
    return meta_results, parser

def plot():
    _, loss_fun = make_toy_funs()

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser = pickle.load(f)

    fig = plt.figure(0)
    fig.set_size_inches((4,3))

    # Show loss surface.
    x = np.arange(-2.0, 2.0, 0.05)
    y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([loss_fun(np.concatenate(([x],[y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    # ----- Primal learning curves -----
    for i, z in zip(results['iter_num'], results['all_learning_curves']):
        x, y = zip(*results['all_param_curves'][i])
        ax.plot(x, y, z, 'o-', label='Meta iter {0}'.format(i))
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=45, azim=60)
    #ax.legend(loc=1, frameon=False)

    #plt.show()
    plt.savefig('learning_curves.png')
    plt.savefig('learning_curves.pdf')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()
