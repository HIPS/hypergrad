"""Gradient descent to optimize everything"""
"""Aiming for smooth curves by running for a long time with small steps."""
import numpy as np
import pickle

from hypergrad.nn_utils import VectorParser
from hypergrad.nn_utils import logit, inv_logit, d_logit
from hypergrad.optimizers import sgd3, rms_prop, adam, simple_sgd

# ----- Fixed params -----
N_epochs = 50
N_meta_thin = 1  # How often to save meta-curve info.
init_params = np.array([1.4, 3.9])

# ----- Superparameters - aka meta-meta params that control metalearning -----
meta_alpha = 0.010
meta_gamma = 0.9 # Setting this to zero makes things much more stable
N_meta_iter = 3
# ----- Initial values of learned hyper-parameters -----
init_log_alphas = -4.5
init_invlogit_betas = inv_logit(0.99)
init_V0 = 0.0


def make_toy_funs():
    parser = VectorParser()
    parser.add_shape('weights', 2)

    def rosenbrock(w):
        x = w[1:]
        y = w[:-1]
        return sum(100.0*(x-y**2.0)**2.0 + (1-y)**2.0 + 200.0*y)

    def loss(W_vect, X=0.0, T=0.0, L2_reg=0.0):
        return 800 * logit(rosenbrock(W_vect) / 500)

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
        W0 = init_params
        V0 = cur_hyperparams['V0']
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas = logit(cur_hyperparams['invlogit_betas'])
        log_L2_reg = 0.0
        results = sgd3(indexed_loss_fun, loss_fun, W0, V0,
                       alphas, betas, log_L2_reg, callback=callback)
        hypergrads = hyperparams.copy()
        hypergrads['V0']              = results['dMd_v'] * 0
        hypergrads['log_alphas']      = results['dMd_alphas'] * alphas
        hypergrads['invlogit_betas']  = (results['dMd_betas'] *
                                         d_logit(cur_hyperparams['invlogit_betas']))
        all_x.append(results['x_final'])
        all_learning_curves.append(learning_curve)
        all_param_curves.append(params_curve)
        return hypergrads.vect

    add_fields = ['train_loss', 'valid_loss', 'tests_loss', 'iter_num']
    meta_results = {field : [] for field in add_fields + hyperparams.names}
    def meta_callback(hyperparam_vect, i, g):
        if i % N_meta_thin == 0:
            print "Meta iter {0}".format(i)
            x = all_x[-1]
            cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
            for field in cur_hyperparams.names:
                meta_results[field].append(cur_hyperparams[field])
            meta_results['train_loss'].append(loss_fun(x))
            meta_results['iter_num'].append(i)

    final_result = simple_sgd(hyperloss_grad, hyperparams.vect,
                            meta_callback, N_meta_iter, meta_alpha, meta_gamma)
    meta_results['all_learning_curves'] = all_learning_curves
    meta_results['all_param_curves'] = all_param_curves
    parser.vect = None # No need to pickle zeros
    return meta_results, parser

def plot():
    _, loss_fun = make_toy_funs()

    from mpl_toolkits.mplot3d import proj3d

    def orthogonal_proj(zfront, zback):
        a = (zfront+zback)/(zfront-zback)
        b = -2*(zfront*zback)/(zfront-zback)
        # -0.0001 added for numerical stability as suggested in:
        # http://stackoverflow.com/questions/23840756
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,a,b],
                         [0,0,-0.0001,zback]])

    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser = pickle.load(f)

    fig = plt.figure(0)
    fig.set_size_inches((6,4))

    # Show loss surface.
    x = np.arange(-1.0, 2.4, 0.05)
    y = np.arange(-0.0, 4.5, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([loss_fun(np.concatenate(([x],[y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color='Black')
    proj3d.persp_transformation = orthogonal_proj

    colors = ['Red', 'Green', 'Blue']

    # ----- Primal learning curves -----
    for i, z in zip(results['iter_num'], results['all_learning_curves']):
        x, y = zip(*results['all_param_curves'][i])
        if i == 0:
            ax.plot([x[1]], [y[1]], [z[1]], '*', color='Black', label="Initial weights", markersize=15)
        ax.plot(x, y, z, '-o', color=colors[i], markersize=2, linewidth=2)
        ax.plot([x[-1]], [y[-1]], [z[-1]], 'o', color=colors[i], label='Meta-iteration {0}'.format(i+1), markersize=9)

    ax.set_xlabel('Weight 1', fontproperties='serif')
    ax.set_ylabel('Weight 2', fontproperties='serif')
    #ax.set_zlabel('Training Loss', fontproperties='serif', rotation=90)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_ylim(ax.get_ylim()[::-1])  # Swap y-axis
    ax.view_init(elev=46, azim=120)
    fig.tight_layout()
    ax.legend(numpoints=1, loc=0, frameon=True,  bbox_to_anchor=(0.65, 0.8),
              borderaxespad=0.0, prop={'family':'serif', 'size':'12'})


    #plt.show()
    plt.savefig('learning_curves.png')
    plt.savefig('learning_curves.pdf', pad_inches=0.05, bbox_inches='tight')

if __name__ == '__main__':
    #results = run()
    #with open('results.pkl', 'w') as f:
    #    pickle.dump(results, f)
    plot()
