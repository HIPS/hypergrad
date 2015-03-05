"""Showing that reversing learning dynamics is unstable."""
import numpy as np
import pickle

from hypergrad.nn_utils import VectorParser
from hypergrad.nn_utils import logit, inv_logit, d_logit

from hypergrad.exact_rep import ExactRep
from mpl_toolkits.mplot3d import proj3d
from funkyyak import grad

# ----- Fixed params -----
N_epochs = 46
N_meta_thin = 1  # How often to save meta-curve info.
init_params = np.array([1.4, 3.9])

# ----- Superparameters - aka meta-meta params that control metalearning -----
meta_alpha = 0.010
meta_gamma = 0.9 # Setting this to zero makes things much more stable
N_meta_iter = 1
# ----- Initial values of learned hyper-parameters -----
init_log_alphas = -5.5
init_invlogit_betas = inv_logit(0.8)
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


def sgd3_naive(optimizing_loss, x, v, alphas, betas, meta, fwd_callback=None, reverse_callback=None):
    """Same as sgd2 but simplifies things by not bothering with grads of
    optimizing loss (can always just pass that in as the secondary loss)"""
    x = x.astype(np.float16)
    v = v.astype(np.float16)
    L_grad = grad(optimizing_loss)  # Gradient wrt parameters.
    iters = zip(range(len(alphas)), alphas, betas)

    # Forward pass
    for i, alpha, beta in iters:
        if fwd_callback: fwd_callback(x, i)
        g = L_grad(x, meta, i)
        v = v * beta
        v = v - ((1.0 - beta) * g)
        x = x + alpha * v
        x = x.astype(np.float16)
        v = v.astype(np.float16)

    # Reverse pass
    for i, alpha, beta in iters[::-1]:
        x = x - alpha * v
        g = L_grad(x, meta, i)
        v = v + (1.0 - beta) * g
        v = v / beta
        if reverse_callback: reverse_callback(x, i)
        x = x.astype(np.float16)
        v = v.astype(np.float16)


def run():
    N_iters = N_epochs
    parser, loss_fun = make_toy_funs()
    N_weights = parser.vect.size
    hyperparams = VectorParser()
    hyperparams['log_alphas']      = np.full(N_iters, init_log_alphas)
    hyperparams['invlogit_betas']  = np.full(N_iters, init_invlogit_betas)
    hyperparams['V0']  = np.full(N_weights, init_V0)

    forward_path = []
    forward_learning_curve = []
    def fwd_callback(x, i):
        print type(x[0])
        forward_path.append(x.copy())
        forward_learning_curve.append(loss_fun(x))

    reverse_path = []
    reverse_learning_curve = []
    def reverse_callback(x, i):
        reverse_path.append(x.copy())
        reverse_learning_curve.append(loss_fun(x))

    def indexed_loss_fun(w, log_L2_reg, j):
        return loss_fun(w)

    cur_hyperparams = hyperparams
    W0 = init_params
    V0 = cur_hyperparams['V0']
    alphas = np.exp(cur_hyperparams['log_alphas'])
    betas = logit(cur_hyperparams['invlogit_betas'])
    log_L2_reg = 0.0
    sgd3_naive(indexed_loss_fun, W0, V0,
                alphas, betas, log_L2_reg,
                fwd_callback=fwd_callback, reverse_callback=reverse_callback)

    return forward_path, forward_learning_curve, reverse_path, reverse_learning_curve

def plot():
    _, loss_fun = make_toy_funs()

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
        forward_path, forward_learning_curve, reverse_path, reverse_learning_curve = pickle.load(f)

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

    x, y = zip(*forward_path)
    z = forward_learning_curve
    ax.plot([x[0]], [y[0]], [z[0]], '*', color='Black', label="Initial weights", markersize=15)
    ax.plot(x, y, z, '-o', color=colors[0], markersize=2, linewidth=2)
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'o', color=colors[0], markersize=9)

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
    #ax.legend(numpoints=1, loc=0, frameon=True,  bbox_to_anchor=(0.65, 0.8),
    #          borderaxespad=0.0, prop={'family':'serif', 'size':'12'})
    #plt.show()
    plt.savefig('learning_curve_forward.png')
    plt.savefig('learning_curve_forward.pdf', pad_inches=0.05, bbox_inches='tight')

    ax.set_autoscale_on(False)
    x, y = zip(*reverse_path)
    z = reverse_learning_curve
    ax.plot(x, y, z, '-o', color='Blue', markersize=2, linewidth=2)

    ax.view_init(elev=46, azim=120)
    plt.savefig('learning_curve_reverse.png')
    plt.savefig('learning_curve_reverse.pdf', pad_inches=0.05, bbox_inches='tight')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f)
    plot()
