import numpy as np
import pickle
from collections import defaultdict, OrderedDict
from functools import partial
from funkyyak import grad, kylist, getval
import itertools as it
from copy import copy

import hypergrad.omniglot as omniglot
from hypergrad.nn_utils import make_nn_funs, VectorParser
from hypergrad.optimizers import sgd_meta_only as sgd, rms_prop
from hypergrad.util import RandomState, dictslice
from hypergrad.odyssey import omap

VERBOSE = True
# ----- Fixed params -----
layer_sizes = [784, 400, 200, 55]
N_layers = len(layer_sizes) - 1
N_scripts = 50
N_iters = 50
N_thin = 10
alpha = 1.0
beta = 0.9
seed = 0
# ----- Superparameters -----
log_initialization_scale = -2.0
log_L2_init = -4.0
script_corr_init = 0.5
N_meta_iter = 10
meta_alpha = 0.01
line_search_dists = np.linspace(-3, 3, 10)

# # TEST RUN:
# N_scripts = 5
# N_iters = 5
# N_meta_iter = 3
# line_search_dists = np.linspace(-1, 1, 3)

def run():
    """Three different parsers:
    w_parser[('biases', i_layer)] : neural net weights/biases per layer for a single  script
    script_parser[i_script]       : weights vector for each script
    transform_parser[i_layer]     : transform matrix (scripts x scripts) for each alphabet"""
    RS = RandomState((seed, "top_rs"))
    train_data, valid_data, tests_data = omniglot.load_data_split(
        [11, 2, 2], RS, num_alphabets=N_scripts)
    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size
    transform_parser = make_transform(N_scripts, script_corr_init)
    script_parser = VectorParser()
    for i_script in range(N_scripts):
        script_parser[i_script] = np.zeros(N_weights)

    def get_layers(vect):
        layers = []
        for i_layer in range(N_layers):
            weights_by_scripts = vect.reshape((N_scripts, N_weights))
            weights_idxs, _ = w_parser.idxs_and_shapes[('weights', i_layer)]
            biases_idxs, _  = w_parser.idxs_and_shapes[('biases',  i_layer)]
            assert weights_idxs.stop == biases_idxs.start
            layer_idxs = slice(weights_idxs.start, biases_idxs.stop)
            layers.append(weights_by_scripts[:, layer_idxs])
        return layers

    def transform_weights(z_vect, transform_vect):
        z_layers = get_layers(z_vect)
        transform = transform_parser.new_vect(transform_vect)
        w_layers = [np.dot(transform[i], z) for i, z in enumerate(z_layers)]
        return np.concatenate(w_layers, axis=1).ravel()

    def total_loss(w_vect, data):
        w = script_parser.new_vect(w_vect)
        return sum([loss_fun(w[i], **script_data) for i, script_data in enumerate(data)])

    def regularization(z_vect):
        return np.dot(z_vect, z_vect) * np.exp(log_L2_init)

    results = defaultdict(list)
    def hyperloss(transform_vect, i_hyper, record_results=True):
        RS = RandomState((seed, i_hyper, "hyperloss"))
        def primal_loss(z_vect, transform_vect, i_primal, record_results=False):
            w_vect = transform_weights(z_vect, transform_vect)
            loss = total_loss(w_vect, train_data)
            reg = regularization(z_vect)
            if VERBOSE and record_results and i_primal % N_thin == 0:
                print "Iter {0}: train: {1}, valid: {2}, reg: {3}".format(
                    i_primal,
                    getval(loss) / N_scripts,
                    total_loss(getval(w_vect), valid_data) / N_scripts,
                    getval(reg))
            return loss + reg

        z_vect_0 = RS.randn(script_parser.vect.size) * np.exp(log_initialization_scale)
        z_vect_final = sgd(grad(primal_loss), transform_vect, z_vect_0,
                           alpha, beta, N_iters, callback=None)
        w_vect_final = transform_weights(z_vect_final, transform_vect)
        valid_loss = total_loss(w_vect_final, valid_data)
        if record_results:
            results['valid_loss'].append(getval(valid_loss) / N_scripts) 
            results['train_loss'].append(total_loss(w_vect_final, train_data) / N_scripts)
            results['tests_loss'].append(total_loss(w_vect_final, tests_data) / N_scripts)
        return valid_loss

    grad_transform = grad(hyperloss)(transform_parser.vect, 0, record_results=False)
    for i, d in enumerate(line_search_dists):
        new_transform_vect = transform_parser.vect - d * grad_transform
        hyperloss(new_transform_vect, 0, record_results=True)
        print "Hyper iter {0}".format(i)
        print "Results", {k : v[-1] for k, v in results.iteritems()}
        
    grad_transform_dict = transform_parser.new_vect(grad_transform).as_dict()
    return results, grad_transform_dict

def make_transform(N_scripts, corr):
    uncorrelated_mat = np.eye(N_scripts)
    fully_correlated_mat = np.full((N_scripts, N_scripts), 1.0 / N_scripts)
    transform_mat = (1 - corr) * uncorrelated_mat + corr * fully_correlated_mat
    transform_parser = VectorParser()
    for i_layer in range(N_layers):
        if i_layer > 0:
            transform_parser[i_layer] = uncorrelated_mat
        else:
            transform_parser[i_layer] = transform_mat
    return transform_parser

def plot():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    with open('results.pkl') as f:
        results, grad_transform_dict = pickle.load(f)

    fig = plt.figure(0)
    fig.set_size_inches((6,4))
    ax = fig.add_subplot(111)
    ax.set_title('Performance vs hyper step')
    for k, v in results.iteritems():
        ax.plot(line_search_dists, v, 'o-', label=k)
    ax.set_xlabel('Step size')
    ax.set_ylabel('Negative log prob')
    ax.legend(loc=1, frameon=False)
    plt.savefig('performance.png')

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((6,8))
    for i in range(3):
        image = grad_transform_dict[i]
        img_min, img_max = (-0.5, 0.5)
        image = np.minimum(np.maximum(image, img_min), img_max)
        ax = fig.add_subplot(3,1,i)
        ax.imshow(image, cmap = mpl.cm.binary)
        ax.set_title('Layer {0}'.format(i))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig('Grad_transforms.png')

if __name__ == '__main__':
    # results = run()
    # with open('results.pkl', 'w') as f:
    #     pickle.dump(results, f, 1)
    plot()
