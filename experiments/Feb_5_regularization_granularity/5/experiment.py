"""Runs for paper"""
import numpy as np
import pickle
from collections import defaultdict
from funkyyak import grad, kylist, getval

import hypergrad.mnist as mnist
from hypergrad.mnist import random_partition
from hypergrad.nn_utils import make_nn_funs, VectorParser
from hypergrad.optimizers import sgd_meta_only as sgd
from hypergrad.util import RandomState, dictslice, dictmap
from hypergrad.odyssey import omap

layer_sizes = [784, 10]
N_layers = len(layer_sizes) - 1
batch_size = 50
N_iters = 200
N_train = 10**2 * 2
N_valid = 10**2
N_tests = 10**3
all_N_meta_iter = [10, 5, 3]
alpha = 1.0
meta_alpha = 10**4
beta = 0.8
seed = 0
N_thin = 50
N_meta_thin = 1
log_L2_init = -6.0

def run():
    RS = RandomState((seed, "top_rs"))
    all_data = mnist.load_data_as_dict()
    train_data, tests_data = random_partition(all_data, RS, [N_train, N_tests])
    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size
    init_scales = w_parser.new_vect(np.zeros(N_weights))
    for i in range(N_layers):
        init_scales[('weights', i)] = 1 / np.sqrt(layer_sizes[i])
        init_scales[('biases',  i)] = 1.0
    init_scales = init_scales.vect

    def regularization(w_vect, reg):
        return np.dot(w_vect, w_vect * np.exp(reg))

    def constrain_reg(t_vect, name):
        all_r = w_parser.new_vect(t_vect)
        for i in range(N_layers):
            all_r[('biases', i)] = 0.0
        if name == 'universal':
            r_mean = np.mean([np.mean(all_r[('weights', i)]) for i in range(N_layers)])
            for i in range(N_layers):
                all_r[('weights', i)] = r_mean
        elif name == 'layers':
            for i in range(N_layers):
                all_r[('weights', i)] = np.mean(all_r[('weights', i)])
        elif name == 'units':
            for i in range(N_layers):
                all_r[('weights', i)] = np.mean(all_r[('weights', i)], axis=1, keepdims=True)
        else:
            raise Exception
        return all_r.vect

    def process_reg(t_vect):
        # Remove the redundancy due to sharing regularization within units
        all_r = w_parser.new_vect(t_vect)
        new_r = np.zeros((0,))
        for i in range(N_layers):
            layer = all_r[('weights', i)]
            assert np.all(layer[:, 0] == layer[:, 1])
            cur_r = layer[:, 0]
            new_r = np.concatenate((new_r, cur_r))
        return new_r

    def train_z(data, w_vect_0, reg):
        N_data = data['X'].shape[0]
        def primal_loss(w_vect, reg, i_primal, record_results=False):
            RS = RandomState((seed, i_primal, "primal"))
            idxs = RS.randint(N_data, size=batch_size)
            minibatch = dictslice(data, idxs)
            loss = loss_fun(w_vect, **minibatch)
            reg = regularization(w_vect, reg)
            if record_results and i_primal % N_thin == 0:
                print "Iter {0}: train: {1}".format(i_primal, getval(loss))
            return loss + reg
        return sgd(grad(primal_loss), reg, w_vect_0, alpha, beta, N_iters)

    all_regs, all_tests_loss = [], []
    def train_reg(reg_0, constraint, N_meta_iter, i_top):
        def hyperloss(reg, i_hyper, cur_train_data, cur_valid_data):
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            w_vect_0 = RS.randn(N_weights) * init_scales
            w_vect_final = train_z(cur_train_data, w_vect_0, reg)
            return loss_fun(w_vect_final, **cur_valid_data)
        hypergrad = grad(hyperloss)
        cur_reg = reg_0
        for i_hyper in range(N_meta_iter):
            if i_hyper % N_meta_thin == 0:
                tests_loss = hyperloss(cur_reg, i_hyper, train_data, tests_data)
                all_tests_loss.append(tests_loss)
                all_regs.append(cur_reg.copy())
                print "Hyper iter {0}, test loss {1}".format(i_hyper, all_tests_loss[-1])
                print "Cur_reg", np.mean(cur_reg)
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            cur_split = random_partition(train_data, RS, [N_train - N_valid, N_valid])
            raw_grad = hypergrad(cur_reg, i_hyper, *cur_split)
            constrained_grad = constrain_reg(raw_grad, constraint)
            print constrained_grad
            # cur_reg -= constrained_grad / np.abs(constrained_grad + 1e-8) * meta_alpha
            cur_reg -= constrained_grad * meta_alpha

        return cur_reg


    def new_hyperloss(reg, i_hyper, cur_train_data, cur_valid_data):
        RS = RandomState((seed, i_hyper, "hyperloss"))
        w_vect_0 = RS.randn(N_weights) * init_scales
        w_vect_final = train_z(cur_train_data, w_vect_0, reg)
        return loss_fun(w_vect_final, **cur_valid_data)

    # t_scale = [-1, 0, 1]
    # cur_split = random_partition(train_data, RS, [N_train - N_valid, N_valid])
    # for s in t_scale:
    #     reg = np.ones(N_weights) * log_L2_init + s
    #     loss = new_hyperloss(reg, 0, *cur_split)
    #     print "Results: s= {0}, loss = {1}".format(s, loss)

    reg = np.ones(N_weights) * log_L2_init
    constraints = ['universal', 'layers', 'units']
    for i_top, (N_meta_iter, constraint) in enumerate(zip(all_N_meta_iter, constraints)):
        print "Top level iter {0}".format(i_top)
        reg = train_reg(reg, constraint, N_meta_iter, i_top)

    all_L2_regs = np.array(zip(*map(process_reg, all_regs)))
    return all_L2_regs, all_tests_loss

def plot():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['image.interpolation'] = 'none'
    with open('results.pkl') as f:
        all_L2_regs, all_tests_loss = pickle.load(f)

    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(211)
    color_cycle = ['RoyalBlue', 'DarkOliveGreen', 'DarkOrange', 'MidnightBlue']
    colors = []
    for i, size in enumerate(layer_sizes[:-1]):
        colors += [color_cycle[i]] * size
    for c, L2_reg_curve in zip(colors, all_L2_regs):
        ax.plot(L2_reg_curve, color=c)
    ax.set_ylabel('Log L2 regularization')
    # ax.set_ylim([-3.0, -2.5])
    ax = fig.add_subplot(212)
    ax.plot(all_tests_loss)
    ax.set_ylabel('Test loss')
    ax.set_xlabel('Meta iterations')
    plt.savefig("reg_learning_curve.png")

    initial_filter = np.array(all_L2_regs)[:layer_sizes[0], -1].reshape((28, 28))
    fig.clf()
    fig.set_size_inches((5, 5))
    ax = fig.add_subplot(111)
    ax.matshow(initial_filter, cmap = mpl.cm.binary)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('bottom_layer_filter.png')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
