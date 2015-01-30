""" Just exploring the initialization meta params for now """

import numpy as np
import pickle
from collections import defaultdict
from functools import partial
from funkyyak import grad, kylist, getval

import hypergrad.omniglot as omniglot
from hypergrad.nn_utils import make_nn_funs, VectorParser
from hypergrad.optimizers import sgd_meta_only as sgd, rms_prop
from hypergrad.util import RandomState, dictslice

# ----- Fixed params -----
layer_sizes = [784, 200, 55]
batch_size = 200
N_iters = 100
N_test_alphabets = 20
N_valid_dpts = 100
alpha = 0.05
beta = 0.9
seed = 0
N_alphabets_eval = 1
# ----- Superparameters -----
meta_alpha = 0.05
N_meta_iter = 1
N_hyper_thin = 20
# ----- Initial values of learned hyper-parameters -----
log_scale_init = 1.0
offset_init_std = 0.1

def run(superparams):
    alpha, log_scale_init, offset_init_std = superparams
    RS = RandomState((seed, "top_rs"))
    all_alphabets = omniglot.load_data()
    RS.shuffle(all_alphabets)
    train_alphabets = all_alphabets[:-N_test_alphabets]
    tests_alphabets = all_alphabets[-N_test_alphabets:]
    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size
    hyperparams_0 = VectorParser()
    hyperparams_0['log_scale']  = log_scale_init * np.ones(N_weights)
    hyperparams_0['offset'] = offset_init_std * RS.randn(N_weights)

    def reg_loss_fun(W, data, hyperparam_vect, reg_penalty):
        hyperparams = hyperparams_0.new_vect(hyperparam_vect)
        Z = np.exp(hyperparams['log_scale']) * W + hyperparams['offset']
        return loss_fun(Z, **data) + np.dot(W, W) * reg_penalty

    def hyperloss(hyperparam_vect, i_hyper, alphabets, verbose=True, report_train_loss=False):
        RS = RandomState((seed, i_hyper, "hyperloss"))        
        alphabet = shuffle_alphabet(RS.choice(alphabets), RS)
        N_train = alphabet['X'].shape[0] - N_valid_dpts
        train_data = dictslice(alphabet, slice(None, N_train))
        if report_train_loss:
            valid_data = dictslice(alphabet, slice(None, N_valid_dpts))
        else:
            valid_data = dictslice(alphabet, slice(N_train, None))
        def primal_loss(W, hyperparam_vect, i_primal, reg_penalty=True):
            RS = RandomState((seed, i_hyper, i_primal))
            idxs = RS.permutation(N_train)[:batch_size]
            minibatch = dictslice(train_data, idxs)
            loss = reg_loss_fun(W, minibatch, hyperparam_vect, reg_penalty)
            if verbose and i_primal % 30 == 0:
                print "Iter {0}, loss, {1}".format(i_primal, getval(loss))
                
            return loss

        W0 = np.zeros(N_weights)
        W_final = sgd(grad(primal_loss), hyperparam_vect, W0, alpha, beta, N_iters, callback=None)
        return reg_loss_fun(W_final, valid_data, hyperparam_vect, reg_penalty=False)

    results = defaultdict(list)
    def record_results(hyperparam_vect, i_hyper, g):
        # print "Meta iter {0}. Recording results".format(i_hyper)
        RS = RandomState((seed, i_hyper, "evaluation"))
        new_seed = RS.int32()
        def loss_fun(alphabets, report_train_loss):
            return np.mean([hyperloss(hyperparam_vect, new_seed, alphabets=alphabets,
                                      verbose=False, report_train_loss=report_train_loss)
                            for i in range(N_alphabets_eval)])
        cur_hyperparams = hyperparams_0.new_vect(hyperparam_vect.copy())
        if i_hyper % N_hyper_thin == 0:
            # Storing O(N_weights) is a bit expensive so we thin it out and store in low precision
            for field in cur_hyperparams.names:
                results[field].append(cur_hyperparams[field].astype(np.float16))
        results['train_loss'].append(loss_fun(train_alphabets, report_train_loss=True))
        results['valid_loss'].append(loss_fun(train_alphabets, report_train_loss=False))

    record_results(hyperparams_0.vect, 0, None)
    return [results['train_loss'][0], results['valid_loss'][0]]

def shuffle_alphabet(alphabet, RS):
    # Shuffles both data and label indices
    N_rows, N_cols = alphabet['T'].shape
    alphabet['T'] = alphabet['T'][:, RS.permutation(N_cols)]
    return dictslice(alphabet, RS.permutation(N_rows))

if __name__ == '__main__':
    print "Exploring superparams. Here are all the constants:"
    print globals()
    alphas          = [0.1] * 3
    log_scale_inits = [2.0] * 3
    offset_init_stds = [0.01, 0.1, 0.5] * 3
    superparams = zip(alphas, log_scale_inits, offset_init_stds)
    results = map(run, superparams)
    for s, r in zip(superparams, results):
        print "Superparams:", s
        print "Train/valid loss:", r

    with open('results.pkl', 'w') as f:
        pickle.dump((superparams, results), f, 1)
