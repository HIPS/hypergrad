import numpy as np
import pickle

from funkyyak import grad, kylist, getval

import hypergrad.omniglot as omniglot
from hypergrad.nn_utils import make_nn_funs, VectorParser
from hypergrad.optimizers import sgd_meta_only as sgd, rms_prop
from hypergrad.util import RandomState, dictslice

# ----- Fixed params -----
layer_sizes = [784, 100, 55]
batch_size = 200
N_iters = 50
N_test_alphabets = 10
N_valid_dpts = 100
alpha = 0.1
beta = 0.9
seed = 0
# ----- Superparameters -----
meta_alpha = 0.1
N_meta_iter = 20
# ----- Initial values of learned hyper-parameters -----
log_scale_std  = 0.1
log_scale_mean = 0.0
offset_std = 0.1

def run():
    RS = RandomState(seed)
    all_alphabets = omniglot.load_data()
    RS.shuffle(all_alphabets)
    train_alphabets = all_alphabets[:-N_test_alphabets]
    tests_alphabets = all_alphabets[-N_test_alphabets:]
    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size
    hyperparams_0 = VectorParser()
    hyperparams_0['log_scale']  = log_scale_std * RS.randn(N_weights) + log_scale_mean 
    hyperparams_0['offset']     =    offset_std * RS.randn(N_weights)

    def reg_loss_fun(W, data, hyperparams, reg_penalty):
        Z = np.exp(hyperparams['log_scale']) * W + hyperparams['offset']
        return loss_fun(Z, **data) + np.dot(Z, Z)

    def hyperloss(hyperparam_vect, i_hyper):
        RS = RandomState((seed, i_hyper))        
        alphabet = shuffle_alphabet(RS.choice(train_alphabets), RS)
        N_train = alphabet['X'].shape[0] - N_valid_dpts
        train_data = dictslice(alphabet, slice(None, N_train))
        valid_data = dictslice(alphabet, slice(N_train, None))
        def primal_loss(W, hyperparam_vect, i_primal):
            RS = RandomState((seed, i_hyper, i_primal))
            idxs = RS.permutation(N_train)[:batch_size]
            minibatch = dictslice(train_data, idxs)
            cur_hyperparams = hyperparams_0.new_vect(hyperparam_vect)
            return reg_loss_fun(W, minibatch, cur_hyperparams, reg_penalty=True)

        def callback(*args):
            print ".",

        W0 = np.zeros(N_weights)
        W_final = sgd(grad(primal_loss), hyperparam_vect, W0, alpha, beta, N_iters, callback)
        return reg_loss_fun(W_final, valid_data, hyperparam_vect, reg_penalty=False)

    def meta_callback(*args):
        print "Meta epoch {0}".format(0)

    final_result = rms_prop(grad(hyperloss), hyperparams_0.vect,
                            meta_callback, N_meta_iter, meta_alpha)

def shuffle_alphabet(alphabet, RS):
    # Shuffles both data and label indices
    N_rows, N_cols = alphabet['T'].shape
    alphabet['T'] = alphabet['T'][:, RS.permutation(N_cols)]
    return dictslice(alphabet, RS.permutation(N_rows))

def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser = pickle.load(f)

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
