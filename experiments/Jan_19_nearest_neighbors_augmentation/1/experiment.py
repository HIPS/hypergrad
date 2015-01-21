"""Exploring augmentation with nearest neighbors"""
import numpy as np
import numpy.random as npr
import pickle
import os

from hypergrad.nn_utils import BatchList, plot_mnist_images
from funkyyak import grad
from hypergrad.data import load_data_subset
from hypergrad.kernel_methods import weighted_neighbors_loss, make_sq_exp_kernel

L0 = 2.0
L = 28
N_pix = L * L
N_train = 10**3
N_valid = 10**3
N_test  = 10**3
batch_size = 100
mem_batch_size = 20 # Allowable batch size as constrained by memory
N_meta_iters = 50
meta_L1 = 0.001
meta_alpha = 0.1
A_init_scale = 0.01

def augment_data(data, transform):
    X, T = data
    X = np.concatenate((X, np.dot(X, transform.T)), axis=0)
    T = np.concatenate((T, T), axis=0)
    return X, T

def batchwise_function(fun):
    def new_fun(A, B, C):
        N = C[0].shape[0]
        batch_idxs = BatchList(N, mem_batch_size)
        for i, idxs in enumerate(batch_idxs):
            cur_result = fun(A, B, [x[idxs] for x in C])
            result = cur_result if i == 0 else result + cur_result
        return result / len(batch_idxs)
    return new_fun

def run():
    train_data, valid_data, test_data = load_data_subset(N_train, N_valid, N_test)
    kernel = make_sq_exp_kernel(L0)
    def loss_fun(transform, train_data, valid_data):
        train_data = augment_data(train_data, transform)
        return weighted_neighbors_loss(train_data, valid_data, kernel)
    loss_grad = batchwise_function(grad(loss_fun))
    loss_fun  = batchwise_function(loss_fun)

    batch_idxs = BatchList(N_valid, batch_size)
    A = np.eye(N_pix)
    valid_losses = [loss_fun(A, train_data, valid_data)]
    test_losses  = [loss_fun(A, train_data,  test_data)]
    A += A_init_scale * npr.randn(N_pix, N_pix)
    for meta_iter in range(N_meta_iters):
        print "Iter {0} valid {1} test {2}".format(
            meta_iter, valid_losses[-1], test_losses[-1])

        for idxs in batch_idxs:
            valid_batch = [x[idxs] for x in valid_data]
            d_A        = loss_grad(A, train_data, valid_batch)
            A -= meta_alpha * (d_A + meta_L1 * np.sign(A))
        valid_losses.append(loss_fun(A, train_data, valid_data))
        test_losses.append( loss_fun(A, train_data, test_data))

    return A, valid_losses, test_losses

def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        A, valid_losses, test_losses = pickle.load(f)
    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((8,12))

    ax = fig.add_subplot(211)
    ax.set_title("Meta learning curves")
    ax.plot(valid_losses, 'o-', label="Validation")
    ax.plot(test_losses , 'o-', label="Test")
    ax.set_ylabel("Negative log prob")
    ax.set_xlabel("Step number")
    ax.legend(loc=1, frameon=False)

    ax = fig.add_subplot(212)
    test_images = build_test_images()
    transformed_images = np.dot(test_images, A.T)
    cat_images = np.concatenate((test_images, transformed_images))
    plot_mnist_images(cat_images, ax, ims_per_row=test_images.shape[0])

    plt.savefig("/tmp/fig.png")
    plt.savefig("fig.png")

def build_test_images():
    if os.path.isfile('test_images.pkl'):
        with open('test_images.pkl') as f:
            all_images = pickle.load(f)
    else:
        vert_stripes = np.zeros((L, L))
        vert_stripes[:, ::4] = 1.0
        vert_stripes[:, 1::4] = 1.0
        horz_stripes = np.zeros((L, L))
        horz_stripes[::4, :] = 1.0
        horz_stripes[1::4, :] = 1.0
        mnist_imgs = load_data_subset(4)[0][0]
        all_images = np.concatenate((vert_stripes.reshape(1, L * L),
                                     horz_stripes.reshape(1, L * L),
                                     mnist_imgs), axis=0)
        with open('test_images.pkl', 'w') as f:
            pickle.dump(all_images, f)

    return all_images

if __name__=="__main__":
    # results = run()
    # with open('results.pkl', 'w') as f:
    #     pickle.dump(results, f)
    plot()
