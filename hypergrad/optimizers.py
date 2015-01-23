import itertools as it
import numpy as np
from collections import deque
from funkyyak import grad
from exact_rep import ExactRep

def sgd(loss_fun, batches, N_iter, x, v, alphas, betas, record_learning_curve=False):
    # TODO: Warp alpha and beta to map from real-valued domains (exp and logistic?)
    def print_perf():
        pass
        if (i + 1) % iter_per_epoch == 0:
            print "End of epoch {0}: loss is {1}".format(i / iter_per_epoch,
                                                        loss_fun(X.val, batches.all_idxs))
            
    X, V = ExactRep(x), ExactRep(v)
    x_orig = X.val
    iter_per_epoch = len(batches)
    num_epochs = N_iter/len(batches) + 1
    iters = zip(range(N_iter), alphas, betas, batches * num_epochs)
    loss_grad = grad(loss_fun)
    loss_hvp = grad(lambda x, d, idxs : np.dot(loss_grad(x, idxs), d))
    learning_curve = [loss_fun(x_orig, batches.all_idxs)]
    for i, alpha, beta, batch in iters:
        V.mul(beta)
        g = loss_grad(X.val, batch)
        V.sub((1.0 - beta) * g)
        X.add(alpha * V.val)
        if record_learning_curve and (i+1) % iter_per_epoch == 0:
            learning_curve.append(loss_fun(X.val, batches.all_idxs))
        #print_perf()

    x_final = X.val
    d_x = loss_grad(X.val, batches.all_idxs)
    loss_final = loss_fun(x_final, batches.all_idxs)
    d_v = np.zeros(d_x.shape)
    d_alphas = deque()
    d_betas = deque()
    print_perf()

    for i, alpha, beta, batch in iters[::-1]:
        print_perf()
        d_v += d_x * alpha
        X.sub(alpha * V.val)
        g = loss_grad(X.val, batch)
        d_alphas.appendleft(np.dot(d_x, V.val))
        V.add((1.0 - beta) * g)
        V.div(beta)
        d_betas.appendleft(np.dot(d_v, V.val + g))
        d_x = d_x - (1.0 - beta) * loss_hvp(X.val, d_v, batch)
        d_v = d_v * beta

    d_alphas = np.array(d_alphas)
    d_betas = np.array(d_betas)

    # print "-"*80
    assert np.all(x_orig == X.val)
    return {'x_final'    : x_final,
            'learning_curve' : learning_curve,
            'loss_final' : loss_final,
            'd_x' : d_x,
            'd_v' : d_v,
            'd_alphas' : d_alphas,
            'd_betas'  : d_betas}


def sgd2(optimizing_loss, secondary_loss, batches, N_iter, x0, v0, alphas, betas, meta):
    """
    This version takes a secondary loss, and also returns gradients w.r.t. the data.
    :param optimizing_loss: The loss to be optimized by SGD.
    The first argument must be the parameters, the second must be the metaparameters,
    the third is data indicies.
    :param secondary_loss: Another loss we want to compute the gradient wrt.
    It takes parameters and metaparameters.
    :param batches: A list of slices into the data.
    :param N_iter: Number of iterations of SGD.
    :param x0: Starting parameter values.
    :param v0: Starting velocity.  Should probably be zero.
    :param alphas: Stepsize schedule.
    :param betas: Drag schedule.
    :param meta: A second parameter of the loss function that doesn't get optimized here.
    :return:
    a dict containing:
    Gradients wrt x0, v0, alphas, beta, and meta.
    """
    # TODO: Warp alpha and beta to map from real-valued domains (exp and logistic?)
    def print_perf():
        pass
        if (i + 1) % iter_per_epoch == 0:
            print "End of epoch {0}: loss is {1}".format(i / iter_per_epoch,
                optimizing_loss(X.val, meta, batches.all_idxs))

    X, V = ExactRep(x0), ExactRep(v0)
    x_orig = X.val
    iter_per_epoch = len(batches)
    num_epochs = N_iter/len(batches) + 1
    iters = zip(range(N_iter), alphas, betas, batches * num_epochs)
    L_grad      = grad(optimizing_loss)    # Gradient wrt parameters.
    M_grad      = grad(secondary_loss)     # Gradient wrt parameters.
    L_meta_grad = grad(optimizing_loss, 1) # Gradient wrt metaparameters.
    M_meta_grad = grad(secondary_loss, 1)  # Gradient wrt metaparameters.
    L_hvp      = grad(lambda x, d, idxs:
                      np.dot(L_grad(x, meta, idxs), d))    # Hessian-vector product.
    L_hvp_meta = grad(lambda x, meta, d, idxs:
                      np.dot(L_grad(x, meta, idxs), d), 1) # Returns a size(meta) output.

    learning_curve = [optimizing_loss(X.val, meta, batches.all_idxs)]
    for i, alpha, beta, batch in iters:
        V.mul(beta)
        g = L_grad(X.val, meta, batch)
        V.sub((1.0 - beta) * g)
        X.add(alpha * V.val)
        learning_curve.append(optimizing_loss(X.val, meta, batches.all_idxs))
        #print_perf()

    x_final = X.val
    dLd_x = L_grad(X.val, meta, batches.all_idxs)
    dMd_x = M_grad(X.val, meta)
    L_final = optimizing_loss(x_final, meta, batches.all_idxs)
    M_final = secondary_loss(x_final, meta)
    dLd_v = np.zeros(dLd_x.shape)
    dMd_v = np.zeros(dMd_x.shape)
    dLd_alphas = deque()
    dLd_betas  = deque()
    dMd_alphas = deque()
    dMd_betas  = deque()
    dLd_meta = L_meta_grad(X.val, meta, batches.all_idxs)
    dMd_meta = M_meta_grad(X.val, meta)
    print_perf()

    for i, alpha, beta, batch in iters[::-1]:
        #print_perf()
        dLd_v += dLd_x * alpha
        dMd_v += dMd_x * alpha
        X.sub(alpha * V.val)
        g = L_grad(X.val, meta, batch)
        dLd_alphas.appendleft(np.dot(dLd_x, V.val))
        dMd_alphas.appendleft(np.dot(dMd_x, V.val))
        V.add((1.0 - beta) * g)
        V.div(beta)
        dLd_betas.appendleft(np.dot(dLd_v, V.val + g))
        dMd_betas.appendleft(np.dot(dMd_v, V.val + g))
        dLd_x    -= (1.0 - beta) * L_hvp(X.val, dLd_v, batch)
        dMd_x    -= (1.0 - beta) * L_hvp(X.val, dMd_v, batch)
        dLd_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, dLd_v, batch)
        dMd_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, dMd_v, batch)
        dLd_v = dLd_v * beta
        dMd_v = dMd_v * beta

    dLd_alphas = np.array(dLd_alphas)
    dLd_betas = np.array(dLd_betas)

    # print "-"*80
    assert np.all(x_orig == X.val)
    return {'x_final' : x_final,
            'learning_curve' : learning_curve,
            'L_final' : L_final,
            'M_final' : M_final,
            'dLd_x' : dLd_x,
            'dMd_x' : dMd_x,
            'dLd_v' : dLd_v,
            'dMd_v' : dMd_v,
            'dLd_alphas' : dLd_alphas,
            'dMd_alphas' : dMd_alphas,
            'dLd_betas' : dLd_betas,
            'dMd_betas' : dMd_betas,
            'dLd_meta'  : dLd_meta,
            'dMd_meta'  : dMd_meta}

def sgd3(optimizing_loss, secondary_loss, x0, v0, alphas, betas, meta, callback=None):
    """Same as sgd2 but simplifies things by not bothering with grads of
    optimizing loss (can always just pass that in as the secondary loss)"""
    X, V = ExactRep(x0), ExactRep(v0)
    L_grad = grad(optimizing_loss)  # Gradient wrt parameters.
    grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
    L_hvp_x    = grad(grad_proj, 0) # Returns a size(x) output.
    L_hvp_meta = grad(grad_proj, 1) # Returns a size(meta) output.
    iters = zip(range(len(alphas)), alphas, betas)
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        V.mul(beta).sub((1.0 - beta) * g)
        X.add(alpha * V.val)
        if callback: callback(i, X.val)
    x_final = X.val
    M_grad      = grad(secondary_loss, 0)  # Gradient wrt parameters.
    M_meta_grad = grad(secondary_loss, 1)  # Gradient wrt metaparameters.
    dMd_x = M_grad(X.val, meta)
    dMd_v = np.zeros(dMd_x.shape)
    dMd_alphas = deque()
    dMd_betas  = deque()
    dMd_meta = M_meta_grad(X.val, meta)
    for i, alpha, beta in iters[::-1]:
        dMd_alphas.appendleft(np.dot(dMd_x, V.val))
        X.sub(alpha * V.val)
        g = L_grad(X.val, meta, i)
        V.add((1.0 - beta) * g).div(beta)
        dMd_v += dMd_x * alpha
        dMd_betas.appendleft(np.dot(dMd_v, V.val + g))
        dMd_x    -= (1.0 - beta) * L_hvp_x(X.val, meta, dMd_v, i)
        dMd_meta -= (1.0 - beta) * L_hvp_meta(X.val, meta, dMd_v, i)
        dMd_v    *= beta

    assert np.all(ExactRep(x0).val == X.val)
    return {'x_final' : x_final,
            'dMd_x'      : dMd_x,
            'dMd_v'      : dMd_v,
            'dMd_alphas' : dMd_alphas,
            'dMd_betas'  : dMd_betas,
            'dMd_meta'   : dMd_meta}
