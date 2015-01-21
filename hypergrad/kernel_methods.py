import numpy as np

def make_exp_kernel(L0):
    def exp_kernel(x1, x2):
        x1 = np.expand_dims(x1, 2) # Append a singleton dimension
        x2 = x2.T
        return np.exp(-np.mean(np.abs(x1 - x2), axis=1) / L0)
    return exp_kernel

def make_sq_exp_kernel(L0):
    def sq_exp_kernel(x1, x2):
        x1 = np.expand_dims(x1, 2) # Append a singleton dimension
        x2 = x2.T
        return np.exp(-np.sum((x1 - x2)**2, axis=1) / (2 * L0**2))
    return sq_exp_kernel

def weighted_neighbors_loss(train_data, valid_data, kernel):
    """Computes the negative log prob per data point."""
    X_train, T_train = train_data
    X_valid, T_valid = valid_data
    weight_mat = kernel(X_valid, X_train)
    label_probs = np.dot(weight_mat, T_train)
    label_probs = label_probs / np.sum(label_probs, axis=1, keepdims=True)
    mean_neg_log_prob = - np.mean(np.log(np.sum(label_probs * T_valid,
                                              axis=1)), axis=0)
    return mean_neg_log_prob
