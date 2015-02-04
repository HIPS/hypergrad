import scipy.io
import numpy as np
import pickle
import os
import numpy.random as npr

from hypergrad.util import dictslice
NUM_CHARS = 55
NUM_ALPHABETS = 50
NUM_EXAMPLES = 15
CURATED_ALPHABETS = [6, 10, 23, 38, 39, 8, 9, 21, 22, 41]

def datapath(fname):
    datadir = os.path.expanduser('~/repos/hypergrad/data/omniglot')
    return os.path.join(datadir, fname)

def mat_to_pickle():
    data = scipy.io.loadmat(datapath('chardata.mat'))
    images = data['data'].T.astype(np.float16) # Flattened images, (24345, 784) in range [0, 1]
    alphabet_labels = np.argmax(data['target'], axis=0) # (24345, ) ints representing alphabets
    char_labels = data['targetchar'][0, :] - 1 # (24345, ) ints representing characters
    with open(datapath("omniglot_data.pkl"), "w") as f:
        pickle.dump((images, alphabet_labels, char_labels), f, 1)

def load_data(alphabets_to_load=range(NUM_ALPHABETS)):
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    with open(datapath("omniglot_data.pkl")) as f:
        images, alphabet_labels, char_labels = pickle.load(f)
    # print np.min(char_labels), np.max(char_labels)
    # print np.min(alphabet_labels), np.max(alphabet_labels)
    char_labels = one_hot(char_labels, NUM_CHARS)
    alphabets = []
    for i_alphabet in alphabets_to_load:
        cur_alphabet_idxs = np.where(alphabet_labels == i_alphabet)
        alphabets.append({'X' : images[cur_alphabet_idxs],
                          'T' : char_labels[cur_alphabet_idxs]})
    return alphabets

def load_data_split(num_chars, RS, num_alphabets=NUM_ALPHABETS):
    alphabets_to_load = RS.choice(range(NUM_ALPHABETS), size=num_alphabets, replace=False)
    raw_data = load_data(np.sort(alphabets_to_load))
    shuffled_data = [shuffle(alphabet, RS) for alphabet in raw_data]
    data_split = zip(*[split(alphabet, num_chars) for alphabet in shuffled_data])
    normalized_data = [subtract_mean(data_subset) for data_subset in data_split]
    return normalized_data

def load_curated_alphabets(num_chars):
    raw_data = load_data(CURATED_ALPHABETS)
    shuffled_data = [shuffle(alphabet, RS) for alphabet in raw_data]
    data_split = zip(*[split(alphabet, num_chars) for alphabet in shuffled_data])
    normalized_data = [subtract_mean(data_subset) for data_subset in data_split]
    return normalized_data

def split(alphabet, num_chars):
    assert sum(num_chars) == NUM_EXAMPLES
    cum_chars = np.cumsum(num_chars)
    def select_dataset(count):
        for i, N in enumerate(cum_chars):
            if count < N: return i

    labels = np.argmax(alphabet['T'], axis=1)
    label_counts = [0] * NUM_CHARS
    split_idxs = [[] for n in num_chars]
    for i_dpt, label in enumerate(labels):
        i_dataset = select_dataset(label_counts[label])
        split_idxs[i_dataset].append(i_dpt)
        label_counts[label] += 1

    data_splits = []
    for n, idxs in zip(num_chars, split_idxs):
        data_splits.append(dictslice(alphabet, idxs))
        totals = np.sum(data_splits[-1]['T'], axis=0)
        assert np.all(np.logical_or(totals == 0, totals == n))

    return data_splits
    
def shuffle(alphabet, RS):
    N_rows, N_cols = alphabet['T'].shape
    alphabet['T'] = alphabet['T'][:, RS.permutation(N_cols)]
    return dictslice(alphabet, RS.permutation(N_rows))

def subtract_mean(alphabets):
    all_images = np.concatenate([alphabet['X'] for alphabet in alphabets], axis=0)
    assert np.all(all_images >= 0) and np.all(all_images <= 1)
    mean_img = np.mean(all_images, axis=0)
    for alphabet in alphabets:
        alphabet['X'] = alphabet['X'] - mean_img
    return alphabets

def show_all_chars():
    import matplotlib.pyplot as plt
    from nn_utils import plot_images
    alphabets = load_data()
    fig = plt.figure()
    fig.set_size_inches((12,8))
    n_rows = 8
    n_cols = 20
    for i in range(n_rows):
        i_alphabet = np.random.randint(NUM_ALPHABETS)
        alphabet = alphabets[i_alphabet]
        char_idxs = np.random.randint(alphabet['X'].shape[0], size=n_cols)
        char_ids = np.argmax(alphabet['T'][char_idxs], axis=1)
        ax = fig.add_subplot(n_rows, 1, i)
        plot_images(alphabet['X'][char_idxs], ax, ims_per_row=n_cols)
        ax.set_title("Alphabet {0}, chars {1}".format(i_alphabet, char_ids))
    plt.savefig("random_images.png")

def show_curated_alphabets():
    show_all_alphabets(CURATED_ALPHABETS)

def show_all_alphabets(perm=None):
    if perm is None:
        perm = range(len(alphabets))
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from nn_utils import plot_images
    n_cols = 20
    full_image = np.zeros((0, n_cols * 28))
    alphabets = load_data()
    for i_alphabet in perm:
        alphabet = alphabets[i_alphabet]
        char_idxs = np.random.randint(alphabet['X'].shape[0], size=n_cols)
        char_ids = np.argmax(alphabet['T'][char_idxs], axis=1)
        image = alphabet['X'][char_idxs].reshape((n_cols, 28, 28))
        image = np.transpose(image, axes=[1, 0, 2]).reshape((28, n_cols * 28))
        full_image = np.concatenate((full_image, image))

    fig = plt.figure()
    fig.set_size_inches((8,12))
    ax = fig.add_subplot(111)
    ax.imshow(full_image, cmap=mpl.cm.binary)
    ax.set_xticks(np.array([]))
    ax.set_yticks(np.array([]))
    plt.tight_layout()
    plt.savefig("all_alphabets.png")

if __name__ == "__main__":
    mat_to_pickle()
