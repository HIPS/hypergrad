import scipy.io
import numpy as np
import pickle
import os

NUM_CHARS = 55
NUM_ALPHABETS = 50

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

def load_data():
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    with open(datapath("omniglot_data.pkl")) as f:
        images, alphabet_labels, char_labels = pickle.load(f)
    print np.min(char_labels), np.max(char_labels)
    print np.min(alphabet_labels), np.max(alphabet_labels)
    char_labels = one_hot(char_labels, NUM_CHARS)
    alphabets = []
    for i_alphabet in range(NUM_ALPHABETS):
        cur_alphabet_idxs = np.where(alphabet_labels == i_alphabet)
        alphabets.append({'X' : images[cur_alphabet_idxs],
                          'T' : char_labels[cur_alphabet_idxs]})

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

if __name__ == "__main__":
    mat_to_pickle()
