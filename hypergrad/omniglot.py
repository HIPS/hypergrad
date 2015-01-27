import scipy.io
import numpy as np
import pickle
import os

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
    with open(datapath("omniglot_data.pkl")) as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    mat_to_pickle()
