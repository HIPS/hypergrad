from contextlib import contextmanager
from time import time
import numpy.random as npr
import numpy as np
import hashlib

@contextmanager
def tictoc(text=""):
    print "--- Start clock ---"
    t1 = time()
    yield
    dt = time() - t1
    print "--- Stop clock {0}: {1} seconds elapsed ---".format(text, dt)

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        # print "Cache hit?", self.func, str(args) in self.cache, str(args)[:80]
        return self.cache.setdefault(str(args), self.func(*args))

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def RandomState(obj):
    # Takes an arbitray object as seed (uses its string representation)
    hashed_int = int(hashlib.md5(str(obj)).hexdigest()[:8], base=16) # 32-bit int
    return npr.RandomState(hashed_int)

def dictslice(d, idxs):
    return {k : v[idxs] for k, v in d.iteritems()}
