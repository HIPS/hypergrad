
from contextlib import contextmanager
from time import time

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
        return self.cache.setdefault(str(args), self.func(*args))

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)
