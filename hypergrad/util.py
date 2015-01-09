
from contextlib import contextmanager
from time import time
@contextmanager
def tictoc(text=""):
    print "--- Start clock ---"
    t1 = time()
    yield
    dt = time() - t1
    print "--- Stop clock {0}: {1} seconds elapsed ---".format(text, dt)
