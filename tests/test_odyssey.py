import time
import numpy.random as npr
from hypergrad.odyssey import omap

objects = [1, (3,4), 1.04, "hello", "it's \n a hard string \\\''\to parse", ((1,'a'), 3)]

def identity(x):
    time.sleep(npr.randint(5))
    print x
    return x

def check_omap():
    # This won't work with nosetest. Needs to be run from the same directory as the file.
    ans = omap(identity, objects)
    for x, y in zip(ans, objects):
        assert x == y, "Failed on {0}".format(y)
    print "test ok"

if __name__ == "__main__":
    check_omap()
