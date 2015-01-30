import numpy as np

RADIX_SCALE = 2**52

class ExactRep(object):
    """Fixed-point representation of arrays with auxilliary bits such
    that + - * / ops are all exactly invertible (except for
    overflow)."""
    def __init__(self, val, from_intrep=False):
        if from_intrep:
            self.intrep = val
        else:
            self.intrep = self.float_to_intrep(val)

        self.aux = BitStore(len(val))

    def add(self, A):
        """Reversible addition of vector or scalar A."""
        self.intrep += self.float_to_intrep(A)
        return self

    def sub(self, A):
        self.add(-A)
        return self

    def rational_mul(self, n, d):
        self.aux.push(self.intrep % d, d) # Store remainder bits externally
        self.intrep /= d                  # Divide by denominator
        self.intrep *= n                  # Multiply by numerator
        self.intrep += self.aux.pop(n)    # Pack bits into the remainder

    def mul(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        return self
        
    def div(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        return self

    def float_to_rational(self, a):
        assert np.all(a > 0.0)
        d = 2**16 / np.fix(a+1).astype(int) # Uglier than it used to be: np.int(a + 1)
        n = np.fix(a * d + 1).astype(int)
        return  n, d

    def float_to_intrep(self, x):
        return (x * RADIX_SCALE).astype(np.int64)

    @property
    def val(self):
        return self.intrep.astype(np.float64) / RADIX_SCALE

class BitStore(object):
    """Efficiently stores information with non-integer number of bits (up to 16)."""
    def __init__(self, length):
        # Use an array of Python 'long' ints which conveniently grow
        # as large as necessary. It's about 50X slower though...
        self.store = np.array([0L] * length, dtype=object)

    def push(self, N, M):
        """Stores integer N, given that 0 <= N < M"""
        assert np.all(M <= 2**16)
        self.store *= M
        self.store += N

    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % M
        self.store /= M
        return N

class LongIntArray(object):
    """Behaves like np.array([0L] * length, dtype=object) but faster."""
    def __init__(self, length):
        self.val = []
        self.nbits = 0
        self.grow()

    def grow(self):
        self.val.append(np.zeros(length, dtype=np.int32))
        self.nbits += 32

    def __mod__(self, other):
        pass

    def __iadd__(self, other):
        self.val[-1] += other
        
    def __imul__(self, other):
        pass

    def __idiv__(self, other):
        pass

