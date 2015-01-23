import numpy as np
from hypergrad.nn_utils import VectorParser

def test_vector_parser():
    P = VectorParser()
    A = np.random.randn(4,5)
    B_old = np.random.randn(1,5)
    B = np.random.randn(1,5)
    C = np.random.randn(4)
    D = np.random.randn(1,2,4)

    P["A"] = A
    P["B"] = B_old
    P["C"] = C
    P["B"] = B
    P["D"] = D

    assert np.all(P["A"] == A)
    assert np.all(P["B"] == B)
    assert np.all(P["C"] == C)
    assert np.all(P["D"] == D)
