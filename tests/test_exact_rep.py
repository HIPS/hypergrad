import numpy as np
import numpy.random as npr

from hypergrad.exact_rep import ExactRep
npr.seed(1)

def test_add_sub():
    A = npr.randn(100)
    B = npr.randn(100) * 500
    assert np.mean((A + B) - B == A) < 0.5
    assert np.mean((A - B) + B == A) < 0.5
    exact_A = ExactRep(A)
    orig_value = exact_A.val
    exact_A.add(B)
    assert np.allclose(exact_A.val, A + B)
    exact_A.sub(B)
    assert all(exact_A.val == orig_value)
    exact_A.sub(B)
    assert np.allclose(exact_A.val, A - B)
    exact_A.add(B)
    assert all(exact_A.val == orig_value)

def test_mul_div():
    A = npr.randn(100)
    all_b = [0.95, 0.9, 0.5, 0.3, 1.01]
    for b in all_b:
        A_new = (((A * b + A) - A) / b)
        assert not all(A_new == A)
        exact_A = ExactRep(A)
        orig_value = exact_A.val
        exact_A.mul(b)
        assert np.allclose(exact_A.val, A * b, rtol=1e-3, atol=1e-4)
        exact_A.div(b)
        assert all(exact_A.val == orig_value)

def test_repeated_mul_div():
    A = npr.randn(100)
    exact_A = ExactRep(A)
    orig_value = exact_A.val
    all_b = npr.rand(200)
    A_cur_float = A
    for b in all_b:
        A_cur_float = A_cur_float * b
        exact_A.mul(b)
    assert np.allclose(exact_A.val, A_cur_float)
    for b in all_b[::-1]:
        A_cur_float = A_cur_float / b
        exact_A.div(b)
    assert np.mean(A_cur_float == A) < 0.2
    assert all(exact_A.val == orig_value)

# TODO: Check that bit representation is what I expect
# TODO: Check that storage grows as expected after multiple multiplication cycles
