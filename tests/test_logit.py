import numpy as np

from hypergrad.nn_utils import logit, inv_logit

def test_logit():
    assert np.allclose(logit(0), 0.5, rtol=1e-3, atol=1e-4)
    assert np.allclose(logit(-100), 0, rtol=1e-3, atol=1e-4)
    assert np.allclose(logit( 100), 1, rtol=1e-3, atol=1e-4)

def test_inv_logit():
    assert np.allclose(inv_logit(logit(0.5)), 0.5, rtol=1e-3, atol=1e-4)
    assert np.allclose(inv_logit(logit(0.6)), 0.6, rtol=1e-3, atol=1e-4)
    assert np.allclose(inv_logit(logit(0.1)), 0.1, rtol=1e-3, atol=1e-4)
    assert np.allclose(inv_logit(logit(0.2)), 0.2, rtol=1e-3, atol=1e-4)

