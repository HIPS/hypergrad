import numpy.random as npr
import numpy as np
from hypergrad.exact_rep import BitStore
npr.seed(1)

def test_pop_empty():
    store = BitStore(4)
    popempty = store.pop(3)
    assert all(popempty == [0, 0, 0, 0])
    store.push(popempty, 3)
    store.push([1, 4, 0, 5], 6)
    assert all(store.pop(6) == [1, 4, 0, 5])

def test_long_sequence():
    N_iters = 200
    vect_length = 10
    Ns = []
    Ms = []
    for i in range(N_iters):
        Ms.append(npr.randint(200) + 1)
        Ns.append(npr.randint(Ms[-1], size=vect_length))
    store = BitStore(vect_length)
    coinflips = npr.rand(N_iters)
    new_Ns = []
    for N, M, r in zip(Ns, Ms, coinflips):
        if r < 0.75:
            store.push(N, M)
        else:
            new_Ns.append(store.pop(M))

    for N, M, r in zip(Ns, Ms, coinflips)[::-1]:
        if r < 0.75:
            cur_N = store.pop(M)
            assert np.all(cur_N == N)
        else:
            store.push(new_Ns.pop(), M)

