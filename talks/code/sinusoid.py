
import numpy as np
import matplotlib.pyplot as plt
from funkyyak import grad

def fun(x):
    return np.sin(x)

d_fun = grad(fun)    # First derivative
dd_fun = grad(d_fun) # Second derivative

x = np.linspace(-10, 10, 100)
plt.plot(x, map(fun, x),
         x, map(d_fun, x),
         x, map(dd_fun, x))
