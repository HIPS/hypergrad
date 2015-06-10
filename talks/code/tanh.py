
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad

def tanh(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

d_fun = grad(tanh)           # 1st derivative
dd_fun = grad(d_fun)         # 2nd derivative
ddd_fun = grad(dd_fun)       # 3rd derivative
dddd_fun = grad(ddd_fun)     # 4th derivative
ddddd_fun = grad(dddd_fun)   # 5th derivative
dddddd_fun = grad(ddddd_fun) # 6th derivative

x = np.linspace(-7, 7, 200)
plt.plot(x, map(tanh, x),
         x, map(d_fun, x),
         x, map(dd_fun, x),
         x, map(ddd_fun, x),
         x, map(dddd_fun, x),
         x, map(ddddd_fun, x),
         x, map(dddddd_fun, x))

