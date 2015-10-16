# Gradient-based Optimization of Hyperparameters through Reversible Learning

<img src="https://raw.githubusercontent.com/HIPS/hypergrad/master/experiments/Jan_25_Figure_1/2/learning_curves.png" width="400">

Source code for http://arxiv.org/abs/1502.03492

### Abstract:

Tuning hyperparameters of learning algorithms is hard because gradients are usually unavailable. We compute exact gradients of cross-validation performance with respect to all hyperparameters by chaining derivatives backwards through the *entire training procedure*. These gradients allow us to optimize thousands of hyperparameters, including step-size and momentum schedules, weight initialization distributions, richly parameterized regularization schemes, and neural network architectures. We compute hyperparameter gradients by exactly reversing the dynamics of stochastic gradient descent with momentum.

Authors:
[Dougal Maclaurin](mailto:maclaurin@physics.harvard.edu),
[David Duvenaud](http://mlg.eng.cam.ac.uk/duvenaud/), and
[Ryan P. Adams](http://people.seas.harvard.edu/~rpa/)

### How to run these experiments

You'll need to install [autograd](https://github.com/HIPS/autograd), our automatic differentiation package.
However, autograd (aka funkyYak) has changed a lot since we wrote the hypergrad code, and it would take a little bit of work to make them compatible again.

However, the hypergrad code should work with the version of FunkyYak as of Feb 2, at this revision:
https://github.com/HIPS/autograd/tree/be470d5b8d6c84bfa74074b238d43755f6f2c55c

So if you clone autograd, then type
git checkout be470d5b8d6c84bfa74074b238d43755f6f2c55c,
you should be at the same version we used to run the experiments.

That version also predates the setup.py file, so to get your code to use the old version, you'll either have to copy setup.py into the old revision and reinstall, or add FunkyYak to your PYTHONPATH.


Feel free to email us with any questions at (maclaurin@physics.harvard.edu), (dduvenaud@seas.harvard.edu).

For a look at some directions that didn't pan out, take a look at our early [research log](research-log.md).
