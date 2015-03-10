# Gradient-based Optimization of Hyperparameters through Reversible Learning

<img src="https://raw.githubusercontent.com/HIPS/hypergrad/master/experiments/Jan_25_Figure_1/2/learning_curves.png" width="400">

Source code for http://arxiv.org/abs/1502.03492

### Abstract:

Tuning hyperparameters of learning algorithms is hard because gradients are usually unavailable. We compute exact gradients of cross-validation performance with respect to all hyperparameters by chaining derivatives backwards through the *entire training procedure*. These gradients allow us to optimize thousands of hyperparameters, including step-size and momentum schedules, weight initialization distributions, richly parameterized regularization schemes, and neural network architectures. We compute hyperparameter gradients by exactly reversing the dynamics of stochastic gradient descent with momentum.

Authors:
[Dougal Maclaurin](mailto:maclaurin@physics.harvard.edu),
[David Duvenaud](http://mlg.eng.cam.ac.uk/duvenaud/), and
[Ryan P. Adams](http://people.seas.harvard.edu/~rpa/)

Feel free to email us with any questions at (maclaurin@physics.harvard.edu), (dduvenaud@seas.harvard.edu).

For a look at some directions that didn't pan out, take a look at our early [research log](research-log.md).
