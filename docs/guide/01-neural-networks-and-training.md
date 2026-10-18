# Neural networks and training

This guide assumes you can read code, but have not trained a model before.
It builds from a single prediction to training across independent machines.

## Neural networks

A model takes an input and produces a prediction. Its **parameters** are numbers
that control that computation. Training adjusts those numbers using examples.

A neural network organizes the computation into layers. A typical layer computes
`activation(Wx + b)`: `x` is its input, `W` contains weights, and `b` contains
biases. The activation is a nonlinear function, such as `max(0, x)`, which lets
stacked layers represent nonlinear relationships.

Weights and biases are stored in **tensors**: arrays with one or more dimensions.
The symbol $\theta$ will mean all the model's trainable parameters together.

For the next few sections, use a small linear model:

$$
\hat y = wx + b
$$

Here $x$ is the input, $\hat y$ is the prediction, and $w$ and $b$ are the
parameters. The same training principles apply to networks with billions of
parameters.

### Measuring prediction error

A training example supplies an input $x$ and a target $y$. A **loss function**
turns the prediction and target into a number to minimize.

For this example, use squared error:

$$
L = (\hat y - y)^2
$$

With $x=2$, $y=6$, $w=1$, and $b=0$, the prediction is $2$ and the loss is $16$.
The loss measures the error, but choosing an update also requires knowing how
each parameter affects it.

## Training

Training uses gradients to choose parameter updates that reduce the loss.
An optimizer applies those updates, and the process repeats over batches of data.

### Gradients

A **derivative** measures how a small change in one number changes another.
For example, $\partial L / \partial w$ measures how the loss changes when $w$
changes while the other parameters stay fixed.

The **gradient** collects these derivatives, one for every parameter:

$$
\nabla_\theta L =
\left(\frac{\partial L}{\partial w},\frac{\partial L}{\partial b}\right)
$$

In our example, the gradient is $(-16,-8)$. The negative signs mean that small
increases in either parameter reduce the loss at this point.

**Gradient descent** updates parameters in the opposite direction of the gradient:

$$
\theta_{\text{new}} = \theta - \eta\nabla_\theta L
$$

The **learning rate** $\eta$ controls the size of the update. With $\eta=0.1$,
$w$ becomes $2.6$ and $b$ becomes $0.8$. The new prediction is $6$, giving zero
loss on this example. Larger models rarely fit a batch in one step; a learning
rate that is too large can even increase the loss.

![Contour lines of the loss L = (2w + b − 6)² over the parameters w and b. The
lines are parallel, and the loss is zero along the line 2w + b = 6. From the
start (1, 0), where L = 16, the step −η times the gradient crosses the lines
at a right angle. With η = 0.1 it lands at (2.6, 0.8) on the zero-loss line;
with η = 0.25 it continues past the valley to (5, 2), where L = 36.](../assets/gradient-step.svg)

### Automatic differentiation

Writing a derivative by hand for every parameter would be impractical.
**Automatic differentiation**, or **autodiff**, computes derivatives by applying
known derivative rules to the operations in a computation.

The **forward pass** computes predictions and the loss. During training, the
framework also tracks the operations and retains intermediate values needed for
differentiation.

The **backward pass** propagates derivatives from the loss back toward the
parameters. It uses the chain rule: derivatives multiply along a path through
the computation, and contributions add where paths meet. This reverse traversal
is **backpropagation**, an application of reverse-mode autodiff.

For our model:

$$
\frac{\partial L}{\partial w}
= \frac{\partial L}{\partial\hat y}
  \frac{\partial\hat y}{\partial w}
= 2(\hat y-y)x
= 2(2-6)\cdot2
= -16
$$

![Computation graph of the guide's example. Data x = 2 and parameter w = 1 enter
a multiplication giving wx = 2; adding parameter b = 0 gives the prediction ŷ
= 2; subtracting the target y = 6 gives −4; squaring gives the loss L = 16.
The backward pass runs right to left: the derivative of L is 1 at the loss,
2(ŷ − y) = −8 after the square, −8 through the subtraction and the addition,
so ∂L/∂b = −8, and multiplied by x = 2 at the multiplication, so ∂L/∂w = −16.
The gradient is (−16, −8).](../assets/autodiff.svg)

Autodiff produces gradients. An optimizer uses them to change the parameters.
PyTorch exposes this separation through `loss.backward()` and `optimizer.step()`.
See its [autodiff guide](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).

### The training loop

The forward pass, backward pass, and optimizer update together form a training
step. Training starts from initialized weights or a previously trained
checkpoint and proceeds over **batches**: groups of examples processed together.
A batch's loss is often the average of the examples' losses, so its gradient
reflects that whole batch.

A typical training step is:

1. Select a batch and clear gradients left from the previous step.
2. Run the forward pass and calculate the loss.
3. Run the backward pass to compute gradients.
4. Apply the optimizer update.

SGD uses the gradient descent update above. Other optimizers keep additional
state: momentum tracks past gradients, while AdamW uses running gradient
statistics and weight decay. **Optimizer state** is separate from model weights.

Training repeats these steps over many batches. Evaluation on held-out data
checks whether the model works on examples it did not train on. **Inference**
uses the trained model to make predictions, usually without computing gradients.

---

Next: [Distributed training](02-distributed-training.md)
