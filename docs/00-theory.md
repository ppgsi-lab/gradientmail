# From neural networks to GradientMail

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

## Distributed training

More machines can process more examples at once. They also need a way to combine
what they learn.

In **data parallelism**, each worker trains a replica of the model on different
data. In **model parallelism**, workers hold different parts of the model.
This guide follows data parallelism. A worker can itself be a group of devices
using model parallelism internally.

### Why is it difficult?

The difficulty is keeping useful training progress while coordinating workers:

- **Bandwidth:** moving a billion float32 values requires 4 GB of payload, before
  protocol overhead or the extra traffic of a collective operation.
- **Latency:** even a small message takes time to reach another machine.
- **Different speeds:** a fast worker may wait for a slower worker, called a
  straggler.
- **Failures:** a worker can disappear while others are waiting for it.
- **Divergence:** workers that train independently develop different parameters.
  Combining them changes the optimization process.

Reducing communication helps the system run efficiently, but requires deciding
how much the replicas may differ and how to reconcile them.

### Distributed Data Parallel

**Distributed Data Parallel**, or **DDP**, is PyTorch's implementation of
synchronous data parallel training. In the usual setup, every worker starts
with identical parameters, optimizer state, and optimizer configuration.

For each training step:

1. Each worker computes gradients on its own batch.
2. An **all-reduce** combines those gradients and gives every worker their average.
3. Each worker applies the same optimizer update, keeping the replicas aligned.

For independent per-example losses averaged over equal-sized batches, averaging
the gradients corresponds to the gradient over the combined batch. Two workers
processing 32 examples each therefore train on a global batch of 64 examples.

DDP overlaps gradient communication with the backward pass using groups called
buckets. The synchronized backward pass still needs all participating workers
to finish their collectives before the optimizer update.

This works well on fast, reliable networks. Across slow links or uneven
hardware, communication and waiting can dominate. A failed worker typically
requires recovery of the worker group.
See the [DDP design notes](https://docs.pytorch.org/docs/main/notes/ddp.html).

### DiLoCo

**DiLoCo**, short for **Distributed Low-Communication**, reduces how often
workers synchronize. Each worker runs an **inner optimizer** locally for $H$
steps before participating in a shared update.

Let $\Theta$ be the shared parameters at the start of a round, and $\theta_i$ be
worker $i$'s parameters after those local steps. Its contribution is:

$$
\Delta_i = \Theta - \theta_i
$$

This **outer gradient** is a parameter difference. It summarizes multiple local
updates; it is not the derivative of one batch's loss. The workers average these
differences, and an **outer optimizer** updates $\Theta$. They then load the
result and repeat. The original method uses AdamW inside and Nesterov momentum
outside.

For a scalar example, suppose $\Theta=10$, and two workers reach $8$ and $6$.
Their outer gradients are $2$ and $4$. Plain outer SGD with learning rate $1$
would produce $10-3=7$, the average of their parameters. Momentum changes this
update using previous rounds.

Exchanging updates every $H$ steps reduces communication frequency. The workers
still meet at synchronization rounds, and increasing $H$ allows more divergence.
See the [DiLoCo paper](https://arxiv.org/html/2311.08105v3).

#### Streaming DiLoCo

Synchronizing the entire model occasionally can still cause large communication
bursts. **Streaming DiLoCo** splits the parameters into **fragments** and
synchronizes them on a staggered schedule.

A fragment is a subset of parameters selected for communication. Each worker
still trains the full model; this is not model parallelism.

Workers can continue local training while a fragment is being exchanged. This
reduces peak bandwidth requirements and overlaps communication with computation.
The workers still participate in coordinated synchronization rounds, so a failed
worker can interrupt progress.
See the [Streaming DiLoCo paper](https://arxiv.org/html/2501.18512v1).

#### Decoupled DiLoCo

**Decoupled DiLoCo** retains local training, fragments, and outer optimization,
while removing the requirement to wait for every participant. Its **learners**
progress independently, each with its own model replica and inner optimizer.

A **syncer** maintains the shared fragments. It collects contributions from a
minimum number of learners, called a **quorum**, merges them, applies the outer
optimizer, and sends updated fragments back. A short adaptive waiting window
can include additional contributions after quorum is reached.

Learners report how many local steps and tokens they processed. This lets the
syncer weight contributions from learners that performed different amounts of
work. Learners install updated fragments as they receive them, so their replicas
can hold different versions at the same moment.

A slow or failed learner can be left out of a round. The tradeoff is that
learners train on increasingly different states, and installing a shared
fragment can overwrite local work done while communication was in progress.
This is a different optimization process from DDP; comparable quality must be
established experimentally.
See the [Decoupled DiLoCo paper](https://arxiv.org/html/2604.21428v1).

## GradientMail

GradientMail implements this exchange around existing training loops. PyTorch
or MLX handles the forward pass, autodiff, and inner optimizer; GradientMail
handles fragment exchange and shared updates using Decoupled DiLoCo.

Machines communicate through a **mailbox**, such as an S3-compatible bucket.
They need access to that store, but do not need direct connections to each other.
The store holds tensor payloads and coordination records; computations run on
the training machines.

Every node also runs the syncer runtime. One node is elected through the mailbox
to close rounds. If it fails, another can recover the committed state and take
over. This hosts the syncer role on a learner instead of requiring the dedicated
central synchronizer described in the paper.

### What happens in one round

1. A learner trains locally from an installed fragment, its **anchor**.
2. When a contribution is due, it publishes `anchor - local_parameters` for that
   fragment, together with the anchor version and its step and sample counts.
3. The elected syncer accepts contributions based on the current fragment
   version and rejects stale or incompatible ones. It waits for quorum, or for
   every live learner to report with at least one accepted contribution, then
   allows an adaptive grace window for additional reports.
4. It merges the accepted deltas, applies the outer optimizer, and publishes
   the new fragment and outer optimizer state. A create-once round record
   commits the result.
5. Learners fetch the result in the background and install it between local
   training steps.

The default merge weights each contribution by `samples² / steps` since that
fragment was installed. It uses **radial-directional averaging (RDA)** for most
tensors: average the delta lengths and directions separately, then rescale the
mean direction to the mean length. Parameters whose names contain `embed` use a
weighted mean instead.
The outer optimizer uses Nesterov momentum.

Tensor payloads use Safetensors, with JSON records for metadata. Version checks
prevent applying a delta against the wrong anchor. Committed outer state
supports syncer recovery; inner optimizer state stays local and must be
checkpointed separately.

### Connecting a training loop

The integration has this shape; `train_step` is the application's existing
forward pass, backward pass, and optimizer update:

```python
import gradientmail as gm

with gm.Session(
    model,
    algorithm=gm.DiLoCo(sync_every=100, fragments=8),
    mailbox=gm.S3Mailbox(bucket="my-training", prefix="runs/example"),
) as session:
    for batch in dataloader:
        loss = train_step(model, optimizer, batch)
        session.step(samples=len(batch))
```

Run a learner on each machine with the same mailbox location, algorithm
configuration, and compatible model definitions. Compatibility includes
parameter names, shapes, dtypes, and their meaning in the model. Sharing a file
format alone does not make different model implementations interchangeable.
The first learner initializes the run; later learners load its committed weights.

Here, `sync_every=100` sets the nominal interval for each fragment, with up to
eight fragments staggered across that interval. It does not mean that all eight
fragments are exchanged together every 100 steps. Actual contributions depend
on learner progress and round readiness.

Call `session.step` after each local optimizer update. Report the number of
examples for example-based training, or tokens for token-based training;
`len(batch)` is appropriate only when it measures that count. Learners in the
same run must use the same unit.

### Choosing the tradeoff

By default, training continues during exchange. Installing a committed fragment
replaces that fragment's local parameters, including updates made while its
contribution was in flight. `gm.DiLoCo(..., synchronous=True)` makes every
learner wait when its configured in-flight limit is reached.

`quorum_size` defaults to `1`, allowing a round to proceed with one contribution.
A larger quorum seeks more contributions before merging, at the cost of more
waiting; live membership also affects when a round may close. A crashed learner
stops counting as live after its lease expires.

This design targets independent machines, uneven hardware, and networks where
per-step gradient exchange is expensive. DDP remains a suitable choice for
tightly connected accelerators. GradientMail still requires enough memory for
each learner's model and optimizer, enough bandwidth for fragment exchange,
and validation that the chosen synchronization settings train the model well.
