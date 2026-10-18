# Distributed training

More machines can process more examples at once. They also need a way to combine
what they learn.

In **data parallelism**, each worker trains a replica of the model on different
data. In **model parallelism**, workers hold different parts of the model.
This guide follows data parallelism. A worker can itself be a group of devices
using model parallelism internally.

![Two panels. Data parallelism: three workers each hold all four layers of the
model and train on a different batch; what they learn is then combined. Model
parallelism: one batch enters worker 1, which holds layers 1 and 2;
activations go down to worker 2, which holds layers 3 and 4, and gradients
come back up.](../assets/parallelism.svg)

## Why is it difficult?

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

## Distributed Data Parallel

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

---

Previous: [Neural networks and training](01-neural-networks-and-training.md) · Next: [DiLoCo and its variants](03-diloco-and-its-variants.md)
