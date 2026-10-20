# Posts

- [Pre-training Pythia-14M on rented home GPUs](00-pythia-14m-on-four-salad-gpus.md)

  Training time, held-out loss and GPU cost for Pythia-14M on one to sixteen Salad GPUs communicating through a bucket, with the experimental protocol, published checkpoint comparisons and supporting measurements.

- [MLX and PyTorch learners in one run](01-mlx-and-pytorch-in-one-run.md)

  What crosses the mailbox, what has to match between the two models, and what the library checks.

- [Merging deltas: RDA against a weighted mean](02-merging-deltas-rda.md)

  How contributions are weighted, the two merges on small inputs, and how the choice changes the outer step.

- [Implementing a mailbox backend](03-implementing-a-mailbox-backend.md)

  The four methods and the clock, the guarantees a store must give, and a filesystem backend tested against the suite.
