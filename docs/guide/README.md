# Guide

A collection of explanations for understanding distributed training.

1. [Neural networks and training](01-neural-networks-and-training.md)

   Parameters, loss, gradients, autodiff and the training loop.

2. [Distributed training](02-distributed-training.md)

   Data and model parallelism, what makes it hard, and Distributed Data Parallel.

3. [DiLoCo and its variants](03-diloco-and-its-variants.md)

   Local steps with an outer optimizer, fragments, rounds that do not wait for every worker, and the configuration of each variant.

4. [GradientMail](04-gradientmail.md)

   One round through the mailbox, connecting a training loop, and choosing the tradeoff.
