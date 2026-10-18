---
title: GradientMail
---

## Install

```sh
uv add "gradientmail[s3,torch]"
# or
pip install "gradientmail[s3,torch]"
```

Python 3.12+. Use `mlx` instead of `torch` for MLX on Apple Silicon.

## Documentation

- [Introduction](introduction.md)

  How it works, installing, adding a session to a training loop, and the examples.

- [Guide](docs/guide/README.md)

  A collection of explanations for understanding distributed training.

- [Patterns](docs/patterns/README.md)

  Ways to use GradientMail's mailbox for more than parameter exchange.

- [Posts](docs/posts/README.md)

  A Pythia-14M run on four rented GPUs, MLX and PyTorch learners in one run, merging deltas with RDA, and implementing a mailbox backend.
