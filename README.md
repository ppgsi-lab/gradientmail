# GradientMail

Distributed training with [Decoupled DiLoCo](https://arxiv.org/abs/2604.21428) over S3-compatible storage. Supports PyTorch and MLX.

![Decoupled DiLoCo training](https://ppgsi-lab.github.io/gradientmail/docs/assets/hero.svg)

## Install

Python 3.12+.

```sh
pip install "gradientmail[s3,torch]"
```

Use `mlx` instead of `torch` for MLX on Apple Silicon.

## Usage

Use your model, optimizer and dataloader:

```python
import gradientmail as gm

session = gm.Session(
    model,
    algorithm=gm.DiLoCo(sync_every=100, fragments=8),
    mailbox=gm.S3Mailbox(bucket="my-training", prefix="experiments/run-1"),
)

def train_step(batch) -> int:
    loss = train(model, optimizer, batch)
    return len(batch)

session.run(train_step, dataloader)
```

Use the same bucket and prefix across machines. A node's `role` picks its part: `"both"`, the default, trains and closes rounds; `"learner"` only trains; `"syncer"` only closes rounds, for a machine that does not train, such as a laptop; `"either"` trains until the run elects it its syncer. With `ranks=N`, each node holds one of `N` fixed slices of the data. See the [PyTorch](https://github.com/ppgsi-lab/gradientmail/blob/main/examples/torch_learner.py) and [MLX](https://github.com/ppgsi-lab/gradientmail/blob/main/examples/mlx_learner.py) examples for complete setups.

[Documentation](https://ppgsi-lab.github.io/gradientmail/docs/guide/) · [Examples](examples/) · [Posts](https://ppgsi-lab.github.io/gradientmail/docs/posts/)

## Citation

```bibtex
@software{gradientmail,
  author = {dos Santos Silva, Gabriel Francisco},
  title = {GradientMail: Distributed training through asynchronous message exchange},
  year = {2026},
  url = {https://github.com/ppgsi-lab/gradientmail},
}
```

## License

[MIT](LICENSE).
