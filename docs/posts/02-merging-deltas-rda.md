# Merging deltas: RDA against a weighted mean

*October 2026.*

When a round closes, the elected syncer holds one delta per accepted contribution: the fragment's anchor minus the learner's parameters for that fragment. It merges them into a single delta $\Delta$ and passes it to the outer Nesterov step. GradientMail has two merges. `gm.Rda()`, the default, uses radial-directional averaging (RDA) for most tensors and a weighted mean for embeddings; `gm.WeightedMean()` uses the weighted mean for every tensor.

Both merges use the same contribution weights. They return the same delta when the contributions point the same way, and different lengths and directions when they do not. This post covers how the weights are computed, the two formulas, how they behave on small inputs, and how to select one. The numbers come from calling `weight`, `weighted_mean` and `rda` from `gradientmail.algorithms.diloco` on fp32 tensors with the PyTorch backend.

## Weighting a contribution

A learner keeps a step counter and a sample counter for each fragment. `session.step(samples=n)` adds one step and `n` samples to every fragment's counters, and aligning with a committed version of a fragment resets that fragment's counters to zero. A contribution therefore reports the steps $k_i$ and samples $s_i$ trained since the learner last aligned with that fragment. Its weight is:

$$
w_i = \frac{s_i^2}{k_i} = s_i \cdot \frac{s_i}{k_i}
$$

The code attributes this rule to the Decoupled DiLoCo paper and squares the counts as exact integers before dividing. The second form reads the weight as the samples times the average samples per step. With a fixed batch of $b$ samples per step, $w_i = b^2 k_i$.

| Learner | Steps | Samples per step | Samples | Weight |
|---|---|---|---|---|
| A | 12 | 256 | 3,072 | 786,432 |
| B | 6 | 256 | 1,536 | 393,216 |
| C | 12 | 128 | 1,536 | 196,608 |

A and B use the same batch, so their weights are in the ratio of their samples, 2:1. When every learner uses the same batch size, as the four nodes of the [Pythia-14M run](00-pythia-14m-on-four-salad-gpus.md) did, the weights are proportional to samples. B and C processed the same 1,536 samples, but C took twice as many steps and weighs half as much as B, a quarter of A.

A learner may publish before its scheduled step once the other snapshots in a round are one short of quorum, so that the commit does not overwrite the steps it has trained. Such a contribution carries fewer steps and weighs less. A snapshot with zero steps is rejected as `zero_steps`, so the denominator is always positive. The round record lists each participant with the weight it received; rejected snapshots are listed by learner and reason and take no part in the merge. One set of weights applies to every tensor of the fragment.

## Two formulas

With $W = \sum_i w_i$, the weighted mean is:

$$
\Delta_{\text{mean}} = \frac{1}{W} \sum_i w_i \Delta_i
$$

RDA averages lengths and directions separately. With $\varphi(x) = x / \max(\lVert x \rVert, 10^{-12})$, it computes the weighted mean length $\bar r$ and the weighted mean $u$ of the unit directions, then gives the direction of $u$ the length $\bar r$:

$$
\bar r = \frac{1}{W} \sum_i w_i \lVert \Delta_i \rVert, \qquad u = \frac{1}{W} \sum_i w_i \, \varphi(\Delta_i), \qquad \Delta_{\text{RDA}} = \bar r \, \varphi(u)
$$

The floor of $10^{-12}$ keeps a zero vector at zero instead of dividing by zero. Norms are Euclidean over all elements of one tensor, so each tensor of a fragment gets its own length and direction.

By the triangle inequality, $\lVert \Delta_{\text{mean}} \rVert \le \bar r$, with equality when the deltas point the same way. $\Delta_{\text{RDA}}$ has length $\bar r$ whenever $u$ is nonzero. The figure shows the case from the guide: two deltas of length 1, 40 degrees either side of their mean direction, with equal weights. The weighted mean keeps their components along the mean direction, $\cos 40^\circ = 0.766$, and cancels the sideways ones. RDA gives the same direction length 1.

![Two deltas of length 1, each 40 degrees from their mean direction, merged two ways with equal weights. The weighted mean cancels their sideways parts and has length 0.77, shorter than either delta. RDA averages the lengths and the directions separately and gives the mean direction the mean length, 1.](../assets/merge.svg)

## Small inputs

| Deltas | Weights | Weighted mean | RDA |
|---|---|---|---|
| (1, 0), (3, 0) | 1, 1 | (2, 0) | (2, 0) |
| (1, 0), (0, 1) | 4, 1 | (0.8, 0.2), length 0.825 | (0.970, 0.243), length 1 |
| (3, 0), (0, 1) | 1, 1 | (1.5, 0.5), length 1.581 | (1.414, 1.414), length 2 |
| (1, 0), (−1, 0) | 2, 1 | (0.333, 0) | (1, 0) |
| (1, 0), (−1, 0) | 1.001, 1 | (0.0005, 0) | (1, 0) |
| (1, 0), (−1, 0) | 1, 1 | (0, 0) | (0, 0) |
| (0, 0), (1, 0) | 1, 1 | (0.5, 0) | (0.5, 0) |

Parallel deltas give the same result in both. The second row is the oracle in the backend tests, with weights 4 and 1 from contributions of 2 and 1 samples over one step each. Both results point the same way there, because the deltas have length 1 and are their own directions.

The third row shows the other difference. The weighted mean weights each direction by $w_i \lVert \Delta_i \rVert$, so the longer delta pulls the result to 18.4 degrees from its own direction. RDA weights directions by $w_i$ alone and lands at 45 degrees, halfway between the two.

With opposite deltas, the weighted mean shrinks as the weights approach equality. RDA keeps the length $\bar r = 1$ until the directions cancel exactly, and then returns zero. A zero delta, from a learner whose parameters did not change, has a zero direction under RDA: it lowers $\bar r$ by its weight share and leaves the direction as it was.

## Length and the outer step

The gap between the two lengths depends on how many contributions there are and how much they disagree. For $M$ deltas of length 1 in independent random directions over 100,000 elements, with equal weights, the weighted mean has length 0.705, 0.498 and 0.354 for $M$ = 2, 4 and 8, close to $1/\sqrt{M}$. RDA returns length 1 in each case. Deltas from training are not independent random vectors; how far their weighted mean falls below $\bar r$ depends on how much they agree.

The outer step scales whatever the merge returns. With the defaults `outer_lr=0.7` and `outer_momentum=0.9`, the first Nesterov step from zero momentum moves the fragment by $\eta (1 + \beta) \Delta = 1.33 \, \Delta$. The same outer settings therefore take a longer step under RDA than under the weighted mean whenever the deltas disagree. `gm.DiLoCo` uses the same defaults for both merges. The Pythia-14M run used the weighted mean, and this post does not compare training quality between the two.

## Embeddings

`gm.Rda` has one field, `direct_pattern`, with default `"embed"`. A tensor whose lowercased name contains the pattern is merged with the weighted mean, which the code describes as the paper's treatment of embeddings. The pattern itself is not lowercased, so a pattern with capital letters matches nothing.

In the backend test for this routing, a fragment holds `mlp.weight` and `tok_embed.weight`, both anchored at (1, 1), with the deltas and weights of the second row above, `outer_lr=1` and no momentum. The commit holds (0.2, 0.8) for `tok_embed.weight`, the weighted average of the two learners' parameters, and (0.030, 0.758) for `mlp.weight`.

The match is a substring test on parameter names. In `examples/pythia.py` the input and output embeddings are `gpt_neox.embed_in.weight` and `embed_out.weight`, and both match. In `examples/gpt2.py` they are `wte.weight` and `wpe.weight`, with the output head tied to `wte`; neither matches, so the default would merge them with RDA. Both examples set `merge=gm.WeightedMean()`; the two-machine examples, `torch_learner.py` and `mlx_learner.py`, use the default.

## Arithmetic

Deltas arrive in the algorithm's `delta_dtype`, by default each parameter's own dtype, and the backend loads them as fp32. Norms are returned as Python floats, and the weights are normalized in Python floats before they scale the tensors. The merged delta enters the outer step in fp32; for bf16 or fp16 parameters, the outer state keeps an fp32 master copy.

With one contribution, the weighted mean returns its delta exactly and RDA returns it up to fp32 rounding. RDA of identical deltas returns that delta whatever the weights. On one backend, a round produces the same bytes from the same inputs, and PyTorch and MLX agree within a relative tolerance of $10^{-6}$; the tests check both with the default merge.

## Choosing a merge

`merge` is a field of `gm.DiLoCo`:

```python
import gradientmail as gm

algorithm = gm.DiLoCo(sync_every=100, fragments=8, merge=gm.WeightedMean())
```

It defaults to `gm.Rda()`, and `gm.Rda(direct_pattern="tok_embeddings")` changes the pattern. Standard DiLoCo is `fragments=1, in_flight=1, overlap_steps=0, merge=gm.WeightedMean()` with `quorum_size` equal to the number of nodes. Local SGD is `outer_lr=1, outer_momentum=0, merge=gm.WeightedMean()`: each commit is then the weighted average of the learners' parameters, which [`examples/synthetic.py`](../../examples/synthetic.py) prints next to the expected value.

The merge is part of the algorithm configuration stored in the run's manifest, as `{"kind": "rda", "direct_pattern": "embed"}` or `{"kind": "weighted_mean"}`. A node that joins with a different merge gets `RunConfigMismatch`, so every syncer of a run merges the same way.
