# DiLoCo and its variants

## DiLoCo

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

![Two number lines for the guide's example: the shared parameter Θ is 10 and two
workers end at 8 and 6, so their outer gradients are 2 and 4, with mean 3.
With plain outer SGD at learning rate 1, Θ moves by 3 to 7, the average of the
workers. With Nesterov momentum at learning rate 0.7 and momentum 0.9, and a
momentum of 3 left by an earlier round, the momentum becomes 5.7 and Θ moves
by 5.69 to 4.31, past both workers.](../assets/outer-step.svg)

Exchanging updates every $H$ steps reduces communication frequency. The workers
still meet at synchronization rounds, and increasing $H$ allows more divergence.
See the [DiLoCo paper](https://arxiv.org/html/2311.08105v3).

### In GradientMail

The paper's experiments run 8 workers, each on its own shard of the data, with
$H = 500$ (50 to 2,000 in the ablation), outer learning rate 0.7 and momentum
0.9. Every worker takes part in every outer step and resumes from the same
$\Theta$.

```python
gm.DiLoCo(
    sync_every=500,          # H
    fragments=1,             # the whole model in one round
    in_flight=1,
    overlap_steps=0,         # wait for the outer step before training on
    quorum_size=8,           # every worker
    merge=gm.WeightedMean(),
    outer_lr=0.7,
    outer_momentum=0.9,
)
```

With one fragment, a round every $H$ steps carries the whole model.
`overlap_steps=0` makes `step()` wait, once a round is published, until its
commit has been applied, so every worker resumes from the same parameters.
`quorum_size` equal to the number of workers makes a round wait for all of
them. The weighted mean weighs each contribution by its tokens squared over its
steps, which with equal batches and $H$ steps each is the paper's plain
average.

Three things are not in the paper. A worker that crashes is waited for until
its lease expires and is then left out, where DiLoCo would stop. A worker
publishes ahead of its schedule once the others' snapshots in a round are one
short of the quorum (`follow`, below), so the last worker of a round may
contribute a step or so fewer than $H$; nothing turns that off. And
GradientMail does not shard the data: which batches a node trains on is the
training loop's business, for instance through the
[Shared Queue](../patterns/00-shared-queue.md) pattern.

## Streaming DiLoCo

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

### In GradientMail

The paper keeps three transformer blocks per fragment, dealt in a strided
pattern (fragment $p$ holds every $P$-th block from the $p$-th), synchronizes
each fragment every $H = 100$ steps at its own offset, overlaps the exchange
with $\tau = 1$ inner step (up to 5 is advised), then sets the fragment to
$\alpha\,\theta + (1 - \alpha)\,\tilde\theta$, the local parameters blended
with the outer step's result, with $\alpha = 0.5$. Outer gradients travel in
4 bits. Every replica takes part in every fragment's exchange; the outer
optimizer is Nesterov with learning rate 0.4.

```python
gm.DiLoCo(
    sync_every=100,          # H, per fragment
    fragments=8,             # P: 24 blocks, three per fragment
    partition=gm.Strided(),
    in_flight=1,
    overlap_steps=1,         # τ: train on for a step, then wait for the commit
    alignment=gm.Blend(0.5), # α
    delta_dtype="F4",        # 4-bit outer gradients
    quorum_size=2,           # every replica
    merge=gm.WeightedMean(),
    outer_lr=0.4,
    outer_momentum=0.9,
)
```

Round $r$ targets fragment $r \bmod P$ at step
$\lfloor r / P \rfloor \cdot H + \lfloor ((r \bmod P) + 1) \cdot H / P \rfloor$:
each fragment once every $H$ steps, at offsets $H / P$ apart, the paper's
$t_p$. `overlap_steps` bounds the overlap rather than fixing it: a learner
trains at most $\tau$ steps past a publish and then waits for the commit, which
it applies as soon as it lands. `gm.Blend(0.5)` is $\alpha$. `delta_dtype="F4"`
sends each delta at four bits per element in the MX layout, with a scale per
block of 32; the paper's format is E3M0, four bits with no block scale. The
paper does not state the outer momentum; 0.9 is DiLoCo's.

## Decoupled DiLoCo

**Decoupled DiLoCo** retains local training, fragments, and outer optimization,
while removing the requirement to wait for every participant. Its **learners**
progress independently, each with its own model replica and inner optimizer.

A **syncer** maintains the shared fragments. It collects contributions from a
minimum number of learners, called a **quorum**, merges them, applies the outer
optimizer, and sends updated fragments back. A short adaptive waiting window
can include additional contributions after quorum is reached.

Learners report how many local steps and tokens they processed. This lets the
syncer weight contributions from learners that performed different amounts of
work. Learners align with updated fragments as they receive them, so their
replicas can hold different versions at the same moment.

A slow or failed learner can be left out of a round. The tradeoff is that
learners train on increasingly different states, and aligning with a shared
fragment can overwrite local work done while communication was in progress.
This is a different optimization process from DDP; comparable quality must be
established experimentally.
See the [Decoupled DiLoCo paper](https://arxiv.org/html/2604.21428v1).

![Four timelines of three workers, A, B and C, where a step of C takes 1.6 times
as long as one of A. DDP: after every step the gradients are all-reduced, and
A and B wait for C each time. DiLoCo: four local steps, then the whole model
is averaged; A and B wait for C, and everyone waits during the exchange.
Streaming DiLoCo: a fragment is exchanged every two steps while training
continues, but the workers stay in lockstep, so A and B still wait for C.
Decoupled DiLoCo: no worker waits; each publishes fragments as it goes, and
the syncer closes a round as soon as two of the three have contributed,
sometimes all three.](../assets/sync-patterns.svg)

### In GradientMail

The defaults are this paper's. Its main experiments run 8 learners with
$P = H = 24$, a fragment every step, two fragments in flight ($\tau = 2$),
quorum $K = 1$ with a grace window of up to a step, contributions weighted by
$c^{tokens} \cdot c^{tokens} / c^{steps}$, RDA as the merge for every tensor
but the embeddings, which use the weighted mean, and a learner that overwrites
its fragment with the syncer's ($\alpha = 0$, which the paper found best beyond
two learners). The paper states no outer hyperparameters.

```python
gm.DiLoCo(sync_every=24, fragments=24)
```

which is

```python
gm.DiLoCo(
    sync_every=24,
    fragments=24,
    in_flight=2,               # τ
    overlap_steps=None,        # never wait
    alignment=gm.Overwrite(),  # α = 0
    quorum_size=1,             # K
    grace_seconds=None,        # the window estimated from the round timings
    merge=gm.Rda(),            # weighted mean for names containing "embed"
    outer_lr=0.7,
    outer_momentum=0.9,
)
```

The grace window comes from exponential moving averages of the step time the
learners report, the time a round takes to reach quorum and the time a commit
takes: half the slack those leave within $\tau$ rounds, never more than one
round, and zero while any estimate is missing. The outer learning rate and
momentum are DiLoCo's.

Two things differ from the paper's system. There is no syncer process: every
node closes rounds, elected by a claim per round, and liveness is a lease each
node renews, where the paper runs one syncer shard per learner and reads
metadata the learners send every step. And learners push one snapshot per
round into the mailbox, where the paper's syncer pulls the learners'
parameters.

## Local SGD

Averaging the learners' parameters at every round, with no outer momentum, is
Local SGD (FedAvg):

```python
gm.DiLoCo(sync_every=H, fragments=P, outer_lr=1.0, outer_momentum=0.0, merge=gm.WeightedMean())
```

Each commit is then the token-weighted average of the fragments the learners
published, which [`examples/synthetic.py`](../../examples/synthetic.py) prints
next to the expected value.

## The parameters by mechanism

| Mechanism | Parameter | DiLoCo | Streaming | Decoupled |
|---|---|---|---|---|
| Inner steps per exchange of a fragment | `sync_every` | $H$ | $H$ | $H$ |
| Fragments | `fragments`, `partition` | 1 | $P$, strided | $P$, by bytes |
| Rounds a learner publishes ahead of the last commit it applied | `in_flight` | 1 | 1 | $\tau$ |
| Steps trained past a publish before waiting | `overlap_steps` | 0 | $\tau$ | `None` |
| What the commit does to the steps since the snapshot | `alignment` | `Overwrite()` | `Blend(α)` | `Overwrite()` |
| Snapshots that close a round | `quorum_size` | all | all | $K$ |
| Collecting after quorum | `grace_seconds` | — | — | estimated |
| Merge of the deltas | `merge` | `WeightedMean()` | `WeightedMean()` | `Rda()` |
| Outer optimizer | `outer_lr`, `outer_momentum` | 0.7, 0.9 | 0.4, — | — |
| Precision of the deltas | `delta_dtype` | model's | 4-bit | model's (the paper's int4 is not implemented) |

`gm.Carry()`, which sets the fragment to the commit plus every step trained
since the snapshot, is the delay-corrected merge of
[LOSCAR-SGD](https://arxiv.org/abs/2605.20866), which proves that the average
of the local models then evolves as if nothing had overlapped and finds it
ahead of overwriting by the most when communication is slow against the local
window; there it corrects a plain average, here the outer step's result.

What is in none of the papers is GradientMail's own: `gm.Adaptive()`, which
carries while the mean cosine between a round's deltas is high and overwrites
once it is low, fragment by fragment; `follow`, how many
others' snapshots in a round make a learner publish ahead of its schedule;
`grace_seconds` as a fixed window; `flush_on_close`; `commit_dtype`; the
formats of the outer state, `master_dtype`, `momentum_dtype` and
`residual_dtype`; `global_batch_size`; and the mailbox itself in place of all-reduce or a
syncer. The next chapter covers how to choose among them.

---

Previous: [Distributed training](02-distributed-training.md) · Next: [GradientMail](04-gradientmail.md)
