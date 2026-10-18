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

---

Previous: [Distributed training](02-distributed-training.md) · Next: [GradientMail](04-gradientmail.md)
