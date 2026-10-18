# GradientMail

GradientMail implements the exchange of
[Decoupled DiLoCo](03-diloco-and-its-variants.md#decoupled-diloco) around
existing training loops. PyTorch or MLX handles the forward pass, autodiff, and
inner optimizer; GradientMail handles fragment exchange and shared updates.

Machines communicate through a **mailbox**, such as an S3-compatible bucket.
They need access to that store, but do not need direct connections to each other.
The store holds tensor payloads and coordination records; computations run on
the training machines.

Every node also runs the syncer runtime. One node is elected through the mailbox
to close rounds. If it fails, another can recover the committed state and take
over. This hosts the syncer role on a learner instead of requiring the dedicated
central synchronizer described in the paper.

## What happens in one round

1. A learner trains locally from a committed fragment, its **anchor**.
2. When a contribution is due, it publishes `anchor - local_parameters` for that
   fragment, together with the anchor version and its step and sample counts.
3. The elected syncer accepts contributions based on the current fragment
   version and rejects stale or incompatible ones. It waits for quorum, or for
   every live learner to report with at least one accepted contribution, then
   allows an adaptive grace window for additional reports.
4. It merges the accepted deltas, applies the outer optimizer, and publishes
   the new fragment and outer optimizer state. A create-once round record
   commits the result.
5. Learners fetch the result in the background and align with it between
   local training steps.

![Sequence of one round of a fragment, in five phases. Train: nodes A, B and C
train from the anchor, version 3. Publish: A and C put their delta, anchor
minus parameters, in the mailbox. Collect: the syncer elected for this round
runs on node C; it lists and gets the deltas and checks their versions,
reaches the quorum of two, and opens a grace window, inside which B's delta
arrives and is fetched. Merge and commit: the syncer merges the deltas with
RDA, applies the Nesterov outer step, puts version 4 with its outer state, and
commits the round by creating its record once. Align: each node polls, gets
version 4 and aligns with it between its steps, A first, then C, then B.](../assets/round.svg)

The default merge weights each contribution by `samples² / steps` since the
learner last aligned with that fragment. It uses **radial-directional averaging
(RDA)** for most tensors: average the delta lengths and directions separately,
then rescale the mean direction to the mean length. Parameters whose names
contain `embed` use a weighted mean instead.
The outer optimizer uses Nesterov momentum.

![Two deltas of length 1, each 40 degrees from their mean direction, merged two
ways with equal weights. The weighted mean cancels their sideways parts and
has length 0.77, shorter than either delta. RDA averages the lengths and the
directions separately and gives the mean direction the mean length, 1.](../assets/merge.svg)

Tensor payloads use Safetensors, with JSON records for metadata. Version checks
prevent applying a delta against the wrong anchor. Committed outer state
supports syncer recovery; inner optimizer state stays local and must be
checkpointed separately.

## Connecting a training loop

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

![Eight fragment rows over 200 local steps, with sync_every = 100 and fragments
= 8. DiLoCo exchanges every fragment together at steps 100 and 200, a single
column. Streaming DiLoCo exchanges one fragment at a time, at steps 12, 25,
37, 50, 62, 75, 87 and 100 and again 100 steps later, a staircase: each
fragment is still exchanged once every 100 steps, but each exchange carries an
eighth of the model.](../assets/fragments.svg)

Call `session.step` after each local optimizer update. Report the number of
examples for example-based training, or tokens for token-based training;
`len(batch)` is appropriate only when it measures that count. Learners in the
same run must use the same unit.

## Choosing the tradeoff

By default, training continues during exchange. Aligning with a committed
fragment replaces that fragment's local parameters, including updates made while
its contribution was in flight. `gm.DiLoCo(..., overlap_steps=0)` makes every
learner wait for the commit instead; a larger bound lets it train that many
steps past a publish first, and `mixing` blends what it trained meanwhile with
the commit instead of replacing it.

Closing the session waits for the learner's rounds in flight to commit and sets
the model to the newest committed parameters. Steps trained since a fragment's
last contribution are dropped, as in the DiLoCo papers. With
`gm.DiLoCo(..., flush_on_close=True)`, the learner publishes them instead, and
a round made only of these final contributions commits them without the outer
momentum.

![One learner's fragment over time. While training with the default, the learner
publishes its delta for a round and keeps training; when the commit arrives it
aligns with it, which replaces the steps trained while the round was in
flight. With overlap_steps = 0 it waits for the commit instead, and nothing
is replaced. At close, by default, it waits for its rounds in flight and the
model is set to the newest committed parameters, so the steps trained since
the last snapshot are dropped. With flush_on_close = True those steps go out
as a final snapshot, and the round made of final snapshots commits Θ − Δ
without the outer momentum.](../assets/in-flight.svg)

`quorum_size` defaults to `1`, allowing a round to proceed with one contribution.
A larger quorum seeks more contributions before merging, at the cost of more
waiting; live membership also affects when a round may close. A crashed learner
stops counting as live after its lease expires.

This design targets independent machines, uneven hardware, and networks where
per-step gradient exchange is expensive. DDP remains a suitable choice for
tightly connected accelerators. GradientMail still requires enough memory for
each learner's model and optimizer, enough bandwidth for fragment exchange,
and validation that the chosen synchronization settings train the model well.

---

Previous: [DiLoCo and its variants](03-diloco-and-its-variants.md)
