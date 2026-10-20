# Shared Queue

Training on heterogeneous hardware means accounting for performance differences between nodes. Different GPUs process batches at different speeds, and even machines with the same GPU can slow down when they share resources with other workloads.

Imagine an asynchronous training run on four nodes, A, B, C and D. There are 400 blocks of data to process, so each node receives 100. A processes blocks twice as fast as B and C, and four times as fast as D.

When A finishes its 100 blocks, B and C have each processed 50 and D has processed 25. There are still 175 blocks left, but A has no work to do: those blocks were assigned to B, C and D. Later, B and C finish too, while D still has 50 blocks left. The run spends its final stretch with only its slowest node processing data.

Asynchronous parameter exchange allowed A, B and C to advance without waiting for D, but the fixed data split left them idle at the end. Giving A more blocks upfront would help if its measured speed stayed accurate throughout the run. It would not adapt if another workload started competing for resources halfway through.

A Shared Queue keeps the blocks available to all four nodes. Each node reserves a block, processes it and asks for another. When A finishes its first 100 blocks, it keeps taking pending work. If A later slows down, the other nodes take a larger share of the remaining blocks as they become available more often. No node needs to know the others' speeds.

![Four nodes process the same 24 units, four slices of six groups of dataset steps. A processes a unit in one time unit, B and C in two, D in four. With a fixed split each node owns one slice of every step: A finishes its six units at t = 6, B and C at t = 12, and D works alone until t = 24. With a shared queue each node claims the next unclaimed unit when it is ready: A processes 11 units, B and C 5 each and D 3, and all are done at t = 12.](../assets/take-a-number.svg)

## Distributing the data

To use this pattern in a training loop, the dataset needs to be divided into units that any node can process. Consider a run on four GPUs over 4,000 dataset steps. Each step covers 1,024 sequences, split into four slices of 256.

A fixed split would give each GPU the same slice at every step. To make those slices available to any node, the example groups one slice over 20 consecutive steps into a work unit:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    first: int
    count: int
    lane: int
    lanes: int
```

For example, `Unit(first=0, count=20, lane=2, lanes=4)` covers the third quarter of each batch at steps 0 through 19. It contains 20 local batches of 256 sequences each. A node can process any of the four slices; `lane` identifies the data, not the machine that must process it.

With units of 20 steps, the full run has 800 units. Every node builds the same mapping of identifiers to units, each identifier 1 to 64 lowercase letters, digits or hyphens:

```python
steps = 4000
unit_steps = 20
nodes = 4

units = {
    f"{first // unit_steps:06d}-{lane}": Unit(
        first=first,
        count=min(unit_steps, steps - first),
        lane=lane,
        lanes=nodes,
    )
    for first in range(0, steps, unit_steps)
    for lane in range(nodes)
}
```

![A grid of work units. Columns are groups of 20 dataset steps, 0–19 to 80–99 and onwards to step 4,000; rows are the four lanes, each a quarter of every step's 1,024 sequences. Each cell is one unit named by its group and lane, such as 000000-2. The highlighted unit, Unit(first=0, count=20, lane=2, lanes=4), is the third quarter of each batch at steps 0 to 19: 20 batches of 256 sequences. Any node can process any unit.](../assets/shared-queue-units.svg)

The identifier lets nodes agree on which unit they are reserving. The value tells the local reader which data to load. Each node has these values locally and uses the shared mailbox to record ownership. When two nodes try to reserve the same unit, atomic `put_if_absent` lets only one acquire it; the other looks for available work.

## Loading while the GPU trains

The loader turns a `Unit` into `(step, tensor)` pairs. Here, `sequences(start, count)` is a dataset reader supplied by the application: it fetches `count` consecutive sequences, starting at `start`.

```python
from collections.abc import Iterator

from torch import Tensor


BATCH = 1024


def load_unit(unit: Unit) -> Iterator[tuple[int, Tensor]]:
    share = BATCH // unit.lanes

    for step in range(unit.first, unit.first + unit.count):
        start = step * BATCH + unit.lane * share
        yield step, sequences(start, share)
```

The GPU also needs its next batch ready when it finishes the current one. A [`gm.SharedQueue`](../../src/gradientmail/queue.py) holds the work units, their loader and how many batches to buffer, so it does not need to know how the dataset is stored. `session.run` takes it as its data, reserves units and loads them on a background thread, and buffers the batches for the training step.

Each node constructs its `Session` with the run's mailbox and its own node ID; the queue reserves units through that mailbox, under that ID. `rank` identifies the node, `box` connects to the same S3 bucket and run prefix on every node and `algorithm` is the run's `gm.DiLoCo` configuration. With the model, optimizer and device initialized, each node runs the queue as follows. `learning_rate(step)` supplies the learning-rate schedule, and `train_step` returns the tokens it trained, 2,048 per sequence.

```python
import torch
from torch import Tensor

import gradientmail as gm


micro_batch = 8
share = BATCH // nodes
accumulation = share // micro_batch

session = gm.Session(model, algorithm=algorithm, mailbox=box, node_id=f"node-{rank}")


def train_step(item: tuple[int, Tensor]) -> int:
    step, batch = item
    batch = batch.to(device)

    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step)

    for chunk in batch.split(micro_batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(chunk[:, :-1], chunk[:, 1:]) / accumulation
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return share * 2048


session.run(train_step, gm.SharedQueue(units, load=load_unit, prefetch=8))
```

While the GPU processes a batch, the background thread reads ahead. `prefetch=8` limits the buffer to eight batches; when it fills, the thread waits before reading more data or seeking another unit. This overlaps reading with training when the reader can keep up.

After loading all batches from a unit, the thread looks for another unclaimed unit. It follows the mapping's order, skipping reservations already in the mailbox. Each node can put a different slice first within each group of steps to reduce competing claims; nodes can use different preferences while sharing the same identifiers.

![A background thread lists units, claims the next one with put_if_absent and loads its batches in step order, then goes back for another unit. It fills a buffer of eight batches; six are ready, steps 61 to 66. session.run takes the oldest batch and passes it to train_step, which sets the learning rate for its step, trains and returns the samples it trained. When the buffer is full the thread waits; when no unclaimed unit is left it ends, and run drains the buffer and leaves the loop.](../assets/shared-queue-prefetch.svg)

## When work runs out

Eventually, every unit has an owner. A node that finds no unclaimed units trains its buffered batches, and `run` returns once the node's rounds in flight have committed. Other nodes may still be processing their reservations, so completion is not simultaneous. Smaller units reduce the work tied to one node, but require more reservations. Prefetched batches also contribute to the work a node has already taken.

Ownership is recorded under `units/{identifier}.json` in the mailbox. For each reservation, the queue lists the existing claims, attempts a conditional write with the node's ID and reads the claim back. This requires atomic conditional creation; reading back cannot compensate for a non-atomic store.

![Two nodes claim units through the mailbox. Both list units/ to skip the claimed ones, then both try put_if_absent on units/000003-1.json at the same moment, each with its own node ID. The store creates it for node 1 and answers node 2 that it exists. Node 1 reads the object back, finds its own ID and loads the unit. Node 2 moves on to the next unit in its order, 000003-2, creates it, reads back its own ID and loads it.](../assets/shared-queue-claims.svg)

Reservations are permanent for that run. If a node fails or stops early, its unfinished units are not reassigned. Use a new mailbox prefix for a new run. The exception is an `either` node that the run elects its syncer: it gives back the units it reserved but has not started, and the other nodes claim them ([Roles](../guide/04-gradientmail.md#roles)).

When `run` ends, it stops prefetch and waits for the background thread to finish its current loader or mailbox operation. Transient mailbox failures are retried with backoff, as the protocol's own operations are; any other error from the loader or the mailbox is raised from `run` after the batches buffered before it have been trained.
