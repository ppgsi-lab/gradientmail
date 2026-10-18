# Barrier

Nodes in a GradientMail run do not wait for each other. A node joins when it is ready, publishes when a contribution is due and leaves when its data runs out, and rounds close with whoever reported. Some points in a run call for the opposite.

Consider four rented machines that boot at different times. The first node is up two minutes before the last. Nothing holds it: it trains for those two minutes, the first rounds carry its steps alone and the global batch is divided among fewer nodes than the run was sized for. At the other end, the nodes that consume a [Shared Queue](00-shared-queue.md) finish at different times, and an evaluation that should score the run's final parameters has to wait for the last of them.

A Barrier holds each node at a named point until a number of nodes have reached it, then releases all of them at one instant. The number is the caller's: the machines rented for the run, the algorithm's quorum or any other count. A node that reaches the point after the release passes at once.

![Three nodes, A, B and C, reach the barrier named start at different times, with parties = 3. Each records its arrival, holding the time it arrived, as barriers/start/ followed by its learner ID, and lists that prefix at poll instants every node shares. A arrives first and waits; B arrives next and waits; their polls see one key, then two. C arrives and lists three keys: the barrier opens. Each node reads the arrival times and computes the same release instant; all three wait for it and start training together.](../assets/barrier.svg)

## Lining up

Each node constructs a [`Barrier`](../../src/gradientmail/patterns/barrier.py) with the run's mailbox and its learner ID, then calls `wait` with a name and the number of nodes expected wherever the loop must line up. `rank` identifies the node; `box` connects to the same S3 bucket and run prefix on every node, and `nodes` is how many of them the run was started with. `train_step` is the application's forward pass, backward pass and optimizer update.

```python
from gradientmail import Session, align
from gradientmail.patterns.barrier import Barrier


barrier = Barrier(mailbox=box, learner_id=f"node-{rank}")

session = Session(model, algorithm=algorithm, mailbox=box, learner_id=f"node-{rank}")
barrier.wait("start", parties=nodes)

with session:
    for batch in dataloader:
        loss = train_step(model, optimizer, batch)
        session.step(samples=len(batch))

barrier.wait("done", parties=nodes, timeout=600)
align(model, mailbox=box)
evaluate(model)
```

Constructing the `Session` joins the run and loads the newest committed parameters, so the `start` barrier opens once every node holds them. Training then begins on every node at the same instant, between 2 and 7.6 s after the last arrival with the defaults, a `poll_interval` of 5 s and a `margin` of 1 s.

The `done` barrier comes after the session closes. `close` waits for the node's rounds in flight to commit, so once every node has arrived the run holds the last contribution of each of them, and `align` loads those parameters into the model before evaluation.

`wait` returns the learner IDs this node found arrived when it saw the barrier open, at least `parties` of them.

## Recording arrivals

An arrival is the object `barriers/{name}/{learner_id}` in the mailbox, created with `put_if_absent` and never removed. It holds the time the node arrived, on the mailbox's clock. `wait` creates this node's arrival, then lists the prefix once per `poll_interval` until it holds `parties` keys or more. The arrivals are the whole state: there is no leader, no object two nodes write and nothing to recover. A node that restarts finds its own arrival already there, with its first time, and passes at once if the release has passed; calling `wait` twice with the same name counts the node once. Transient mailbox failures are retried with backoff, as the protocol's own operations are.

Because arrivals persist, a name serves once per run. A loop names each barrier after its iteration, `f"epoch-{epoch}"`, and a new run uses a new mailbox prefix, as the Shared Queue does.

## Releasing together

A node learns that the barrier opened only when one of its own listings shows it, so nodes that left on learning would leave up to a `poll_interval` apart. The Barrier separates learning from leaving. The barrier opens at the time held by the `parties`-th earliest arrival. A node that finds it open reads the arrivals and computes the release: the first poll instant more than `margin` after the opening, plus `margin`. Every node reads the same arrival times, so every node computes the same release, sleeps until it and returns.

The poll instants are shared. Poll `k` falls in `[k·poll_interval, (k+1)·poll_interval)` on the mailbox's clock, so every node lists at the same moments. The first `margin` gives the last arrival's write time to reach every listing before that poll, so every node sees the opening at it; the second covers that listing and the reads of the arrivals. Neither depends on `poll_interval`: a longer interval delays the release and lists less often, but nodes still leave together. A node whose requests take longer than `margin`, during an outage or its retries, finds the release passed and returns as soon as it learns of the opening, and the others still leave together. Every node must use the same `poll_interval` and `margin`.

The instants are on `mailbox.clock()`, the store's time as each node estimates it. `S3Mailbox` estimates it from the `Date` header of every response, which has one-second resolution: a request sent at local time `s` whose response arrived at `r` dated `D` places the store's offset from the local clock between `D - r` and `D + 1 - s`. Intersecting responses narrows that window toward the round trip when they are dated at different fractions of a second, which is why poll `k`'s fraction of a second advances by the golden ratio instead of falling on whole seconds. A node whose wall clock already lies in the window keeps it, as one synchronized by NTP does; any other node takes the window's middle. `MemoryMailbox` shares one clock among the nodes of a process.

Against Hugging Face's gateway, three `S3Mailbox` instances on one machine, started with their clocks 2.5 s behind, on time and 3.5 s ahead, read the same instant within 90 ms after 5 requests each and within 11 to 23 ms after 20 to 120. A barrier with the defaults, which they reached 0, 2.3 and 4.1 s apart, released them within 24 ms of each other, 7.2 s after the last arrival; with a `poll_interval` of 1 s, within 23 ms, 2.4 s after it. The estimate is only as good as a node's traffic: its precision approaches the one-way trip to the store, while a node with a wall clock off by seconds can be up to half a second off after its first response. A barrier polling every 5 s adds 12 responses a minute; a `Session` on the same mailbox adds its own.

## When a node never arrives

`parties` is a count, so a node that crashed before reaching the barrier would hold the others indefinitely. `timeout` bounds the wait: after that many seconds `wait` raises `TimeoutError`, and the caller decides whether to go on with the nodes that arrived or to stop. The arrival stays recorded, so calling `wait` again resumes waiting instead of counting the node twice, and a node that arrives late finds the others' arrivals and passes even after they gave up.

A node waiting at a barrier keeps renewing its lease, so the run counts it as live, but it publishes nothing while it waits. With `quorum_size` 1, the default, rounds close without it. A round whose quorum needs its snapshot stays open until the barrier does; place such a barrier where every node has taken the same number of steps, as at the start, or after the session closes, as at the end.
