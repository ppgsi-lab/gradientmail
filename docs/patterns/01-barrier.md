# Barrier

Nodes in a GradientMail run do not wait for each other. A node joins when it is ready, publishes when a contribution is due and leaves when its data runs out, and rounds close with whoever reported. Some points in a run call for the opposite.

Consider four rented machines that boot at different times. The first node is up two minutes before the last. Nothing holds it: it trains for those two minutes, the first rounds carry its steps alone and the global batch is divided among fewer nodes than the run was sized for. At the other end, the nodes that consume a [Shared Queue](00-shared-queue.md) finish at different times, and an evaluation that should score the run's final parameters has to wait for the last of them.

A Barrier holds each node at a named point until a number of nodes have reached it. The number is the caller's: the machines rented for the run, the algorithm's quorum or any other count. A node that reaches the point after the barrier opened passes at once.

![Three nodes, A, B and C, reach the barrier named start at different times, with parties = 3. Each records its arrival as barriers/start/ followed by its learner ID and polls the listing of that prefix. A arrives first and waits; B arrives next and waits; their polls see one key, then two. C arrives, lists three keys and passes at once; A and B pass on their next poll. All three train from the same moment.](../assets/barrier.svg)

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

Constructing the `Session` joins the run and loads the newest committed parameters, so the `start` barrier opens once every node holds them. Training then begins on every node within one `poll_interval` of the last arrival.

The `done` barrier comes after the session closes. `close` waits for the node's rounds in flight to commit, so once every node has arrived the run holds the last contribution of each of them, and `align` loads those parameters into the model before evaluation.

`wait` returns the learner IDs that had arrived when the barrier opened, at least `parties` of them.

## Recording arrivals

An arrival is the object `barriers/{name}/{learner_id}` in the mailbox, created with `put_if_absent` and never removed. `wait` creates this node's arrival, then lists the prefix every `poll_interval` seconds until it holds `parties` keys or more. The listing is the whole state: there is no leader and nothing to recover. A node that restarts finds its own arrival already there and passes if the barrier opened meanwhile, and calling `wait` twice with the same name counts the node once. Transient mailbox failures are retried with backoff, as the protocol's own operations are.

Because arrivals persist, a name serves once per run. A loop names each barrier after its iteration, `f"epoch-{epoch}"`, and a new run uses a new mailbox prefix, as the Shared Queue does.

## When a node never arrives

`parties` is a count, so a node that crashed before reaching the barrier would hold the others indefinitely. `timeout` bounds the wait: after that many seconds `wait` raises `TimeoutError`, and the caller decides whether to go on with the nodes that arrived or to stop. The arrival stays recorded, so calling `wait` again resumes waiting instead of counting the node twice, and a node that arrives late finds the others' arrivals and passes even after they gave up.

A node waiting at a barrier keeps renewing its lease, so the run counts it as live, but it publishes nothing while it waits. With `quorum_size` 1, the default, rounds close without it. A round whose quorum needs its snapshot stays open until the barrier does; place such a barrier where every node has taken the same number of steps, as at the start, or after the session closes, as at the end.
