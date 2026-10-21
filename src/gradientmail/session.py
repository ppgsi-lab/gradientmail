"""The public facade: one object runs a node's part in a run, from joining it to its commits.

``Session`` resolves the framework backend from the model and names the node after the host and
process when the caller does not; constructing one does no I/O. ``join`` creates or adopts the
run and returns once the model holds the newest global parameters, and from then on the
learner's communication loop and the syncer run on two daemon threads. ``run`` is the training
loop: it pulls batches from the data, hands each to the caller's ``step`` and accounts the
samples that returns; once the data is exhausted it marks the node done (``done/{node}.json``)
and waits for the node's rounds in flight to commit. The data is any iterable, or a
``SharedQueue`` whose units the run's nodes claim through the mailbox (``queue`` module). When
training fails, the learner stops at once and releases its lease, so the other nodes do not
wait for it, and the failure is raised.

With ``ranks``, the nodes split the data in that many fixed slices: ``join`` takes the lowest
rank that no other node holds and that is not done, waiting while every one is, and ``rank``
tells the caller which slice to build its data from before ``run``. A rank is held through a
claim in the mailbox and the node's lease (``protocol`` module), so the node that takes over the
rank of one that crashed starts that slice over, and the run's data is exhausted once every
rank ever held is marked done (``done/{rank}.json``), an exclusive syncer waiting meanwhile for
a node to take over a crashed rank. A node that took the rank of one whose lease had lapsed,
slow rather than gone, loses it once that node renews its lease: ``run`` stops training at once
and raises ``RankLost``.

A node's ``role`` says which of the two parts it runs. A ``"both"`` node, the default, trains
and takes part in closing rounds; which node closes a round is settled through the mailbox
(``syncer`` module), so a run needs no separate process to operate. The node that closes a
round hands the commit to its own learner in memory; only the other nodes download it. A
``"learner"`` node only trains and never claims a round. A ``"syncer"`` node only closes rounds,
as the run's exclusive syncer: while its lease lives the other nodes leave every round to it and
take over only as from any other closer, once its lease ends or a round outlasts its timeout.
It never creates the run nor publishes a snapshot: ``join`` waits for a learner to create the
run, ``run`` ignores the step and the data and returns once the run is finished, no learner
holding a live lease and the data exhausted (``syncer`` module).

An ``"either"`` node trains as a ``"both"`` node until the run elects it its syncer. It stands
for election at ``join`` and on every poll of its syncer while no node owns the role and
another node trains (``syncer`` module), so a node alone never stops training. Elected, it
trains the unit in progress to its end, gives back the units it claimed and has not started, so
that the other nodes train them, closes its learner as at the end of its data but without
marking itself done, and from then on closes rounds as the run's exclusive syncer; ``run``
returns once the run is finished. It trains again, joining the run again as a learner, aligned
with the newest commits, and claiming units again, once a dedicated syncer owns the role, which
outranks it, or once no learner is left while units of its queue are unclaimed, which nobody
would train otherwise. A node that is not elected takes over the rounds of an elected syncer
that is gone, as from any closer, and stands for election again, so the run keeps one syncer
that does not train. Its data must be a ``SharedQueue``, the work an elected node can give back
and take up again; it takes no ``ranks``, which are for fixed slices of the data.
"""

import asyncio
import os
import re
import socket
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

from gradientmail.adapters import Adapter, Backend, adapter_for, backend_for, default_backend
from gradientmail.algorithms import Algorithm
from gradientmail.clock import MonotonicClock
from gradientmail.compression import Compression
from gradientmail.events import (
    BecameSyncer,
    Callback,
    ResumedTraining,
    SyncerStopped,
    combined,
    log_events,
)
from gradientmail.learner import Learner, check_schema
from gradientmail.learner import align as align_newest
from gradientmail.mailbox import Mailbox
from gradientmail.protocol import (
    Exchange,
    LocalCommits,
    ProtocolError,
    RunConfigMismatch,
    SyncerKind,
    rank_id,
)
from gradientmail.queue import Feed, SharedQueue
from gradientmail.syncer import Syncer

__all__ = ["Role", "Session", "align"]

type Role = Literal["learner", "syncer", "both", "either"]


def align(model: object, *, mailbox: Mailbox) -> int:
    """Load the run's newest committed parameters into ``model`` without joining the run.

    For evaluation or inference: nothing is published and no claim is made. Returns the newest
    committed round, -1 when the run has committed none yet; raises ``ProtocolError`` when
    ``mailbox`` holds no run and ``SchemaMismatch`` when ``model`` does not match it.
    """

    async def read() -> int:
        exchange = Exchange(mailbox, clock=MonotonicClock())
        manifest = await exchange.manifest()
        if manifest is None:
            raise ProtocolError("the mailbox holds no run")
        records = (await align_newest(adapter_for(model), exchange, manifest)).records
        return records[-1].round if records else -1

    return asyncio.run(read())


class Session:
    """One node of a run, running the caller's training step over its data.

    Constructing a session does no I/O. ``join`` joins the run, creating it from the model's
    parameters when it does not exist yet, and returns once the model holds the newest committed
    version of every fragment; the communication and syncer threads run from then on, so the
    node holds a lease while, say, it waits at a ``Barrier``. ``run`` joins first when the caller
    has not, trains through the data and returns once the node's last rounds are committed.

    Parameters
    ----------
    model
        A torch or MLX module, or any ``Adapter``. A syncer trains nothing, but the model's
        schema must still be the run's.
    algorithm
        The run's algorithm; its configuration must equal the one the run was created with.
    mailbox
        The store the run lives in.
    role
        What this node does. ``"both"`` trains and takes part in closing rounds, ``"learner"``
        only trains, ``"syncer"`` only closes rounds, as the run's exclusive syncer, and
        ``"either"`` trains as ``"both"`` until the run elects it its syncer, for which its data
        must be a ``SharedQueue`` (module docstring).
    node_id
        ``[a-z0-9-]{1,64}``. Defaults to the host name and process id, so two processes on one
        host never share an id.
    ranks
        How many fixed slices the run's data is split in, at most 1000, one node training each
        (module docstring); ``rank`` is this node's after ``join``. Every node of the run passes
        the same count. A ``"syncer"`` trains no slice and an ``"either"`` node trains a
        ``SharedQueue``: neither takes ranks (``ValueError``).
    poll_interval
        Seconds between polls for new round records and snapshots.
    backend
        Where this node's deltas and, when it closes rounds, outer steps are computed. Defaults
        to the model's framework, or to the installed one (torch, then MLX) when ``model`` is an
        ``Adapter``.
    compression
        The codec this node's payloads are stored with: ``"off"`` for safetensors as is, or
        ``"zstd"`` for byte-shuffled zstandard (the ``zstd`` extra), 16% smaller for fp32 and
        31% for bf16. Every node reads every codec; nodes of one run may differ.
    lease_every, lease_timeout
        Seconds between renewals of this node's lease, and seconds without a renewal after
        which the other nodes stop waiting for it. A round closes at ``quorum_size`` snapshots
        or once every live node has reported, and a node that finishes releases its lease at
        once, so nodes join and leave without stalling the run. A syncer's lease is renewed and
        judged alike: once it lapses, the other nodes close the rounds themselves.
    callbacks
        What receives the node's events (``events`` module), each in turn, on the thread that
        produced the event; one that raises is logged and does not affect the run or the other
        callbacks. The default logs them; leaving ``log_events`` out silences the node's log.
    """

    def __init__(
        self,
        model: object,
        *,
        algorithm: Algorithm,
        mailbox: Mailbox,
        role: Role = "both",
        node_id: str | None = None,
        ranks: int | None = None,
        poll_interval: float = 1.0,
        backend: Backend | None = None,
        compression: Compression = "off",
        lease_every: float = 60.0,
        lease_timeout: float = 120.0,
        callbacks: Sequence[Callback] = (log_events,),
    ) -> None:
        def default_node_id() -> str:
            pid = f"-{os.getpid()}"
            host = re.sub(r"[^a-z0-9-]", "-", socket.gethostname().lower())
            return f"{host[: 64 - len(pid)]}{pid}"

        def synchronize(syncer: Syncer) -> None:
            try:
                asyncio.run(syncer.run())
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                on_event(SyncerStopped(self._node_id, exc))
                self._failure = exc

        def learner() -> Learner:
            return Learner(
                adapter=self._adapter,
                algorithm=algorithm,
                mailbox=mailbox,
                node_id=self._node_id,
                backend=backend,
                poll_interval=poll_interval,
                lease_every=lease_every,
                lease_timeout=lease_timeout,
                compression=compression,
                local=local,
                ranks=ranks,
                on_event=on_event,
            )

        if ranks is not None:
            if role in ("syncer", "either"):
                raise ValueError(f"a {role!r} node takes no ranks")
            if not 1 <= ranks <= 1000:
                raise ValueError(f"ranks must be between 1 and 1000, got {ranks}")
        if backend is None:
            backend = default_backend() if isinstance(model, Adapter) else backend_for(model)
        self._adapter = model if isinstance(model, Adapter) else backend.adapter(model)
        self._algorithm = algorithm
        self._role = role
        self._node_id = node_id if node_id is not None else default_node_id()
        self._poll_interval = poll_interval
        on_event = combined(*callbacks)
        self._on_event = on_event
        local = LocalCommits() if role in ("both", "either") else None
        self._exchange = Exchange(mailbox, clock=MonotonicClock())
        self._new_learner = learner
        self._learner = None if role == "syncer" else learner()
        exclusive: SyncerKind | None = (
            "0-dedicated" if role == "syncer" else "1-elected" if role == "either" else None
        )
        self._syncer = (
            None
            if role == "learner"
            else Syncer(
                algorithm=algorithm,
                mailbox=mailbox,
                node_id=self._node_id,
                backend=backend,
                poll_interval=poll_interval,
                exclusive=exclusive,
                lease_every=lease_every,
                lease_timeout=lease_timeout,
                compression=compression,
                local=local,
                units=lambda: self._units,
                on_event=on_event,
            )
        )
        self._units: frozenset[str] = frozenset()
        self._failure: BaseException | None = None
        self._thread = (
            None
            if self._syncer is None
            else threading.Thread(
                target=synchronize,
                args=(self._syncer,),
                name=f"gradientmail-syncer-{self._node_id}",
                daemon=True,
            )
        )
        self._stage: Literal["new", "joined", "finished"] = "new"

    def join(self) -> None:
        """Create the run or adopt it, align the model with its newest commits, take a rank
        with ``ranks``, then start the communication and syncer threads.

        A syncer only adopts the run, waiting for a learner to create it, and leaves the model
        as it is. With ``ranks``, the node holds its lease from here on and waits while every
        rank is held or done; ``rank`` is set once it returns. An ``"either"`` node then stands for
        election unless a node owns the syncer role already or no other node trains, and when
        it wins stops training at once (module docstring). Raises ``SchemaMismatch`` or
        ``RunConfigMismatch`` when the model or the algorithm differs from the run's. Joining
        again does nothing.
        """
        if self._stage != "new":
            return
        learner, syncer, thread = self._learner, self._syncer, self._thread
        elected = False
        if learner is None:
            # The syncer thread starts first: a dedicated syncer holds its lease while it waits
            # for the run, so that the learner that creates the run leaves it round 0.
            if thread is not None:
                thread.start()
            try:
                asyncio.run(self._adopt())
            except BaseException:
                if syncer is not None:
                    syncer.stop()
                if thread is not None:
                    thread.join()
                raise
        else:
            learner.start()
            try:
                elected = (
                    self._role == "either" and syncer is not None and asyncio.run(syncer.elect())
                )
            except BaseException:
                learner.abort()
                raise
            if thread is not None:
                thread.start()
        self._stage = "joined"
        if elected and learner is not None:
            self._pause(learner)

    def run[W, Batch](
        self, step: Callable[[Batch], int], data: SharedQueue[W, Batch] | Iterable[Batch]
    ) -> None:
        """Train ``step`` over ``data``, then wait for this node's rounds in flight to commit.

        ``step`` trains the model on one batch, optimizer step included, and returns how many
        samples, tokens for a language model, it trained on. Between steps the session aligns the
        model with the commits that arrived and publishes the rounds that are due; it blocks
        only when the algorithm bounds ``overlap_steps``. A ``SharedQueue`` is exhausted once no
        unit is left to claim and its buffer is drained. Then the node, or its rank, is marked
        done and the learner closes (``Learner.close``): it waits for the rounds in flight, or
        with the algorithm's ``flush_on_close`` publishes what was trained since them first,
        and releases the lease; the syncer stops last, so a node alone still commits what it
        published.

        When ``step``, the data or the syncer raises, ``KeyboardInterrupt`` included, the learner
        stops at once without publishing or waiting (``Learner.abort``), which releases its
        lease, the syncer stops and the exception propagates. So it does when the node loses its
        rank, which raises ``RankLost`` and leaves the rank unmarked. A session runs once.

        A syncer never calls ``step`` nor reads ``data``, so one script serves every role: it
        closes rounds until the run is finished, no learner holding a live lease and the data
        exhausted, then releases its lease and returns. A ``KeyboardInterrupt`` stops it
        the same way and propagates, and so does a failure of the syncer.

        An ``"either"`` node trains as above until it is elected; then its learner closes as at
        the end of its data, but the node is not marked done, and it serves as the syncer until
        the run is finished, or until a dedicated syncer arrives or no learner is left for the
        units of its queue nobody claimed, when a new learner joins and training goes on
        (module docstring). Once its data is exhausted it leaves as any node does, the role
        given up if it holds it. Its data must be a ``SharedQueue``: ``ValueError`` otherwise.
        """
        if self._stage == "finished":
            raise RuntimeError("a Session runs once")
        if self._role == "either" and not isinstance(data, SharedQueue):
            raise ValueError("an 'either' node trains on a SharedQueue, whose units it gives back")
        if isinstance(data, SharedQueue):
            self._units = frozenset(data.units)
        self.join()
        try:
            if self._role == "syncer":
                if self._thread is not None:
                    self._thread.join()
                if self._failure is not None:
                    raise self._failure
                return
            learner = self._learner
            while True:
                if learner is None:
                    if not self._served():
                        return
                    learner = self._resume()
                if self._trained(step, data, learner):
                    rank = learner.rank
                    done = self._node_id if rank is None else rank_id(rank)
                    asyncio.run(self._exchange.mark_done(done))
                    learner.close()
                    return
                self._pause(learner)
                learner = None
        except BaseException:
            if self._learner is not None:
                self._learner.abort()
            raise
        finally:
            if self._syncer is not None:
                self._syncer.stop()
            if self._thread is not None:
                self._thread.join()
            self._stage = "finished"

    @property
    def rank(self) -> int | None:
        """The rank this node holds among the run's ``ranks``, from ``join`` on; None without
        ranks."""
        return None if self._learner is None else self._learner.rank

    @property
    def rounds_seen(self) -> int:
        """The newest round whose committed fragment the model holds; -1 on a fresh run, and for
        a syncer, whose model is never aligned, an ``"either"`` node serving as one included."""
        return -1 if self._learner is None else self._learner.rounds_seen

    @property
    def batch_size(self) -> int | None:
        """This node's share of the algorithm's global batch, for the training loop to size its
        batches by; it changes as learners join and leave. ``None`` when the algorithm sets
        none, and for a syncer, which trains nothing, an ``"either"`` node serving as one
        included."""
        return None if self._learner is None else self._learner.batch_size

    async def _adopt(self) -> None:
        """Wait for a learner to create the run; raise when the model or the algorithm is not
        the run's."""
        while (manifest := await self._exchange.manifest()) is None:
            await asyncio.sleep(self._poll_interval)
        check_schema(self._adapter.schema(), manifest)
        expected = self._algorithm.to_json()
        if dict(manifest.algorithm) != dict(expected):
            raise RunConfigMismatch(expected, manifest.algorithm)

    def _trained[W, Batch](
        self,
        step: Callable[[Batch], int],
        data: SharedQueue[W, Batch] | Iterable[Batch],
        learner: Learner,
    ) -> bool:
        """Train ``learner`` with ``step`` over ``data``; True once the data is exhausted.

        A shared queue stops short once this node's syncer is elected: at the end of the unit in
        progress, the units claimed and not started given back (``Feed``); then False.
        """

        def trained(batch: Batch) -> None:
            samples = step(batch)
            if self._failure is not None:
                raise self._failure
            learner.step(samples)

        match data:
            case SharedQueue():
                with Feed(
                    data, exchange=self._exchange, node_id=self._node_id, pause=self._elected
                ) as feed:
                    for _, batch in feed:
                        trained(batch)
                    if feed.paused:
                        feed.release()
                        return False
                return True
            case _:
                for batch in data:
                    trained(batch)
                return True

    def _elected(self) -> bool:
        return self._syncer is not None and self._syncer.holds_role

    def _pause(self, learner: Learner) -> None:
        """Stop training to serve as the elected syncer: the learner closes as at the end of its
        data, and the node is not marked done."""
        learner.close()
        self._learner = None
        self._on_event(BecameSyncer(self._node_id))

    def _served(self) -> bool:
        """Wait while this node holds the elected syncer's role: True once it gave the role up to
        another node, False once the syncer returned, the run finished. Raises what the syncer
        died of."""
        thread = self._thread
        while thread is not None and self._elected():
            thread.join(self._poll_interval)
            if not thread.is_alive():
                if self._failure is not None:
                    raise self._failure
                return False
        return True

    def _resume(self) -> Learner:
        """Train again under another node's syncer: a new learner joins the run."""
        learner = self._new_learner()
        self._learner = learner
        learner.start()
        self._on_event(ResumedTraining(self._node_id, learner.rounds_seen))
        return learner
