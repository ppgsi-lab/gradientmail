"""The public facade: one object wraps a training loop from join to close, learner and syncer.

``Session`` resolves the framework backend from the model, names the node after the host and
process when the caller does not, joins the run before returning so the model holds the newest
global parameters by the time training starts, and then runs the learner's communication loop
and the syncer on two daemon threads. Every session takes part in closing rounds; which one
does the work is settled through the mailbox (``syncer`` module), so a run has no separate
process to operate.
"""

import asyncio
import os
import re
import socket
import threading
from collections.abc import Sequence
from typing import Self

from gradientmail.adapters import Adapter, Backend, adapter_for, backend_for, default_backend
from gradientmail.algorithms import Algorithm
from gradientmail.clock import MonotonicClock
from gradientmail.compression import Compression
from gradientmail.events import Callback, SyncerStopped, combined, log_events
from gradientmail.learner import Learner
from gradientmail.learner import align as align_newest
from gradientmail.mailbox import Mailbox
from gradientmail.protocol import Exchange, ProtocolError
from gradientmail.syncer import Syncer

__all__ = ["Session", "align"]


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
    """One node of a run, wrapped around the caller's own training loop.

    Construction joins the run, creating it from the model's parameters when it does not exist
    yet, and returns only once the model holds the newest committed version of every fragment;
    the communication and syncer threads run from then on. Call ``step`` after every optimizer
    step and ``close`` when training ends, or use the instance as a context manager. ``close``
    publishes what was trained since the last rounds and waits for those rounds to commit, so
    the run's parameters end with this node's last steps.

    Parameters
    ----------
    model
        A torch or MLX module, or any ``Adapter``.
    algorithm
        The run's algorithm; its configuration must equal the one the run was created with.
    mailbox
        The store the run lives in.
    learner_id
        ``[a-z0-9-]{1,64}``. Defaults to the host name and process id, so two processes on one
        host never share an id.
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
        or once every live node has reported, and a closing node releases its lease at once, so
        nodes join and leave without stalling the run.
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
        learner_id: str | None = None,
        poll_interval: float = 1.0,
        backend: Backend | None = None,
        compression: Compression = "off",
        lease_every: float = 60.0,
        lease_timeout: float = 120.0,
        callbacks: Sequence[Callback] = (log_events,),
    ) -> None:
        def default_learner_id() -> str:
            pid = f"-{os.getpid()}"
            host = re.sub(r"[^a-z0-9-]", "-", socket.gethostname().lower())
            return f"{host[: 64 - len(pid)]}{pid}"

        def synchronize() -> None:
            try:
                asyncio.run(self._syncer.run())
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                on_event(SyncerStopped(learner_id, exc))
                self._failure = exc

        if backend is None:
            backend = default_backend() if isinstance(model, Adapter) else backend_for(model)
        adapter = model if isinstance(model, Adapter) else backend.adapter(model)
        if learner_id is None:
            learner_id = default_learner_id()
        on_event = combined(*callbacks)
        self._learner = Learner(
            adapter=adapter,
            algorithm=algorithm,
            mailbox=mailbox,
            learner_id=learner_id,
            backend=backend,
            poll_interval=poll_interval,
            lease_every=lease_every,
            lease_timeout=lease_timeout,
            compression=compression,
            on_event=on_event,
        )
        self._syncer = Syncer(
            algorithm=algorithm,
            mailbox=mailbox,
            learner_id=learner_id,
            backend=backend,
            poll_interval=poll_interval,
            lease_timeout=lease_timeout,
            compression=compression,
            on_event=on_event,
        )
        self._failure: BaseException | None = None
        self._learner.start()
        self._thread = threading.Thread(
            target=synchronize, name=f"gradientmail-syncer-{learner_id}", daemon=True
        )
        self._thread.start()

    def step(self, *, samples: int) -> None:
        """Account for one training step of ``samples`` tokens; never awaits or does I/O.

        Blocks only when the algorithm bounds ``overlap_steps`` and a round published that many
        steps ago is still outstanding. Raises what either background thread died of.
        """
        if self._failure is not None:
            raise self._failure
        self._learner.step(samples)

    def close(self) -> None:
        """Wait for the learner's rounds in flight to commit (``Learner.close``), stop both
        threads.

        The syncer keeps closing rounds until the learner is done, so a node closing alone
        still commits what it published when quorum allows.
        """
        try:
            self._learner.close()
        finally:
            self._syncer.stop()
            self._thread.join()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def rounds_seen(self) -> int:
        """The newest round whose committed fragment the model holds; -1 on a fresh run."""
        return self._learner.rounds_seen

    @property
    def batch_size(self) -> int | None:
        """This node's share of the algorithm's global batch, for the training loop to size its
        batches by; it changes as nodes join and leave. ``None`` when the algorithm sets none."""
        return self._learner.batch_size
