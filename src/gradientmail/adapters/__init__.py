"""Framework backends: how a model's parameters become bytes, and how bytes become arithmetic.

A ``Backend`` is everything GradientMail knows about one framework. On a learner it wraps a
model in an ``Adapter``, the three-method byte boundary (``schema``, ``snapshot``, ``apply``);
on a syncer it loads ``TensorBytes`` as ``Value``s, the handful of fp32 operations the outer
step is made of, which serialize themselves back. ``TorchBackend`` and ``MlxBackend`` live
in their own modules and are imported only when asked for, so this package needs nothing
beyond the standard library. Validation against the schema is the
caller's job (``Learner`` before every ``apply``, ``Syncer`` before merging), so adapters
stay thin.
"""

from collections.abc import Iterable, Mapping
from importlib.util import find_spec
from typing import Protocol, Self, runtime_checkable

from gradientmail.serialization import DType, ParameterSpec, Schema, TensorBytes

__all__ = [
    "Adapter",
    "Backend",
    "MemoryAdapter",
    "MissingExtra",
    "Value",
    "adapter_for",
    "backend_for",
    "default_backend",
]


class MissingExtra(ImportError):
    """An optional dependency is not installed; ``extra`` names the extra that provides it."""

    extra: str

    def __init__(self, extra: str) -> None:
        self.extra = extra
        super().__init__(f"this needs the {extra!r} extra: pip install 'gradientmail[{extra}]'")


@runtime_checkable
class Adapter(Protocol):
    def schema(self) -> Schema:
        """Canonical trainable parameters, sorted by name."""
        ...

    def snapshot(self, names: Iterable[str]) -> Mapping[str, TensorBytes]:
        """Owned copies of ``names``, consistent at call time."""
        ...

    def apply(self, tensors: Mapping[str, TensorBytes]) -> None:
        """Overwrite parameters in place; never rebinds them, so optimizer state survives."""
        ...


@runtime_checkable
class Value(Protocol):
    """An fp32 tensor held by a backend, with the arithmetic the outer step is made of.

    Every operation returns a new value of the same backend. ``norm`` is the Euclidean norm
    over all elements; ``to_bytes`` rounds to ``dtype`` (nearest even) and serializes.
    """

    def __add__(self, other: Self) -> Self: ...

    def __sub__(self, other: Self) -> Self: ...

    def __mul__(self, scalar: float) -> Self: ...

    def norm(self) -> float: ...

    def to_bytes(self, dtype: DType) -> TensorBytes: ...


@runtime_checkable
class Backend(Protocol):
    """One framework: it adapts that framework's models and does the syncer's arithmetic."""

    def adapter(self, model: object) -> Adapter:
        """An Adapter over ``model``; ``TypeError`` when it is not this framework's model."""
        ...

    def load(self, tensor: TensorBytes) -> Value:
        """``tensor`` as an fp32 value of its shape."""
        ...


class MemoryAdapter(Adapter):
    """An Adapter over a dict of ``TensorBytes``: the Phase 2 learner and the fake in tests."""

    def __init__(self, tensors: Mapping[str, TensorBytes]) -> None:
        self._tensors = dict(tensors)

    @property
    def tensors(self) -> Mapping[str, TensorBytes]:
        return self._tensors

    def schema(self) -> Schema:
        return Schema(
            tuple(
                ParameterSpec(name, t.dtype, t.shape) for name, t in sorted(self._tensors.items())
            )
        )

    def snapshot(self, names: Iterable[str]) -> Mapping[str, TensorBytes]:
        return {name: self._tensors[name] for name in names}

    def apply(self, tensors: Mapping[str, TensorBytes]) -> None:
        self._tensors.update(tensors)


def backend_for(model: object) -> Backend:
    """The backend of ``model``'s framework, read from the modules of ``type(model)``'s MRO.

    A user's ``class MLP(nn.Module)`` lives in the user's module, so the whole MRO is searched;
    no framework is imported to decide. Anything else is a ``TypeError``.
    """
    roots = {cls.__module__.partition(".")[0] for cls in type(model).__mro__}
    if "torch" in roots:
        from gradientmail.adapters.torch import TorchBackend

        return TorchBackend()
    if "mlx" in roots:
        from gradientmail.adapters.mlx import MlxBackend

        return MlxBackend()
    raise TypeError(
        f"no backend for {type(model).__module__}.{type(model).__qualname__}; "
        "pass a torch or MLX module, or an Adapter"
    )


def adapter_for(model: object) -> Adapter:
    """``model`` itself when it is an Adapter; otherwise ``backend_for(model).adapter(model)``."""
    if isinstance(model, Adapter):
        return model
    return backend_for(model).adapter(model)


def default_backend() -> Backend:
    """The installed backend: torch, then MLX. ``MissingExtra("torch")`` when neither is."""
    if find_spec("torch") is not None:
        from gradientmail.adapters.torch import TorchBackend

        return TorchBackend()
    if find_spec("mlx") is not None:
        from gradientmail.adapters.mlx import MlxBackend

        return MlxBackend()
    raise MissingExtra("torch")
