"""Component registry for dynamic creation of datasets, methods, missingness generators, etc."""

from __future__ import annotations

from typing import Any


class Registry:
    """A generic registry that maps string names to classes/factories."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type] = {}

    def register(self, name: str, cls: type | None = None):
        """Register a class. Can be used as a decorator or called directly.

        Usage:
            @registry.register("mean")
            class MeanImputer: ...

            # or
            registry.register("mean", MeanImputer)
        """
        if cls is not None:
            self._registry[name] = cls
            return cls

        def decorator(cls_inner: type) -> type:
            self._registry[name] = cls_inner
            return cls_inner

        return decorator

    def get(self, name: str) -> type:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in {self._name} registry. Available: [{available}]"
            )
        return self._registry[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered component by name with given kwargs."""
        cls = self.get(name)
        return cls(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={self.list()})"


# Global registries
DATASET_REGISTRY = Registry("dataset")
METHOD_REGISTRY = Registry("method")
MISSINGNESS_REGISTRY = Registry("missingness")
TRACKER_REGISTRY = Registry("tracker")
DOWNSTREAM_REGISTRY = Registry("downstream")
