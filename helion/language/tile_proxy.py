from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from typing import Protocol
from typing import Sequence
from typing import TypeVar
from typing import cast
from typing_extensions import Self

import torch
from torch.utils._pytree import tree_map_only

from .. import exc
from .._compiler.compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable

    _T = TypeVar("_T")

    class _TLS(Protocol):
        index_calls: _CheckForIndexCalls | None


_tls: _TLS = cast("_TLS", threading.local())


class Tile(torch.Tensor):
    """
    This class should not be instantiated directly, it is the result of
    hl.tile(...) and represents a single tile of the iteration space.

    Tile's can be used as indices to tensors, e.g. `tensor[tile]`.  Tile's
    can also be use as sizes for allocations, e.g. `torch.empty([tile])`.
    There are also properties such as :meth:`tile.index <index>`, :meth:`tile.begin <begin>`,
    :meth:`tile.end <end>`, :meth:`tile.id <id>` and :meth:`tile.block_size <block_size>` that can be used to retrieve various
    information about the tile.

    Masking is implicit for tiles, so if the final tile is smaller than
    the block size loading that tile will only load the valid elements
    and reduction operations know to ignore the invalid elements.
    """

    def __init__(self, block_id: int) -> None:
        super().__init__()
        self.block_id = block_id

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        from ..language.memory_ops import load
        from ..language.memory_ops import store

        if func is torch.Tensor.__getitem__:
            if len(args) != 2 or kwargs:
                raise exc.IncorrectTileUsage(func)
            tensor, index = args
            assert isinstance(tensor, torch.Tensor)
            return load(tensor, cls._prepare_index(index))
        if func is torch.Tensor.__setitem__:
            if len(args) != 3 or kwargs:
                raise exc.IncorrectTileUsage(func)
            tensor, index, value = args
            assert isinstance(tensor, torch.Tensor)
            assert isinstance(value, torch.Tensor)
            return store(tensor, cls._prepare_index(index), value)
        if (
            func is torch.Tensor.__index__
            and (index_calls := getattr(_tls, "index_calls", None)) is not None
        ):
            index_calls.count += 1
        if func is torch.Tensor.__format__:
            return repr(args[0])
        raise exc.IncorrectTileUsage(func)

    @staticmethod
    def _prepare_index(index: object) -> list[object]:
        if isinstance(index, (list, tuple)):
            return [*index]
        assert isinstance(index, Tile)
        return [index]

    def __repr__(self, tensor_contents: None = None) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"Tile({self.block_id!r})"

    @classmethod
    def _tiles_to_sizes(cls, it: _T) -> _T:
        return tree_map_only(Tile, cls._tile_to_size, it)

    @staticmethod
    def _tile_to_size(x: Tile) -> torch.SymInt:
        return CompileEnvironment.current().block_sizes[x.block_id].var

    @property
    def index(self) -> torch.Tensor:
        """
        Alias for :func:`~helion.language.tile_index`, which retrieves a tensor containing the offsets for a tile.
        """
        from .tile_ops import tile_index

        return tile_index(self)

    @property
    def begin(self) -> int:
        """
        Alias for :func:`~helion.language.tile_begin`, which retrieves the start offset of a tile.
        """
        from .tile_ops import tile_begin

        return tile_begin(self)

    @property
    def end(self) -> int:
        """
        Alias for :func:`~helion.language.tile_end`, which retrieves the end offset of a tile.
        """
        from .tile_ops import tile_end

        return tile_end(self)

    @property
    def block_size(self) -> int:
        """
        Alias for :func:`~helion.language.tile_block_size`, which retrieves the block_size of a tile.
        """
        from .tile_ops import tile_block_size

        return tile_block_size(self)

    @property
    def id(self) -> int:
        """
        Alias for :func:`~helion.language.tile_id`, which retrieves the id of a tile.
        """
        from .tile_ops import tile_id

        return tile_id(self)


class _CheckForIndexCalls:
    """
    Unfortunately, the `__torch_function__` method of `TileIndexProxy` does not work
    properly when operations like view() are called on a `TileIndexProxy` object.  It calls
    `__torch_function__(Tensor.__index__, ...)` but then discards the result because it is not
    an integer (if a SymInt is returned).

    This class is a workaround to detect this case and turn tiles to sizes in the caller.
    """

    @classmethod
    def retry_call(
        cls,
        fn: Callable[..., object],
        proxy_args: Sequence[object],
        proxy_kwargs: dict[str, object],
    ) -> object:
        index_calls = cls()
        try:
            with index_calls:
                return fn(*proxy_args, **proxy_kwargs)
        except TypeError:
            if index_calls.count == 0:
                raise
        # This is likely a view op, try again with tiles_to_sizes
        proxy_args = Tile._tiles_to_sizes(proxy_args)
        proxy_kwargs = Tile._tiles_to_sizes(proxy_kwargs)
        return fn(*proxy_args, **proxy_kwargs)

    def __init__(self) -> None:
        self.count = 0

    def __enter__(self) -> Self:
        assert getattr(_tls, "index_calls", None) is None
        _tls.index_calls = self
        return self

    def __exit__(self, *args: object) -> None:
        _tls.index_calls = None
