from __future__ import annotations

import itertools
import logging
import re
import sys
import time
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal

from torch._inductor.runtime.triton_compat import OutOfResources
from torch._inductor.runtime.triton_compat import PTXASError

if TYPE_CHECKING:
    from ..runtime.config import Config


class LambdaLogger:
    """
    A self-contained logger that does not propagate to the root logger and
    prints each record to stderr in the form:

        [<elapsed>s] <message>

    where *elapsed* is the whole-second wall-clock time since the logger
    instance was created.

    Takes lambdas as arguments, which are called when the log is emitted.
    """

    _count: itertools.count[int] = itertools.count()

    def __init__(self, level: int) -> None:
        self.level = level
        self._logger: logging.Logger = logging.getLogger(
            f"{__name__}.{next(self._count)}"
        )
        self._logger.setLevel(level)
        self._logger.propagate = False
        self.reset()

    def reset(self) -> None:
        self._logger.handlers.clear()
        self._logger.addHandler(_make_handler())

    def __call__(
        self, *msg: str | Callable[[], str], level: int = logging.INFO
    ) -> None:
        """
        Log a message at a specified log level.

        Args:
            msg: The message(s) to log. Can be strings or callables that return strings.
            level: The log level for the message.
        """
        if level >= self.level:
            self._logger.log(level, " ".join(map(_maybe_call, msg)))

    def warning(self, *msg: str | Callable[[], str]) -> None:
        return self(*msg, level=logging.WARNING)

    def debug(self, *msg: str | Callable[[], str]) -> None:
        return self(*msg, level=logging.DEBUG)


def _make_handler() -> logging.Handler:
    start = time.perf_counter()

    class _ElapsedFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            elapsed = int(time.perf_counter() - start)
            return f"[{elapsed}s] {record.getMessage()}"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ElapsedFormatter())
    return handler


def _maybe_call(fn: Callable[[], str] | str) -> str:
    """
    Call a callable or return the string directly.

    Args:
        fn: A callable that returns a string or a string.

    Returns:
        The resulting string.
    """
    if callable(fn):
        return fn()
    return fn


def format_triton_compile_failure(config: Config, err: BaseException) -> str:
    return (
        "Triton compile failed. This likely indicates a bug in Triton. "
        "Skipping failing config.\n"
        f"Config: {config!r}\n"
        f"Error: {type(err).__name__}: {err}"
    )


# Common logic to decide how to surface Triton errors
_EXPECTED_TRITON_ERRORS_RE: re.Pattern[str] = re.compile(
    "|".join(
        map(
            re.escape,
            [
                "[CUDA]: invalid argument",  # CUDA Error
                "PassManager::run failed",  # Triton Error
                "TServiceRouterException",  # Remote compile failed
            ],
        )
    )
)


def classify_triton_exception(err: BaseException) -> Literal["raise", "warn", "debug"]:
    """
    Classify a Triton compile/runtime exception during autotuning.

    Returns one of:
      - "raise": unexpected error, caller should raise
      - "warn": notable expected error (e.g., PassManager pipeline failure)
      - "debug": benign/expected error; caller can log at debug level
    """
    # Known exception types first
    if isinstance(err, OutOfResources):
        return "debug"
    # Different PTXASError classes may be raised from different modules; match by name as well
    if isinstance(err, PTXASError) or err.__class__.__name__ == "PTXASError":
        return "warn"

    msg = str(err)
    if "PassManager::run failed" in msg:
        return "warn"
    if _EXPECTED_TRITON_ERRORS_RE.search(msg):
        return "debug"
    return "raise"
