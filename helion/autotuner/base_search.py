from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import functools
from itertools import starmap
import logging
import math
from math import inf
from multiprocessing import connection
import os
import random
import sys
import time
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import NoReturn

if TYPE_CHECKING:
    from triton.runtime.jit import JITFunction

import torch
import torch.multiprocessing as mp
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map
from triton.testing import do_bench

from .. import exc
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import make_precompiler
from .config_generation import ConfigGeneration
from .config_generation import FlatConfig
from .logger import LambdaLogger
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import triton

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from . import ConfigSpec


class BaseAutotuner(abc.ABC):
    """
    Abstract base class for all autotuners and classes that wrap autotuners, like caching.
    """

    @abc.abstractmethod
    def autotune(self) -> Config:
        raise NotImplementedError


class BaseSearch(BaseAutotuner):
    """
    Base class for search algorithms. This class defines the interface and utilities for all
    search algorithms.

    Attributes:
        kernel (BoundKernel): The kernel to be tuned.
        settings (Settings): The settings associated with the kernel.
        config_spec (ConfigSpec): The configuration specification for the kernel.
        args (Sequence[object]): The arguments to be passed to the kernel.
        counters (collections.Counter): A counter to track various metrics during the search.
    """

    def __init__(self, kernel: BoundKernel, args: Sequence[object]) -> None:
        """
        Initialize the BaseSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
        """
        super().__init__()
        self.kernel = kernel
        self.settings: Settings = kernel.settings
        self.config_spec: ConfigSpec = kernel.config_spec
        self.args: Sequence[object] = args
        self.counters: collections.Counter[str] = collections.Counter()
        self.log = LambdaLogger(self.settings.autotune_log_level)
        seed = self.settings.autotune_random_seed
        random.seed(seed)
        self.log(f"Autotune random seed: {seed}")
        self._original_args: Sequence[object] = self._clone_args(self.args)
        (
            self._baseline_output,
            self._kernel_mutates_args,
            self._baseline_post_args,
        ) = self._compute_baseline()

    def _clone_args(self, args: Sequence[object]) -> Sequence[object]:
        def _clone_leaf(leaf: object) -> object:
            if isinstance(leaf, torch.Tensor):
                clone = leaf.detach().clone()
                clone.requires_grad_(leaf.requires_grad)
                return clone
            return leaf

        return tree_map(_clone_leaf, args)

    def _compute_baseline(self) -> tuple[object, bool, Sequence[object] | None]:
        """
        Return output and post-run input arguments of the default-config kernel.
        Also detect if the kernel mutates any of its input arguments.
        """
        new_args = self._clone_args(self._original_args)
        baseline_config = self.config_spec.default_config()
        baseline_output = self.kernel.compile_config(
            baseline_config, allow_print=False
        )(*new_args)
        original_args_flat, _ = tree_flatten(self._original_args)
        new_args_flat, _ = tree_flatten(new_args)
        mutated = False
        for old, new in zip(original_args_flat, new_args_flat, strict=False):
            if (
                isinstance(old, torch.Tensor)
                and isinstance(new, torch.Tensor)
                and (not torch.equal(new, old))
            ):
                mutated = True
                break
        baseline_post_args = self._clone_args(new_args)
        return baseline_output, mutated, baseline_post_args

    def _validate_against_baseline(
        self, config: Config, output: object, args: Sequence[object]
    ) -> bool:
        try:
            torch.testing.assert_close(output, self._baseline_output)
            if self._kernel_mutates_args:
                torch.testing.assert_close(args, self._baseline_post_args)
        except AssertionError as e:
            self.counters["accuracy_mismatch"] += 1
            self.log.warning(f"Accuracy mismatch for {config!r}: {e!s}")
            return False
        return True

    def benchmark(self, config: Config) -> float:
        """
        Benchmark a specific configuration.

        This method compiles the kernel with the given configuration and measures its performance.

        Args:
            config: The configuration to benchmark.

        Returns:
            The performance of the configuration in seconds.
        """
        fn = self.kernel.compile_config(config, allow_print=False)
        if self.start_precompile_and_check_for_hangs(config, fn)():
            return self.benchmark_function(config, fn)
        return inf

    def benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """
        Benchmark a compiled function.  This function is called by the autotuner to measure the
        performance of a specific configuration.

        Args:
            config: The configuration to benchmark.
            fn: A precompiled version of config.

        Returns:
            The performance of the configuration in seconds.
        """
        self.counters["benchmark"] += 1
        self.log.debug(lambda: f"Running benchmark for {config!r}")
        try:
            # TODO(jansel): early exit with fewer trials if early runs are slow
            t0 = time.perf_counter()
            if self._kernel_mutates_args:
                self.args = self._clone_args(self._original_args)
            output = fn(*self.args)  # make sure the kernel is compiled
            if not self._validate_against_baseline(config, output, self.args):
                # Accuracy check failed; reject this config
                return inf
            t1 = time.perf_counter()
            res = do_bench(
                functools.partial(fn, *self.args),
                return_mode="median",
            )
            t2 = time.perf_counter()
            self.log.debug(
                lambda: f"result: {res:.4f}ms (took {t1 - t0:.1f}s + {t2 - t1:.1f}s)",
            )
            return res  # pyright: ignore[reportReturnType]
        except Exception as e:
            action = classify_triton_exception(e)
            if action == "raise":
                raise exc.TritonError(
                    f"{type(e).__qualname__}: {e}",
                    self.kernel.format_kernel_decorator(config, self.settings),
                ) from e
            if action == "warn":
                self.log.warning(format_triton_compile_failure(config, e))
            else:
                self.log.debug(f"Benchmarking failed: {type(e).__name__}: {e}")
            return inf

    def start_precompile_and_check_for_hangs(
        self, config: Config, fn: CompiledConfig
    ) -> PrecompileFuture:
        """
        Unfortunately, Triton can hang when compiling a kernel. This function tries to
        compile the kernel with the given configuration and checks if it hangs in a subprocess.
        We also do this in parallel (when called from parallel_benchmark) to do faster autotuning.
        Note that we compile in parallel, but we benchmark one-by-one to avoid noisy results.

        Args:
            config: The config that generated fn.
            fn: The function to be precompiled.

        Returns:
            True if the compilation was successful, False if it hung.
        """
        if not self.settings.autotune_precompile:
            return PrecompileFuture.skip(self, config, True)
        ctx = mp.get_context("fork")

        def extract_launcher(
            triton_kernel: triton.JITFunction,
            grid: tuple[int, ...],
            *args: object,
            **kwargs: object,
        ) -> NoReturn:
            """Custom launcher that extracts arguments instead of executing."""
            raise _ExtractedLaunchArgs(triton_kernel, grid, args, kwargs)

        try:
            # Call main function with extraction launcher to extract arguments
            fn(*self.args, _launcher=extract_launcher)
            # Should not reach here
            raise RuntimeError("Expected _ExtractedLaunchArgs exception")
        except _ExtractedLaunchArgs as e:
            precompiler = make_precompiler(e.kernel, config)(*e.args, **e.kwargs)
            if precompiler is already_compiled:
                return PrecompileFuture.skip(self, config, True)
        except Exception:
            log.warning(
                "Helion autotuner precompile error for %s",
                self.kernel.format_kernel_decorator(config, self.settings),
                exc_info=True,
            )
            raise
        process: mp.Process = ctx.Process(target=precompiler)  # pyright: ignore[reportAssignmentType]
        return PrecompileFuture(
            search=self,
            config=config,
            process=process,
            timeout=self.settings.autotune_compile_timeout,
        )

    def parallel_benchmark(self, configs: list[Config]) -> list[tuple[Config, float]]:
        """
        Benchmark multiple configurations in parallel.

        Args:
            configs: A list of configurations to benchmark.

        Returns:
            A list of tuples containing configurations and their performance.
        """
        fns = [self.kernel.compile_config(c, allow_print=False) for c in configs]
        if self.settings.autotune_precompile:
            is_workings = PrecompileFuture.wait_for_all(
                [
                    *starmap(
                        self.start_precompile_and_check_for_hangs,
                        zip(configs, fns, strict=True),
                    )
                ]
            )
        else:
            is_workings = [True] * len(configs)
        results = []
        for config, fn, is_working in zip(configs, fns, is_workings, strict=True):
            if is_working:
                # benchmark one-by-one to avoid noisy results
                results.append((config, self.benchmark_function(config, fn)))
            else:
                results.append((config, inf))
        return results

    def autotune(self) -> Config:
        """
        Perform autotuning to find the best configuration.

        This method searches for the optimal configuration by benchmarking multiple configurations.

        Returns:
            The best configuration found during autotuning.
        """
        start = time.perf_counter()
        self.log.reset()
        best = self._autotune()
        end = time.perf_counter()
        kernel_decorator = self.kernel.format_kernel_decorator(best, self.settings)
        self.log(
            f"Autotuning complete in {end - start:.1f}s after searching {self.counters['benchmark']} configs.\n"
            "One can hardcode the best config and skip autotuning with:\n"
            f"    {kernel_decorator}\n",
            level=logging.INFO + 5,
        )
        if self.settings.print_output_code:
            triton_code = self.kernel.to_triton_code(best)
            print(triton_code, file=sys.stderr)
        return best

    def _autotune(self) -> Config:
        """
        Abstract method to perform the actual autotuning.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError


class PopulationMember(NamedTuple):
    """
    Represents a member of the population in population-based search algorithms.

    Attributes:
        perf (float): The performance of the configuration.
        flat_values (FlatConfig): The flat representation of the configuration values.
        config (Config): The full configuration object.
    """

    perf: float
    flat_values: FlatConfig
    config: Config


def performance(member: PopulationMember) -> float:
    """
    Retrieve the performance of a population member.  Used as a sort key.

    Args:
        member: The population member.

    Returns:
        The performance of the member.
    """
    return member.perf


class PopulationBasedSearch(BaseSearch):
    """
    Base class for search algorithms that use a population of configurations.

    Attributes:
        population (list[PopulationMember]): The current population of configurations.
        flat_spec (list[ConfigSpecFragment]): The flattened configuration specification.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
    ) -> None:
        """
        Initialize the PopulationBasedSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
        """
        super().__init__(kernel, args)
        self.population: list[PopulationMember] = []
        self.config_gen: ConfigGeneration = ConfigGeneration(self.config_spec)

    @property
    def best(self) -> PopulationMember:
        """
        Retrieve the best configuration in the population.

        Returns:
            The best population member.
        """
        return min(self.population, key=performance)

    def benchmark_flat(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Benchmark a flat configuration.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with the benchmark results.
        """
        config = self.config_gen.unflatten(flat_values)
        return PopulationMember(self.benchmark(config), flat_values, config)

    def parallel_benchmark_flat(
        self, to_check: list[FlatConfig]
    ) -> list[PopulationMember]:
        """
        Benchmark multiple flat configurations in parallel.

        Args:
            to_check: A list of flat configurations to benchmark.

        Returns:
            A list of population members with the benchmark results.
        """
        configs = [*map(self.config_gen.unflatten, to_check)]
        result = []
        for flat_values, config_in, (config_out, perf) in zip(
            to_check, configs, self.parallel_benchmark(configs), strict=True
        ):
            assert config_in is config_out
            result.append(PopulationMember(perf, flat_values, config_in))
        return result

    def statistics(self) -> str:
        """
        Generate statistics for the current population.

        Returns:
            A string summarizing the population performance.
        """
        return population_statistics(self.population)


def population_statistics(population: list[PopulationMember]) -> str:
    """
    Create a summary of the population performance.

    Args:
        population: The population of configurations.

    Returns:
        A string summarizing the performance of the population.
    """
    population = sorted(population, key=performance)
    if math.isinf(population[-1].perf):
        working = [x for x in population if not math.isinf(x.perf)]
        if len(working) == 0:
            raise exc.NoConfigFound
        return (
            f"failed={len(population) - len(working)} "
            f"min={working[0].perf:.4f} "
            f"mid={working[len(working) // 2].perf:.4f} "
            f"max={working[-1].perf:.4f} "
            f"best={population[0].config!s}"
        )
    return (
        f"min={population[0].perf:.4f} "
        f"mid={population[len(population) // 2].perf:.4f} "
        f"max={population[-1].perf:.4f} "
        f"best={population[0].config!s}"
    )


@dataclasses.dataclass
class PrecompileFuture:
    """
    Wraps a child process where we are precompiling a kernel.

    Attributes:
        search (BaseSearch): The search object that initiated the precompilation.
        config (Config): The configuration to be precompiled.
        process (mp.Process | None): The process running the precompilation.
        timeout (float): The timeout for the precompilation.
        start_time (float): The time when the precompilation started.
        end_time (float | None): The time when the precompilation ended.
        ok (bool | None): The result of the precompilation (True if successful, False otherwise).
    """

    search: BaseSearch
    config: Config
    process: mp.Process | None
    timeout: float
    # Set when the process is actually started. For queued futures this is None.
    start_time: float | None = None
    end_time: float | None = None
    ok: bool | None = None

    @property
    def elapsed(self) -> float:
        """Return the elapsed time since the start of the precompilation."""
        if self.start_time is None:
            return 0.0
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def seconds_left(self) -> float:
        """Return the number of seconds left before the timeout."""
        if self.end_time is not None:
            return 0
        if self.start_time is None:
            return self.timeout
        return self.timeout - (time.time() - self.start_time)

    def is_alive(self) -> bool:
        """Check if the precompilation process is still alive."""
        if (p := self.process) is None:
            return False
        return p.is_alive()

    @property
    def started(self) -> bool:
        """Whether the process has been started."""
        return self.start_time is not None

    def start(self) -> None:
        """Start the underlying process and set the timer if not already started."""
        if self.process is None or self.started:
            return
        self.start_time = time.time()
        self.process.start()

    @staticmethod
    def skip(search: BaseSearch, config: Config, ok: bool) -> PrecompileFuture:
        """Dummy precompile future that is already done."""
        ts = time.time()
        return PrecompileFuture(
            search=search,
            config=config,
            process=None,
            timeout=0,
            ok=ok,
            start_time=ts,
            end_time=ts,
        )

    def __call__(self) -> bool:
        """Wait for the precompilation to finish and return true on success."""
        if self.ok is not None:
            return self.ok
        process = self.process
        assert process is not None
        try:
            # Start now if not already started (single-future path)
            if not self.started:
                self.start()
            process.join(self.seconds_left())
        finally:
            self._mark_complete()
        assert self.ok is not None
        return self.ok

    @staticmethod
    def wait_for_all(
        futures: list[PrecompileFuture],
    ) -> list[bool]:
        """
        Wait for all precompile futures to complete.

        Args:
            futures: A list of PrecompileFuture objects.

        Returns:
            A list of boolean values indicating completion status.
        """
        remaining = [f for f in futures if f.ok is None]
        try:
            while remaining:
                remaining = PrecompileFuture._wait_for_all_step(remaining)
        except Exception:
            for f in remaining:
                if (p := f.process) is not None:
                    with contextlib.suppress(Exception):
                        p.terminate()
            raise
        result = []
        for f in futures:
            assert f.ok is not None
            result.append(f.ok)
        return result

    @staticmethod
    def _wait_for_all_step(
        futures: list[PrecompileFuture],
    ) -> list[PrecompileFuture]:
        """Start up to the concurrency cap, wait for progress, and return remaining futures."""
        # Concurrency cap from the settings of the first future's search
        cap = futures[0].search.settings.autotune_precompile_jobs or os.cpu_count() or 1
        running = [f for f in futures if f.started and f.ok is None and f.is_alive()]

        # Start queued futures up to the cap
        queued = collections.deque(f for f in futures if not f.started and f.ok is None)
        while len(running) < cap and queued:
            job = queued.popleft()
            job.start()
            if job.is_alive():
                running.append(job)

        # Wait for at least one to finish or time out
        timeout = min([f.seconds_left() for f in running], default=0.0)
        handles = [f.process.sentinel for f in running]  # pyright: ignore[reportOptionalMemberAccess]
        if handles and timeout > 0:
            connection.wait(handles, timeout)
        remaining: list[PrecompileFuture] = []
        for f in futures:
            if f.ok is not None:
                continue
            if f.started and (not f.is_alive() or f.seconds_left() <= 0):
                f._mark_complete()
            else:
                remaining.append(f)
        return remaining

    def _mark_complete(self) -> bool:
        """
        Mark the precompile future as complete and kill the process if needed.

        Returns:
            True if the precompilation was successful, False otherwise.
        """
        self.end_time = time.time()
        process = self.process
        assert process is not None
        # If the process hasn't been started yet (shouldn't happen in normal flow),
        # start and immediately terminate to maintain invariants.
        if not self.started:
            self.start()
        if not process.is_alive():
            self.ok = process.exitcode == 0
            return self.ok
        process.terminate()
        process.join(10)
        msg = f"Timeout after {self.elapsed:.0f}s compiling {self.config}"
        if process.is_alive():
            self.search.log.warning(
                msg,
                "(SIGKILL required)",
            )
            process.kill()
            process.join()
        else:
            self.search.log.warning(msg)

        self.ok = False
        return False


class _ExtractedLaunchArgs(Exception):
    """Exception that carries kernel launch arguments for precompiler extraction."""

    kernel: JITFunction[object]
    grid: object
    args: tuple[object, ...]
    kwargs: dict[str, object]

    def __init__(
        self,
        triton_kernel: JITFunction[object],
        grid: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        super().__init__()
        self.kernel = triton_kernel
        self.grid = grid
        self.args = args
        self.kwargs = kwargs
