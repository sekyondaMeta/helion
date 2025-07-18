from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import unittest

import torch

import helion
from helion._testing import TestCase
from helion._testing import import_path
import helion.language as hl

if TYPE_CHECKING:
    from helion import Kernel

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"


def type_propagation_report(fn: Kernel, *args, ignore=False):
    return fn.bind(args)._debug_str()


class TestTypePropagation(TestCase):
    def test_add(self):
        output = type_propagation_report(
            basic_kernels.add,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_torch_ops_pointwise(self):
        output = type_propagation_report(
            basic_kernels.torch_ops_pointwise,
            torch.ones([1024], dtype=torch.int32),
            torch.ones([1024], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_all_ast_nodes(self):
        output = type_propagation_report(
            import_path(datadir / "all_ast_nodes.py").all_ast_nodes,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
            ignore=True,
        )
        self.assertExpectedJournal(output)

    def test_hl_zeros_usage(self):
        output = type_propagation_report(
            basic_kernels.hl_zeros_usage,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_hl_full_usage(self):
        output = type_propagation_report(
            basic_kernels.hl_full_usage,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_pointwise_device_loop(self):
        output = type_propagation_report(
            basic_kernels.pointwise_device_loop,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_method_call(self):
        @helion.kernel
        def fn(x):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile].sin()
            return out

        output = type_propagation_report(
            fn,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_matmul(self):
        output = type_propagation_report(
            import_path(examples_dir / "matmul.py").matmul,
            torch.ones([512, 512]),
            torch.ones([512, 512]),
        )
        self.assertExpectedJournal(output)


if __name__ == "__main__":
    unittest.main()
